"""Утилиты для подготовки локальных моделей ChatKodi.

Этот скрипт анализирует железо хоста, выбирает подходящий вариант LLM,
скачивает нужные ресурсы в локальный кэш и проверяет, что релиз можно
запустить без обращения к сети.
"""

import os
import json
import sys
import time
import hashlib
import builtins as _builtins
import re
import psutil
import platform
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
from typing import Optional, Tuple, Dict, List
import warnings

from core.paths import model_cache_search_dirs, resolve_model_cache_dir, resolve_project_path
from core.atomic_io import atomic_write_json, read_json_with_backup
from core.logging_utils import get_logger
from core.model_integrity import (
    MODEL_HASH_MANIFEST,
    create_model_hash_manifest,
    inspect_hf_model_payload,
    verify_model_hash_manifest,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def _safe_console_render(text: str, encoding: Optional[str]) -> str:
    if not encoding:
        return text
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def print(*args, **kwargs):
    file = kwargs.get("file", sys.stdout)
    if file is None:
        return

    if file in (sys.stdout, sys.stderr):
        kwargs["sep"] = _safe_console_render(str(kwargs.get("sep", " ")), getattr(file, "encoding", None))
        kwargs["end"] = _safe_console_render(str(kwargs.get("end", "\n")), getattr(file, "encoding", None))
        args = tuple(
            _safe_console_render(str(arg), getattr(file, "encoding", None))
            for arg in args
        )

    return _builtins.print(*args, **kwargs)


def _normalize_model_ref(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _model_name_from_path(model_path: str) -> str:
    path_obj = Path(model_path)
    candidates = [path_obj.name]

    offline_config = path_obj / "offline_config.json"
    if offline_config.exists():
        try:
            with open(offline_config, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("model_name"):
                candidates.append(str(data["model_name"]))
        except Exception:
            pass

    config_json = path_obj / "config.json"
    if config_json.exists():
        try:
            with open(config_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("_name_or_path"):
                candidates.append(str(data["_name_or_path"]))
        except Exception:
            pass

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate:
            return candidate
    return path_obj.name


def _model_size_from_ref(*values: str) -> Optional[str]:
    joined = " ".join(str(value or "") for value in values).lower()
    for marker, label in (
        ("0.5b", "0.5B"),
        ("1.5b", "1.5B"),
        ("3b", "3B"),
        ("4b", "4B"),
        ("7b", "7B"),
    ):
        if marker in joined:
            return label
    return None


def _path_matches_model_name(model_name: str, model_path: str) -> bool:
    requested = _normalize_model_ref(model_name.split("/")[-1])
    actual = _normalize_model_ref(_model_name_from_path(model_path).split("/")[-1])
    if not requested or not actual:
        return True
    return requested == actual or requested in actual or actual in requested


class SystemProfiler:
    """Профилировщик системы для выбора оптимальной модели."""
    
    @staticmethod
    def get_system_profile():
        """Получает профиль системы."""
        try:
            cpu_cores = psutil.cpu_count(logical=False) or 2
            cpu_threads = psutil.cpu_count(logical=True) or 4
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except:
            cpu_cores = 2
            cpu_threads = 4
            ram_gb = 4.0
        
        profile = {
            "cpu_cores": cpu_cores,
            "cpu_threads": cpu_threads,
            "ram_gb": ram_gb,
            "os": platform.system(),
            "is_laptop": SystemProfiler._is_laptop(),
            "has_cuda": SystemProfiler._has_cuda(),
            "cpu_freq": SystemProfiler._get_cpu_freq(),
        }
        
        return profile
    
    @staticmethod
    def _is_laptop():
        """Определяет, является ли устройство ноутбуком."""
        try:
            # Проверяем наличие батареи
            battery = psutil.sensors_battery()
            return battery is not None
        except:
            return False
    
    @staticmethod
    def _has_cuda():
        """Проверяет наличие CUDA."""
        from core.gpu_info import cuda_available

        return cuda_available()
    
    @staticmethod
    def _get_cpu_freq():
        """Получает частоту CPU."""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return freq.current
        except:
            pass
        return 0.0
    
    @staticmethod
    def get_system_tier(profile=None):
        """Определяет уровень системы."""
        if profile is None:
            profile = SystemProfiler.get_system_profile()
        
        ram_gb = profile["ram_gb"]
        cpu_cores = profile["cpu_cores"]
        has_cuda = profile["has_cuda"]
        
        # Определяем уровень системы
        if ram_gb < 6 or cpu_cores < 4:
            return "low"      # Слабый ноутбук/ПК
        elif not has_cuda and ram_gb < 12:
            return "medium"   # Средний ПК без видеокарты
        elif has_cuda and ram_gb >= 8:
            return "high"     # Мощный ПК с видеокартой
        elif ram_gb >= 16:
            return "high"     # Много оперативки
        else:
            return "medium"
    
    @staticmethod
    def get_recommended_models(system_profile=None, tier=None):
        """Возвращает список рекомендованных моделей для системы."""
        if system_profile is None:
            system_profile = SystemProfiler.get_system_profile()
        
        if tier is None:
            tier = SystemProfiler.get_system_tier(system_profile)
        
        has_cuda = system_profile["has_cuda"]
        ram_gb = system_profile["ram_gb"]
        cpu_cores = system_profile["cpu_cores"]
        is_laptop = system_profile["is_laptop"]
        
        # Все доступные модели Qwen в порядке от легкой к тяжелой
        all_models = [
            # Очень легкие (для слабых систем)
            {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "min_ram_gb": 3,
                "recommended_ram_gb": 4,
                "min_cpu_cores": 2,
                "size": "0.5B",
                "description": "Очень легкая, для слабых ПК/ноутбуков",
                "for_tier": ["low"],
                "priority": 10
            },
            {
                "name": "Qwen/Qwen2-1.5B-Instruct",
                "min_ram_gb": 4,
                "recommended_ram_gb": 6,
                "min_cpu_cores": 2,
                "size": "1.5B",
                "description": "Легкая, базовые возможности",
                "for_tier": ["low", "medium"],
                "priority": 8
            },
            # Средние (баланс производительности)
            {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",
                "min_ram_gb": 6,
                "recommended_ram_gb": 8,
                "min_cpu_cores": 4,
                "size": "1.5B",
                "description": "Хороший баланс скорость/качество",
                "for_tier": ["medium"],
                "priority": 7
            },
            {
                "name": "Qwen/Qwen2.5-3B-Instruct",
                "min_ram_gb": 8,
                "recommended_ram_gb": 12,
                "min_cpu_cores": 4,
                "size": "3B",
                "description": "Более умная, требует ресурсов",
                "for_tier": ["medium", "high"],
                "priority": 6
            },
            # Мощные (для ПК с видеокартой)
            {
                "name": "Qwen/Qwen3-4B-Instruct-2507",
                "min_ram_gb": 8,
                "recommended_ram_gb": 12,
                "min_cpu_cores": 6,
                "size": "4B",
                "description": "Мощная модель, хорошее качество",
                "for_tier": ["high"],
                "priority": 5 if has_cuda else 3
            },
            {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "min_ram_gb": 12,
                "recommended_ram_gb": 16,
                "min_cpu_cores": 8,
                "size": "7B",
                "description": "Очень мощная, только с видеокартой",
                "for_tier": ["high"],
                "priority": 4 if has_cuda else 1
            }
        ]
        
        # Фильтруем по уровню системы
        tier_models = []
        for model in all_models:
            if tier in model["for_tier"]:
                tier_models.append(model)
        
        # Дополнительная фильтрация по ресурсам
        suitable_models = []
        for model in tier_models:
            if ram_gb >= model["min_ram_gb"] and cpu_cores >= model["min_cpu_cores"]:
                # Рассчитываем финальный приоритет
                score = model["priority"]
                
                # Бонус за соответствие требованиям
                if ram_gb >= model["recommended_ram_gb"]:
                    score += 2
                
                # Бонус за видеокарту для тяжелых моделей
                if has_cuda and model["size"] in ["4B", "7B"]:
                    score += 3
                
                # Штраф для ноутбуков без видеокарты
                if is_laptop and not has_cuda and model["size"] in ["4B", "7B"]:
                    score -= 2
                
                suitable_models.append({
                    **model,
                    "score": score,
                    "suitable": True,
                    "estimated_speed": SystemProfiler._estimate_speed(model, system_profile)
                })
        
        # Сортируем по приоритету
        suitable_models.sort(key=lambda x: x["score"], reverse=True)
        
        # Если нет подходящих для текущего уровня, ищем любые подходящие
        if not suitable_models:
            for model in all_models:
                if ram_gb >= model["min_ram_gb"] and cpu_cores >= model["min_cpu_cores"]:
                    suitable_models.append({
                        **model,
                        "score": 1,
                        "suitable": True,
                        "estimated_speed": SystemProfiler._estimate_speed(model, system_profile)
                    })
        
        return suitable_models
    
    @staticmethod
    def _estimate_speed(model, system_profile):
        """Оценивает скорость работы модели."""
        has_cuda = system_profile["has_cuda"]
        cpu_cores = system_profile["cpu_cores"]
        ram_gb = system_profile["ram_gb"]
        
        size = model["size"]
        if "0.5B" in size:
            base_speed = "Очень быстро"
        elif "1.5B" in size:
            base_speed = "Быстро"
        elif "3B" in size:
            base_speed = "Средне"
        elif "4B" in size:
            base_speed = "Медленно" if not has_cuda else "Быстро"
        elif "7B" in size:
            base_speed = "Очень медленно" if not has_cuda else "Средне"
        else:
            base_speed = "Неизвестно"
        
        return base_speed
    
    @staticmethod
    def get_best_model_for_system():
        """Возвращает лучшую модель для текущей системы."""
        suitable_models = SystemProfiler.get_recommended_models()
        
        if suitable_models:
            return suitable_models[0]["name"]
        else:
            # Резервный переход на самую лёгкую модель
            return "Qwen/Qwen2.5-0.5B-Instruct"
    
    @staticmethod
    def print_system_info():
        """Печатает информацию о системе."""
        profile = SystemProfiler.get_system_profile()
        tier = SystemProfiler.get_system_tier(profile)
        
        print(f"\n{'='*60}")
        print("📊 АНАЛИЗ СИСТЕМЫ")
        print(f"{'='*60}")
        print(f"   Уровень системы: {tier.upper()}")
        print(f"   ОЗУ: {profile['ram_gb']:.1f} GB")
        print(f"   Ядра CPU: {profile['cpu_cores']}")
        print(f"   Потоки CPU: {profile['cpu_threads']}")
        if profile['cpu_freq'] > 0:
            print(f"   Частота CPU: {profile['cpu_freq']:.1f} MHz")
        print(f"   CUDA: {'✅ Доступна' if profile['has_cuda'] else '❌ Недоступна'}")
        print(f"   Ноутбук: {'Да' if profile['is_laptop'] else 'Нет'}")
        print(f"   ОС: {profile['os']}")
        
        # Рекомендации
        suitable_models = SystemProfiler.get_recommended_models(profile, tier)
        if suitable_models:
            print(f"\n   🎯 РЕКОМЕНДОВАННЫЕ МОДЕЛИ:")
            for i, model in enumerate(suitable_models[:3], 1):
                print(f"      {i}. {model['name']}")
                print(f"         Размер: {model['size']} | {model['description']}")
                print(f"         Скорость: {model['estimated_speed']}")
                print(f"         Требуется: {model['min_ram_gb']}+ GB RAM")
        
        return profile, tier


class SmartModelDownloader:
    """
    Умный загрузчик моделей с ЧЕТКИМ разделением типов моделей.
    
    Исправления:
    1. Не путает LLM и эмбеддинговые модели
    2. Четко проверяет тип модели перед возвратом
    3. Разные директории для разных типов моделей
    4. Улучшенная проверка целостности с учетом типа модели
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        offline_mode: bool = False,
        read_only: bool = False,
    ):
        self.cache_dir = resolve_model_cache_dir(cache_dir)
        self.offline_mode = offline_mode
        self.read_only = bool(read_only)
        if not self.read_only:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.downloaded_models = set()
        
        # Создаем поддиректории для разных типов моделей
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.llms_dir = self.cache_dir / "llms"
        if not self.read_only:
            self.embeddings_dir.mkdir(exist_ok=True)
            self.llms_dir.mkdir(exist_ok=True)
        
        # Файл для отслеживания скачанных моделей
        self.meta_file = self.cache_dir / "models_meta.json"
        self.models_meta = self._load_models_meta()
        
        # Файл для выбора модели
        self.choice_file = self.cache_dir / "model_choice.json"
        
        # Предустановленные пути для поиска
        self.common_paths = [
            *model_cache_search_dirs(),
            self.embeddings_dir,
            self.llms_dir,
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "torch" / "hub",
        ]

    def _resolve_cached_model_path(self, model_path: str) -> str:
        """Resolve portable/stale model paths against the active cache dir."""
        if not model_path:
            return model_path

        raw_path = Path(model_path)
        candidates: list[Path] = []

        def add(candidate: Path) -> None:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved not in candidates:
                candidates.append(resolved)

        parts = list(raw_path.parts)
        lower_parts = [part.lower() for part in parts]
        if "models_cache" in lower_parts:
            idx = lower_parts.index("models_cache")
            after_cache = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path()
            if str(after_cache):
                add(self.cache_dir / after_cache)
            add(resolve_project_path(Path("models_cache") / after_cache))

        if raw_path.is_absolute():
            add(raw_path)
        else:
            add(self.cache_dir / raw_path)
            add(resolve_project_path(raw_path))

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0]) if candidates else model_path
    
    def _load_models_meta(self) -> Dict:
        """Загружает метаданные о скачанных моделях."""
        if self.meta_file.exists():
            try:
                data = read_json_with_backup(self.meta_file, default={})
                if not isinstance(data, dict):
                    return {}

                # Validation may resolve paths in memory, but loading metadata
                # must never rewrite portable relative paths as machine-specific
                # absolute paths. Only genuinely invalid entries are removed,
                # and diagnostics run in read-only mode do not persist even that.
                to_delete = []
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    resolved_path = self._resolve_cached_model_path(str(value.get("path", "")))
                    if "qwen" in key.lower():
                        lowered = resolved_path.lower()
                        if "sentence-transformers" in lowered or "paraphrase" in lowered:
                            to_delete.append(key)

                for key in to_delete:
                    print(f"🧹 Удаляю неправильную запись при загрузке: {key}")
                    del data[key]
                if to_delete:
                    self._save_models_meta_payload(data)
                return data
            except:
                return {}
        return {}

    def _save_models_meta_payload(self, payload: Dict) -> None:
        if self.read_only:
            return
        atomic_write_json(self.meta_file, payload)

    def _save_models_meta(self):
        """Сохраняет метаданные о скачанных моделях."""
        try:
            self._save_models_meta_payload(self.models_meta)
        except Exception as e:
            print(f"⚠️  Не могу сохранить метаданные: {e}")

    def _portable_cache_path(self, model_path: str) -> str:
        try:
            resolved = Path(model_path).resolve()
            relative = resolved.relative_to(self.cache_dir.resolve())
            return str(relative)
        except Exception:
            return str(model_path)
    
    def _determine_model_type(self, model_name: str) -> str:
        """Определяет тип модели по её имени."""
        model_lower = model_name.lower()
        
        if "sentence-transformers" in model_lower:
            return "embedding"
        elif "paraphrase" in model_lower and "minilm" in model_lower:
            return "embedding"
        elif "embedding" in model_lower:
            return "embedding"
        elif "qwen" in model_lower or "llama" in model_lower or "mistral" in model_lower:
            return "llm"
        elif "instruct" in model_lower or "chat" in model_lower:
            return "llm"
        else:
            # По умолчанию считаем модель LLM, если явно не указано иное
            return "llm"
    
    def _get_target_dir(self, model_name: str) -> Path:
        """Возвращает целевую директорию для модели в зависимости от типа."""
        model_type = self._determine_model_type(model_name)
        
        if model_type == "embedding":
            return self.embeddings_dir
        else:  # llm или неизвестный тип
            return self.llms_dir
    
    def _check_model_integrity(self, model_path: str, expected_type: str = None) -> Tuple[bool, str]:
        """
        Проверяет целостность локальной модели с учетом типа.
        
        Returns:
            (is_valid, reason)
        """
        path = Path(model_path)
        
        if not path.exists():
            return False, "Путь не существует"
        
        # Проверяем наличие папки и что она не пустая
        try:
            items = list(path.iterdir())
            if not items:
                return False, "Папка пустая"
        except OSError:
            return False, "Не могу прочитать папку"
        
        # Проверяем наличие оффлайн-конфига
        offline_config = path / "offline_config.json"
        if offline_config.exists():
            try:
                with open(offline_config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if config.get("partial_download"):
                    return False, "Частичная загрузка: отсутствуют полные веса модели"
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Некорректный offline_config.json в %s: %s", path, exc)
                return False, "Некорректный offline_config.json"

        hash_report = verify_model_hash_manifest(path)
        if hash_report.get("manifest_present") and not hash_report.get("ok"):
            errors = "; ".join(hash_report.get("errors") or [])
            return False, f"Ошибка SHA-256: {errors}"
        hash_suffix = (
            f", SHA-256 проверен ({hash_report.get('checked_files', 0)} файлов)"
            if hash_report.get("ok")
            else ", SHA-256 манифест отсутствует"
        )
        
        # Одна и та же строгая структурная проверка используется менеджером,
        # runtime и portable-верификатором. Папка только с config/tokenizer
        # больше не считается моделью: должны присутствовать все заявленные
        # шарды весов и пригодный токенизатор.
        structural = inspect_hf_model_payload(path, require_tokenizer=True)
        if not structural.get("ok"):
            errors = "; ".join(str(item) for item in structural.get("errors") or [])
            return False, f"Неполная модель: {errors or 'структурная проверка не пройдена'}"

        if expected_type in {"llm", "embedding"}:
            detected_type = expected_type
        elif (path / "sentence_bert_config.json").is_file() or (path / "modules.json").is_file():
            detected_type = "embedding"
        else:
            detected_type = "llm"

        path_str = str(path).lower()
        if detected_type == "llm" and (
            "sentence-transformers" in path_str or "paraphrase" in path_str
        ):
            return False, "Путь указывает на эмбеддинги, но ожидалась LLM"

        weight_files = int(structural.get("weight_files") or 0)
        weight_gb = float(structural.get("total_weight_bytes") or 0) / (1024 ** 3)
        return True, (
            f"Целостность ({detected_type}, веса: {weight_files} файлов/{weight_gb:.2f} ГБ"
            f"{hash_suffix})"
        )
    
    def _create_offline_config(self, model_path: str, model_name: str, partial: bool = False) -> None:
        """Создает конфигурационные файлы для оффлайн работы."""
        model_type = self._determine_model_type(model_name)
        
        config = {
            "model_name": model_name,
            "model_type": model_type,
            "local_path": model_path,
            "offline_mode": True,
            "requires_internet": False,
            "downloaded_at": time.time(),
            "downloaded_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "integrity_check": "passed",
            "partial_download": partial
        }
        if not partial and (Path(model_path) / MODEL_HASH_MANIFEST).is_file():
            config["hash_manifest"] = MODEL_HASH_MANIFEST
        
        if partial:
            config["warning"] = "Модель скачана частично. Возможны ошибки в работе."
        
        config_path = Path(model_path) / "offline_config.json"
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"📄 Создан оффлайн-конфиг для {model_name} ({model_type})")
        except Exception as e:
            print(f"⚠️  Не могу создать конфиг: {e}")
    
    def _mark_model_as_downloaded(self, model_name: str, model_path: str, partial: bool = False):
        """Отмечает модель как скачанную."""
        model_type = self._determine_model_type(model_name)
        
        # КРИТИЧНЫЙ ФИКС: проверяем, что путь соответствует типу
        path_str = str(model_path).lower()
        
        if model_type == "llm":
            # LLM НЕ должны быть в embeddings папке или иметь embedding-имена
            if ("sentence-transformers" in path_str or 
                "paraphrase" in path_str or 
                "minilm" in path_str):
                print(f"🚨 ОШИБКА: LLM модель {model_name} имеет путь к эмбеддингам!")
                print(f"   Путь: {model_path}")
                print(f"   Игнорирую запись в метаданные")
                return  # НЕ сохраняем такую запись!
        
        self.models_meta[model_name] = {
            "path": self._portable_cache_path(model_path),
            "model_type": model_type,
            "downloaded_at": time.time(),
            "offline_ready": True,
            "partial": partial
        }
        self.downloaded_models.add(model_name)
        self._save_models_meta()
    
    def _find_model_in_cache(self, model_name: str, expected_type: str = None) -> List[str]:
        """
        Ищет модель в кэш-директориях с учетом типа модели.
        
        Args:
            model_name: Имя модели для поиска
            expected_type: Ожидаемый тип модели ("llm" или "embedding")
            
        Returns:
            Список найденных путей
        """
        if expected_type is None:
            expected_type = self._determine_model_type(model_name)
        
        found_paths = []
        
        # Стандартные имена папок для этой модели
        possible_dir_names = [
            model_name.replace("/", "--"),
            f"models--{model_name.replace('/', '--')}",
            hashlib.md5(model_name.encode()).hexdigest()[:8],
        ]
        
        # КРИТИЧНЫЙ ФИКС: если ищем LLM, НЕ ищем в embeddings директории вообще
        if expected_type == "llm":
            search_dirs = [self.llms_dir]
            # Также добавляем корень кэша, но фильтруем эмбеддинги
            for base_path in [self.cache_dir, Path.home() / ".cache" / "huggingface" / "hub"]:
                if base_path.exists():
                    search_dirs.append(base_path)
        else:
            search_dirs = [self.embeddings_dir] + self.common_paths
        
        # Ищем во всех возможных местах
        for base_path in search_dirs:
            if not base_path.exists():
                continue
                
            for dir_name in possible_dir_names:
                path = base_path / dir_name
                if path.exists() and path.is_dir():
                    # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: фильтруем по пути
                    path_str = str(path).lower()
                    if expected_type == "llm" and ("sentence-transformers" in path_str or "paraphrase" in path_str):
                        continue  # Пропускаем эмбеддинги при поиске LLM
                    
                    found_paths.append(str(path))
        
        # Ищем рекурсивно по config.json (только для LLM ищем generation_config.json)
        for base_path in search_dirs:
            if not base_path.exists():
                continue
                
            for root, dirs, files in os.walk(base_path):
                # Для LLM ищем специфические файлы
                if expected_type == "llm":
                    if "generation_config.json" in files or "qwen.tiktoken" in files:
                        # Дополнительная проверка: точно ли это LLM
                        if "sentence-transformers" not in root.lower() and "paraphrase" not in root.lower():
                            found_paths.append(root)
                else:  # embedding-модель
                    if "config.json" in files and "sentence_bert_config.json" in files:
                        found_paths.append(root)
        
        return list(set(found_paths))  # Убираем дубликаты
    
    def is_model_available(self, model_name: str) -> Tuple[bool, Optional[str]]:
        """
        Проверяет, доступна ли модель локально.
        
        Returns:
            (is_available, local_path_or_error)
        """
        expected_type = self._determine_model_type(model_name)
        
        # Проверяем в метаданных
        if model_name in self.models_meta:
            model_info = self.models_meta[model_name]
            model_path = self._resolve_cached_model_path(str(model_info.get("path", "")))
            meta_type = model_info.get("model_type", "")
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: если это Qwen, но путь к эмбеддингам - удаляем
            if "qwen" in model_name.lower():
                path_lower = model_path.lower()
                if "sentence-transformers" in path_lower or "paraphrase" in path_lower:
                    print(f"🧹 Удаляю поврежденную запись: {model_name} -> {model_path}")
                    del self.models_meta[model_name]
                    self._save_models_meta()
                    # Продолжаем поиск
                elif model_path and os.path.exists(model_path):
                    if not _path_matches_model_name(model_name, model_path):
                        del self.models_meta[model_name]
                        self._save_models_meta()
                    else:
                        is_valid, reason = self._check_model_integrity(model_path, expected_type)
                        if is_valid and "llm" in reason.lower():
                            return True, model_path
            
            # Старая логика для других случаев
            elif meta_type and meta_type != expected_type:
                print(f"⚠️  Несоответствие типа в метаданных: ожидалось {expected_type}, а в мета {meta_type}")
                # Удаляем неправильную запись
                del self.models_meta[model_name]
                self._save_models_meta()
            elif model_path and os.path.exists(model_path):
                if expected_type == "llm" and not _path_matches_model_name(model_name, model_path):
                    del self.models_meta[model_name]
                    self._save_models_meta()
                else:
                    is_valid, reason = self._check_model_integrity(model_path, expected_type)
                    if is_valid:
                        return True, model_path
                    else:
                        print(f"⚠️  Модель в метаданных повреждена: {reason}")
        
        # Ищем в кэш-директориях
        found_paths = self._find_model_in_cache(model_name, expected_type)
        
        for path in found_paths:
            is_valid, reason = self._check_model_integrity(path, expected_type)
            if is_valid and expected_type in reason.lower():
                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА для LLM
                if expected_type == "llm":
                    path_lower = path.lower()
                    if "sentence-transformers" in path_lower or "paraphrase" in path_lower:
                        print(f"⚠️  Пропускаю {path} - путь указывает на эмбеддинги")
                        continue
                    if not _path_matches_model_name(model_name, path):
                        continue
                
                # Обновляем метаданные
                partial = "partial" in reason.lower()
                self._mark_model_as_downloaded(model_name, path, partial)
                return True, path
            else:
                print(f"⚠️  Найдена неподходящая модель по пути {path}: {reason}")
        
        return False, "Модель не найдена локально"
    
    def download_model_with_fallback(self, 
                                   model_name: str,
                                   force_download: bool = False) -> str:
        """
        Умная загрузка модели с fallback'ами.
        
        Стратегия:
        1. Проверяем локальную копию с учетом типа модели
        2. Если force_download=False и модель есть — используем её
        3. Если нет — скачиваем с интернета
        4. Если интернет недоступен — пробуем облегченную версию
        """
        print(f"\n🤖 Работа с моделью: {model_name}")
        model_type = self._determine_model_type(model_name)
        print(f"   Тип модели: {model_type.upper()}")
        print(f"   Оффлайн режим: {'ДА' if self.offline_mode else 'НЕТ'}")
        print(f"   Принудительная загрузка: {'ДА' if force_download else 'НЕТ'}")
        
        # 1. Проверяем локальную модель (если не force_download)
        if not force_download:
            is_available, local_path = self.is_model_available(model_name)
            if is_available:
                print(f"✅ Модель уже доступна локально ({model_type})")
                print(f"   Путь: {local_path}")
                return local_path
        
        # 2. Если оффлайн режим, но модели нет
        if self.offline_mode:
            print(f"❌ Оффлайн режим, но модель {model_name} не найдена")
            print(f"💡 Решения:")
            print(f"   1. Отключите оффлайн режим: измените offline_mode=False")
            print(f"   2. Скачайте модель вручную в {self._get_target_dir(model_name)}")
            print(f"   3. Используйте более легкую модель")
            raise FileNotFoundError(f"Модель {model_name} не найдена в оффлайн режиме")
        
        # 3. Пробуем скачать с интернета
        print(f"🌐 Пытаюсь скачать модель из интернета...")
        
        try:
            # Проверяем подключение к интернету
            print(f"   Проверка подключения к HuggingFace...")
            
            # Определяем целевую директорию
            target_dir = self._get_target_dir(model_name)
            
            # Создаем локальную директорию
            local_dir_name = model_name.replace("/", "--")
            local_dir = target_dir / local_dir_name
            local_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"   Целевая папка: {target_dir}")
            print(f"   Локальная папка: {local_dir}")
            print(f"   Это может занять несколько минут...")
            
            # Определяем параметры скачивания в зависимости от типа
            ignore_patterns = ["*.safetensors.index.json", "*.msgpack", "*.h5", "*.ot", "*.tflite"]
            
            # Для эмбеддингов не качаем тяжелые onnx файлы
            if model_type == "embedding":
                ignore_patterns.extend(["*.onnx", "openvino_model*"])
            
            # Скачиваем модель с прогрессом
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(target_dir),
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=ignore_patterns
            )
            
            print(f"✅ Модель успешно скачана! ({model_type})")
            print(f"   Путь: {model_path}")

            print("🔐 Создаю SHA-256 манифест модели...")
            create_model_hash_manifest(model_path)
            
            # Создаем оффлайн-конфиг
            self._create_offline_config(model_path, model_name, partial=False)
            
            # Отмечаем как скачанную
            self._mark_model_as_downloaded(model_name, model_path, partial=False)
            
            # Проверяем целостность
            is_valid, reason = self._check_model_integrity(model_path, model_type)
            if is_valid:
                print(f"✅ Целостность проверена: {reason}")
            else:
                print(f"⚠️  Скачанная модель может быть повреждена: {reason}")
            
            # Показываем размер
            try:
                total_size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file())
                print(f"📊 Размер модели: {total_size / (1024**3):.2f} GB")
            except:
                pass
            
            return model_path
            
        except Exception as e:
            print(f"❌ Ошибка скачивания: {e}")
            
            # Для LLM моделей пробуем скачать более легкую версию
            if model_type == "llm" and "qwen" in model_name.lower():
                print(f"\n🔄 {model_name} не скачался, пробую более легкие варианты...")
                
                # Предлагаем альтернативы
                lighter_models = [
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2-1.5B-Instruct",
                ]
                
                for alt_model in lighter_models:
                    print(f"\n   Пробую {alt_model}...")
                    try:
                        return self.download_model_with_fallback(alt_model, force_download)
                    except Exception as alt_e:
                        print(f"   ❌ {alt_model}: {alt_e}")
                        continue
                
                print(f"\n❌ Не удалось скачать ни одну Qwen модель")
            
            # Пробуем скачать только ключевые файлы (легкие)
            print(f"🔄 Пробую альтернативный метод загрузки...")
            
            try:
                # Создаем структуру папок
                target_dir = self._get_target_dir(model_name)
                local_dir = target_dir / model_name.replace("/", "--")
                local_dir.mkdir(exist_ok=True, parents=True)
                
                # Скачиваем только конфиги и легкие файлы
                files_to_download = [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "generation_config.json",
                    "special_tokens_map.json",
                ]
                
                # Для LLM добавляем model.safetensors.index.json
                if model_type == "llm":
                    files_to_download.append("model.safetensors.index.json")
                
                downloaded_count = 0
                for filename in files_to_download:
                    try:
                        print(f"   Скачиваю {filename}...", end=" ", flush=True)
                        hf_hub_download(
                            repo_id=model_name,
                            filename=filename,
                            cache_dir=str(target_dir),
                            local_dir=str(local_dir),
                            local_dir_use_symlinks=False,
                            resume_download=True
                        )
                        downloaded_count += 1
                        print("✅")
                    except Exception as file_error:
                        print(f"❌ ({file_error})")
                
                # Проверяем, что скачалось
                if downloaded_count >= 2:  # Хотя бы config и tokenizer
                    print(f"⚠️  Скачаны только метаданные модели ({downloaded_count} файлов)")
                    
                    # Создаем оффлайн-конфиг с предупреждением
                    self._create_offline_config(str(local_dir), model_name, partial=True)
                    raise RuntimeError(
                        "Полные веса модели не скачаны; папка оставлена для безопасного возобновления"
                    )
                else:
                    raise Exception(f"Не удалось скачать ключевые файлы (скачано: {downloaded_count})")
                    
            except Exception as fallback_error:
                print(f"❌ Альтернативный метод тоже не сработал: {fallback_error}")
                
                # Предлагаем решения
                print(f"\n💡 Попробуйте одно из решений:")
                print(f"   1. Проверьте подключение к интернету")
                print(f"   2. Убедитесь, что есть 10+ GB свободного места")
                print(f"   3. Скачайте модель вручную:")
                print(f"      https://huggingface.co/{model_name}")
                print(f"      Распакуйте в: {self._get_target_dir(model_name)}")
                
                # Для Qwen предлагаем более лёгкие модели
                if "qwen" in model_name.lower():
                    print(f"   4. Используйте облегченную модель:")
                    print(f"      Qwen/Qwen2.5-0.5B-Instruct (самая легкая)")
                    print(f"      Qwen/Qwen2.5-1.5B-Instruct (баланс)")
                
                # Проверяем дисковое пространство
                try:
                    total, used, free = shutil.disk_usage(".")
                    free_gb = free / (1024**3)
                    print(f"\n💾 Свободно места на диске: {free_gb:.1f} GB")
                    if free_gb < 10:
                        print("⚠️  Мало места! Нужно минимум 10 GB")
                except:
                    pass
                
                raise
    
    def get_all_local_models(self) -> Dict[str, Dict]:
        """Возвращает все локальные модели с определением типа."""
        local_models = {}
        
        # Ищем во всех кэш-директориях
        all_paths = []
        for base_path in self.common_paths:
            if base_path.exists():
                for root, dirs, files in os.walk(base_path):
                    if "config.json" in files or "offline_config.json" in files:
                        all_paths.append(root)
        
        # Обрабатываем найденные пути
        for path in all_paths:
            try:
                # Определяем тип модели
                model_type = "unknown"
                
                # Проверяем по файлам
                files_in_dir = [f.name for f in Path(path).iterdir() if f.is_file()]
                
                if "sentence_bert_config.json" in files_in_dir:
                    model_type = "embedding"
                elif "generation_config.json" in files_in_dir:
                    model_type = "llm"
                elif "model.safetensors" in files_in_dir or "pytorch_model.bin" in files_in_dir:
                    model_type = "llm"
                
                # Проверяем целостность
                is_valid, reason = self._check_model_integrity(path, model_type)
                
                # Пытаемся определить имя модели
                model_name = Path(path).name
                
                # Читаем оффлайн-конфиг если есть
                config_path = Path(path) / "offline_config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            model_name = config.get("model_name", model_name)
                            if "model_type" in config:
                                model_type = config["model_type"]
                    except:
                        pass
                
                # Читаем config.json для точного имени
                config_path = Path(path) / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            if "_name_or_path" in config:
                                name = config["_name_or_path"]
                                if "/" in name:
                                    model_name = name.split("/")[-1]
                                else:
                                    model_name = name
                    except:
                        pass
                
                # КРИТИЧНЫЙ ФИКС: если это LLM, но путь содержит эмбеддинги - меняем тип
                path_lower = path.lower()
                if model_type == "llm" and ("sentence-transformers" in path_lower or "paraphrase" in path_lower):
                    print(f"⚠️  Исправляю тип модели {model_name}: путь указывает на эмбеддинги")
                    model_type = "embedding"
                
                # Добавляем в результат только если это не подпапка другой модели
                if not any(path.startswith(p + os.sep) for p in local_models.keys()):
                    local_models[path] = {
                        "name": model_name,
                        "type": model_type,
                        "path": path,
                        "valid": is_valid,
                        "reason": reason,
                        "size_gb": self._get_folder_size_gb(path)
                    }
                
            except Exception as e:
                continue
        
        return local_models
    
    def _get_folder_size_gb(self, folder_path: str) -> float:
        """Возвращает размер папки в GB."""
        try:
            total_size = sum(f.stat().st_size for f in Path(folder_path).rglob('*') if f.is_file())
            return total_size / (1024**3)
        except:
            return 0.0
    
    def cleanup_broken_models(self):
        """Очищает поврежденные модели."""
        print(f"🧹 Проверка поврежденных моделей...")
        
        broken_count = 0
        kept_count = 0
        
        # Получаем все модели
        all_models = self.get_all_local_models()
        
        for path, model_info in all_models.items():
            is_valid = model_info["valid"]
            
            if not is_valid:
                print(f"   Удаляю поврежденную: {model_info['name']}")
                print(f"     Тип: {model_info['type']}")
                print(f"     Причина: {model_info['reason']}")
                print(f"     Путь: {path}")
                
                try:
                    # Удаляем только если в нашей кэш-директории
                    if str(path).startswith(str(self.cache_dir)):
                        shutil.rmtree(path, ignore_errors=True)
                        broken_count += 1
                    else:
                        print(f"     ⚠️  Пропускаю (не в нашей кэш-директории)")
                except Exception as e:
                    print(f"     ❌ Не удалось удалить: {e}")
            else:
                kept_count += 1
        
        # Очищаем метаданные удаленных моделей
        models_to_delete = []
        for model_name, model_info in self.models_meta.items():
            model_path = model_info.get("path", "")
            if model_path and not os.path.exists(model_path):
                models_to_delete.append(model_name)
        
        for model_name in models_to_delete:
            del self.models_meta[model_name]
        
        if broken_count or models_to_delete:
            self._save_models_meta()
            print(f"✅ Удалено {broken_count} поврежденных моделей")
            print(f"   Удалено {len(models_to_delete)} записей из метаданных")
        else:
            print(f"✅ Все модели в порядке")
        
        print(f"📊 Сохранено моделей: {kept_count}")
    
    def repair_model(self, model_name: str) -> bool:
        """
        Пытается починить поврежденную модель.
        """
        print(f"\n🔧 Пытаюсь починить модель: {model_name}")
        
        is_available, local_path = self.is_model_available(model_name)
        if not local_path:
            print(f"❌ Модель не найдена для ремонта")
            return False
        
        try:
            path = Path(local_path)
            
            # Проверяем, что есть хотя бы что-то
            if not path.exists():
                print(f"❌ Путь не существует")
                return False
            
            valid, reason = self._check_model_integrity(str(path), self._determine_model_type(model_name))
            if not valid:
                print(f"❌ Модель нельзя пометить готовой: {reason}")
                return False
            if not (path / MODEL_HASH_MANIFEST).is_file():
                print("🔐 Создаю SHA-256 манифест модели...")
                create_model_hash_manifest(path)
            self._create_offline_config(str(path), model_name, partial=False)
            self._mark_model_as_downloaded(model_name, str(path), partial=False)
            print(f"✅ Целостность модели подтверждена: {reason}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка ремонта: {e}")
            return False
    
    def save_model_choice(self, model_name: str, model_path: str, system_profile: dict):
        """Сохраняет выбор модели для использования в main.py."""
        try:
            import json
            choice_info = {
                "selected_model": model_name,
                "model_path": self._portable_cache_path(model_path),
                "selected_by": "auto",
                "system_profile": system_profile,
                "timestamp": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            if self.read_only:
                return False
            atomic_write_json(self.choice_file, choice_info)
            print(f"📝 Сохранен выбор модели: {self.choice_file}")
            return True
        except Exception as e:
            print(f"⚠️  Не удалось сохранить выбор модели: {e}")
            return False
    
    def get_saved_model_choice(self):
        """Получает сохраненный выбор модели."""
        if self.choice_file.exists():
            try:
                import json
                with open(self.choice_file, 'r', encoding='utf-8-sig') as f:
                    choice = json.load(f)
                if isinstance(choice, dict) and choice.get("model_path"):
                    choice["model_path"] = self._resolve_cached_model_path(str(choice.get("model_path")))
                return choice
            except:
                pass
        return None


# Главная функция для загрузки всех моделей
def download_all_models(offline_mode: bool = False, force: bool = False, 
                       auto_select: bool = True):
    """
    Скачивает все необходимые модели с автовыбором для системы.
    """
    print("=" * 60)
    print("🤖 УМНАЯ ЗАГРУЗКА МОДЕЛЕЙ С АВТОВЫБОРОМ")
    print("=" * 60)
    
    # Профилируем систему
    profiler = SystemProfiler()
    system_profile, tier = profiler.print_system_info()
    
    downloader = SmartModelDownloader(offline_mode=offline_mode)
    
    # 1. Модель для эмбеддингов
    print("\n" + "-" * 40)
    print("1. МОДЕЛЬ ДЛЯ ЭМБЕДДИНГОВ")
    print("-" * 40)
    
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        embed_path = downloader.download_model_with_fallback(
            model_name=embed_model_name,
            force_download=force
        )
        print(f"✅ Готово: {embed_path}")
    except Exception as e:
        print(f"⚠️  Не удалось: {e}")
        print(f"   Можно использовать только BM25 поиск")
    
    # 2. Автоматический выбор LLM модели
    print("\n" + "-" * 40)
    print("2. АВТОВЫБОР LLM МОДЕЛИ")
    print("-" * 40)
    
    selected_model = None
    downloaded_model = None
    downloaded_path = None
    
    if auto_select:
        suitable_models = profiler.get_recommended_models(system_profile, tier)
        
        if suitable_models:
            print(f"\n📋 ДОСТУПНЫЕ МОДЕЛИ ДЛЯ ВАШЕЙ СИСТЕМЫ ({tier.upper()}):")
            for i, model in enumerate(suitable_models[:5], 1):
                print(f"\n   {i}. {model['name']}")
                print(f"      Размер: {model['size']}")
                print(f"      Описание: {model['description']}")
                print(f"      Ожидаемая скорость: {model['estimated_speed']}")
                print(f"      Требует: {model['min_ram_gb']} GB RAM, {model['min_cpu_cores']} ядер")
            
            # Выбираем лучшую модель
            best_model = suitable_models[0]
            selected_model = best_model['name']
            
            print(f"\n🎯 ВЫБРАНА МОДЕЛЬ: {selected_model}")
            print(f"   {best_model['description']}")
            print(f"   Ожидаемая скорость: {best_model['estimated_speed']}")
        else:
            print(f"\n⚠️  Нет подходящих моделей для вашей системы")
            print(f"   Использую минимальную модель...")
            selected_model = "Qwen/Qwen2.5-0.5B-Instruct"
    else:
        # Ручной выбор - старая логика
        model_priority = [
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ]
        
        # Проверяем, какая модель уже есть
        for model_name in model_priority:
            is_available, local_path = downloader.is_model_available(model_name)
            if is_available and not force:
                selected_model = model_name
                downloaded_model = model_name
                downloaded_path = local_path
                print(f"✅ Использую уже скачанную модель: {selected_model}")
                break
        
        if not selected_model:
            selected_model = model_priority[0]  # По умолчанию самая мощная
    
    # Скачиваем выбранную модель (если не нашли уже скачанную)
    if not downloaded_model:
        try:
            print(f"\n📥 Скачиваю выбранную модель: {selected_model}")
            model_path = downloader.download_model_with_fallback(
                model_name=selected_model,
                force_download=force
            )
            
            downloaded_model = selected_model
            downloaded_path = model_path
            print(f"✅ Успешно скачана: {selected_model}")
            
        except Exception as e:
            print(f"❌ Не удалось скачать {selected_model}: {e}")
            print(f"🔄 Пробую скачать более легкую модель...")
            
            # Fallback на более легкие модели
            fallback_models = [
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2-1.5B-Instruct",
            ]
            
            for model_name in fallback_models:
                if model_name == selected_model:
                    continue
                    
                try:
                    print(f"\n   Пробую {model_name}...")
                    model_path = downloader.download_model_with_fallback(
                        model_name=model_name,
                        force_download=force
                    )
                    downloaded_model = model_name
                    downloaded_path = model_path
                    print(f"✅ Успешно скачана {model_name} как fallback")
                    selected_model = model_name
                    break
                except Exception as fallback_e:
                    print(f"   ❌ {model_name}: {fallback_e}")
                    continue
    
    # Вывод результатов
    if downloaded_model:
        if downloaded_path:
            downloaded_model = _model_name_from_path(downloaded_path)
        print("\n" + "=" * 60)
        print("🎉 ВСЕ ГОТОВО!")
        print("=" * 60)
        
        # Сохраняем выбор модели
        downloader.save_model_choice(selected_model, downloaded_path, system_profile)
        
        print(f"\n📋 СКАЧАННЫЕ МОДЕЛИ:")
        print(f"   • Эмбеддинги: {embed_model_name}")
        print(f"   • LLM: {downloaded_model}")
        print(f"   • Уровень системы: {tier.upper()}")
        
        if system_profile['has_cuda']:
            print(f"   • CUDA: ✅ Доступна (ускорение включено)")
        else:
            print(f"   • CUDA: ❌ Недоступна (только CPU)")
        
        print(f"\n💾 ПУТИ:")
        print(f"   Кэш-директория: {downloader.cache_dir}")
        print(f"   Эмбеддинги: {downloader.embeddings_dir}")
        print(f"   LLM модели: {downloader.llms_dir}")
        if downloaded_path:
            print(f"   LLM путь: {downloaded_path}")
        
        print(f"\n🚀 ЗАПУСК:")
        print(f"   python main.py")
        
        print(f"\n⚠️  ВАЖНО:")
        print(f"   При следующем запуске программа будет работать")
        print(f"   полностью оффлайн, используя локальные модели.")
        
        # Создаем файл-маркер об успешной загрузке
        success_file = downloader.cache_dir / "download_success.txt"
        try:
            with open(success_file, 'w', encoding='utf-8') as f:
                f.write(f"Модели успешно скачаны: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM: {downloaded_model}\n")
                f.write(f"Уровень системы: {tier}\n")
                f.write(f"ОЗУ: {system_profile['ram_gb']:.1f} GB\n")
                f.write(f"CUDA: {system_profile['has_cuda']}\n")
            print(f"📝 Создан маркер успеха: {success_file}")
        except:
            pass
        
    else:
        print("\n❌ НЕ УДАЛОСЬ СКАЧАТЬ НИ ОДНУ МОДЕЛЬ")
        print("\n💡 РЕШЕНИЯ:")
        print("   1. Проверьте подключение к интернету")
        print("   2. Убедитесь, что есть 10+ GB свободного места")
        print("   3. Скачайте модель вручную:")
        print("      https://huggingface.co/Qwen")
        print("      Распакуйте в папку .chatkodi/models_cache/llms/")
        
        # Показываем дисковое пространство
        try:
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            print(f"\n💾 Свободно места на диске: {free_gb:.1f} GB")
            if free_gb < 10:
                print("⚠️  Мало места! Нужно минимум 10 GB")
        except:
            pass


# Утилиты для других модулей
def check_models_availability() -> Dict[str, any]:
    """
    Проверяет доступность всех необходимых моделей.
    Используется в main.py перед запуском.
    
    Returns:
        Словарь с доступностью моделей
    """
    from core.gpu_info import primary_gpu
    from core.llm.model_select import describe_model, find_embedding_model, find_llm, list_gguf

    result = {
        "embeddings": False,
        "llm": False,
        "embeddings_path": None,
        "llm_path": None,
        "llm_name": None,
        "llm_type": None,
        "llm_size": "Unknown",
        "all_models": [],
        "system_tier": "unknown",
        "has_cuda": primary_gpu() is not None,
    }
    embedding = find_embedding_model()
    if embedding is not None:
        result["embeddings"] = True
        result["embeddings_path"] = str(embedding)
    llm = find_llm()
    if llm is not None:
        info = describe_model(llm)
        result.update({
            "llm": True,
            "llm_path": str(llm),
            "llm_name": info["name"],
            "llm_type": "llm",
            "llm_size": info["size"],
        })
    result["all_models"] = [f"{p.name} (llm)" for p in list_gguf("llms")] + [
        f"{p.name} (embeddings)" for p in list_gguf("embeddings")
    ]
    return result


def get_offline_mode() -> bool:
    """
    Определяет, можно ли работать в оффлайн режиме.
    Проверяет наличие всех необходимых моделей.
    """
    availability = check_models_availability()
    
    if availability["embeddings"] and availability["llm"]:
        print("✅ Все модели доступны локально, работаем оффлайн")
        print(f"   LLM модель: {availability['llm_name']} ({availability['llm_size']})")
        print(f"   Уровень системы: {availability['system_tier'].upper()}")
        print(f"   CUDA: {'✅ Доступна' if availability['has_cuda'] else '❌ Недоступна'}")
        return True
    else:
        print("⚠️  Не все модели доступны локально")
        print(f"   Эмбеддинги: {'✅' if availability['embeddings'] else '❌'}")
        print(f"   LLM: {'✅' if availability['llm'] else '❌'}")
        
        if not availability["embeddings"]:
            print("   Запустите: python download_models.py")
        elif not availability["llm"]:
            print("   Запустите: python download_models.py")
        
        return False


# Функция для командной строки
def main():
    """Точка входа для командной строки."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Умный загрузчик моделей с автовыбором')
    parser.add_argument('--offline', action='store_true', 
                       help='Только оффлайн режим (не качать из интернета)')
    parser.add_argument('--force', action='store_true',
                       help='Принудительная перезагрузка всех моделей')
    parser.add_argument('--check', action='store_true',
                       help='Только проверка доступности моделей')
    parser.add_argument('--list', action='store_true',
                       help='Показать локальные модели')
    parser.add_argument('--cleanup', action='store_true',
                       help='Очистить поврежденные модели')
    parser.add_argument('--repair', action='store_true',
                       help='Попытаться починить поврежденные модели')
    parser.add_argument('--manual', action='store_true',
                       help='Ручной выбор модели (без автовыбора)')
    
    args = parser.parse_args()
    
    if args.check:
        availability = check_models_availability()
        print("\n📊 СТАТУС МОДЕЛЕЙ:")
        print(f"   Эмбеддинги: {'✅' if availability['embeddings'] else '❌'}")
        print(f"   LLM: {'✅' if availability['llm'] else '❌'}")
        
        if availability['llm_path']:
            print(f"   LLM путь: {availability['llm_path']}")
        if availability['llm_name']:
            print(f"   LLM имя: {availability['llm_name']}")
        if availability['llm_size']:
            print(f"   LLM размер: {availability['llm_size']}")
        
        print(f"   Уровень системы: {availability['system_tier'].upper()}")
        print(f"   CUDA: {'✅ Доступна' if availability['has_cuda'] else '❌ Недоступна'}")
        
        if availability['all_models']:
            print(f"\n📁 ВСЕ ЛОКАЛЬНЫЕ МОДЕЛИ:")
            for model in availability['all_models']:
                print(f"   • {model}")
        
        if availability['embeddings'] and availability['llm']:
            print("\n✅ Все модели доступны, можно работать оффлайн")
            sys.exit(0)
        else:
            print("\n❌ Не все модели доступны")
            sys.exit(1)
    
    elif args.list:
        downloader = SmartModelDownloader()
        local_models = downloader.get_all_local_models()
        
        if local_models:
            print(f"\n📁 ЛОКАЛЬНЫЕ МОДЕЛИ ({len(local_models)}):")
            for path, info in local_models.items():
                name = info["name"]
                model_type = info["type"]
                size = info["size_gb"]
                valid = info["valid"]
                reason = info["reason"]
                
                status = "✅" if valid else "❌"
                print(f"\n   {status} {name} ({model_type})")
                print(f"     Путь: {path}")
                print(f"     Размер: {size:.1f} GB")
                print(f"     Статус: {'рабочая' if valid else 'повреждена'}")
                if not valid:
                    print(f"     Причина: {reason}")
        else:
            print("📭 Локальных моделей нет")
            print(f"   Запустите: python {__file__}")
    
    elif args.cleanup:
        downloader = SmartModelDownloader()
        downloader.cleanup_broken_models()
    
    elif args.repair:
        downloader = SmartModelDownloader()
        
        # Пробуем починить все модели по приоритету
        model_priority = [
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen2.5-1.5B-Instruct", 
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        repaired = False
        for model_name in model_priority:
            if downloader.repair_model(model_name):
                repaired = True
                break
        
        if not repaired:
            print("❌ Не удалось починить ни одну модель")
            sys.exit(1)
    
    else:
        # Основная загрузка
        if args.manual:
            # Ручной выбор
            download_all_models(offline_mode=args.offline, force=args.force, auto_select=False)
        else:
            # Автоматический выбор (по умолчанию)
            download_all_models(offline_mode=args.offline, force=args.force, auto_select=True)


if __name__ == "__main__":
    # Проверяем наличие psutil
    try:
        import psutil
    except ImportError:
        print("❌ psutil не установлен!")
        print("💡 Установите: pip install psutil")
        print("   Без psutil автовыбор модели не будет работать корректно.")
        print("\n   Запускаю с базовыми настройками...")
        # Устанавливаем флаг что psutil нет
        import sys
        sys.modules['psutil'] = None
    
    main()
