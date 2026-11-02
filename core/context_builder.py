"""
core/context_builder.py

Context Builder v2:
- собирает контекст НЕ просто из топ-чанков, а:
  1) предпочитает чанки с якорями (date_match / id_match / matched_ids)
  2) добавляет окна (соседние чанки) вокруг якорных
  3) держит бюджет по документам и общий бюджет по символам
  4) сохраняет разнообразие документов

Этот модуль не зависит от LLM. Он формирует "documents_context" строку.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

from core.retrieval_signals import is_anchor_chunk


@dataclass
class ContextV2Config:
    enabled: bool = True

    max_docs: int = 6
    max_chars: int = 32000

    window_before: int = 1
    window_after: int = 1

    max_chunks_per_doc: int = 5
    prefer_anchor_chunks: bool = True

    allow_full_doc_fallback: bool = True
    full_doc_max_chars: int = 7000


def load_context_v2_config(cfg: Dict[str, Any]) -> ContextV2Config:
    d = cfg or {}
    return ContextV2Config(
        enabled=bool(d.get("enabled", True)),
        max_docs=int(d.get("max_docs", 6)),
        max_chars=int(d.get("max_chars", 32000)),
        window_before=int(d.get("window_before", 1)),
        window_after=int(d.get("window_after", 1)),
        max_chunks_per_doc=int(d.get("max_chunks_per_doc", 5)),
        prefer_anchor_chunks=bool(d.get("prefer_anchor_chunks", True)),
        allow_full_doc_fallback=bool(d.get("allow_full_doc_fallback", True)),
        full_doc_max_chars=int(d.get("full_doc_max_chars", 7000)),
    )


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def build_context_v2(
    chunk_results: List[Tuple[float, Dict[str, Any]]],
    doc_processor,
    *,
    cfg: ContextV2Config,
    chunk_id_to_pos: Dict[int, int],
    pos_to_chunk_id: Dict[int, int],
    chunk_texts: List[str],
    chunk_meta_list: List[Dict[str, Any]],
) -> str:
    """
    chunk_results: список лучших чанков (score, meta) из HybridSearch.results
    doc_processor: чтобы достать документ по doc_id (имя/modified_time/size_chars)
    остальные параметры — чтобы доставать соседние чанки по индексу.
    """
    if not chunk_results:
        return ""

    # группируем по документу
    by_doc: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    meta_by_chunk_id: Dict[int, Dict[str, Any]] = {}
    for sc, meta in chunk_results:
        by_doc[meta["doc_id"]].append((float(sc), meta))
        try:
            cid = int(meta.get("chunk_id"))
            meta_by_chunk_id[cid] = meta
        except Exception:
            pass

    # порядок документов: по лучшему скору
    doc_order = sorted(
        by_doc.keys(),
        key=lambda d: max(x[0] for x in by_doc[d]),
        reverse=True
    )

    # выбор чанков: якорные впереди, затем просто по скору
    chosen_by_doc: Dict[str, List[int]] = defaultdict(list)
    chosen_positions: Set[int] = set()

    for doc_id in doc_order:
        # сортируем чанки внутри документа
        items = sorted(by_doc[doc_id], key=lambda x: x[0], reverse=True)
        if cfg.prefer_anchor_chunks:
            items = sorted(
                items,
                key=lambda x: (
                    0 if x[1].get("answer_rescue") else 1,
                    0 if x[1].get("anchor_match") else 1,
                    0 if is_anchor_chunk(x[1]) else 1,
                    -x[0],
                )
            )

        # выбираем базовые чанки
        for sc, meta in items:
            if len(chosen_by_doc[doc_id]) >= cfg.max_chunks_per_doc:
                break

            chunk_id = _safe_int(meta.get("chunk_id"), -1)
            pos = None
            if chunk_id >= 0:
                pos = chunk_id_to_pos.get(chunk_id)
            if pos is None:
                pos = _safe_int(meta.get("chunk_index"), -1)
            if pos is None or pos < 0 or pos >= len(chunk_meta_list):
                continue
            if chunk_meta_list[pos].get("doc_id") != doc_id:
                continue

            if pos not in chosen_positions:
                chosen_by_doc[doc_id].append(pos)
                chosen_positions.add(pos)

    # добавляем окна вокруг выбранных позиций
    if cfg.window_before > 0 or cfg.window_after > 0:
        expanded_by_doc: Dict[str, List[int]] = defaultdict(list)

        for doc_id, poses in chosen_by_doc.items():
            poses_sorted = sorted(set(poses))
            expanded: Set[int] = set(poses_sorted)

            for pos in poses_sorted:
                for p in range(pos - cfg.window_before, pos + cfg.window_after + 1):
                    if p < 0 or p >= len(chunk_meta_list):
                        continue
                    # сосед должен быть тем же doc_id
                    if chunk_meta_list[p].get("doc_id") != doc_id:
                        continue
                    expanded.add(p)

            expanded_by_doc[doc_id] = sorted(expanded)

        chosen_by_doc = expanded_by_doc

    # формируем текст с бюджетом
    parts: List[str] = []
    total = 0
    docs_used = 0

    for doc_id in doc_order:
        if docs_used >= cfg.max_docs:
            break

        poses = chosen_by_doc.get(doc_id) or []
        if not poses:
            continue

        doc = doc_processor.get_document(doc_id)
        if not doc:
            continue

        header = (
            f"📄 ДОКУМЕНТ: {doc.name}\n"
            f"📅 Изменен: {time.ctime(doc.modified_time)}\n"
            f"📏 Размер: {doc.size_chars:,} символов\n"
            + "-" * 70 + "\n"
        )
        if total + len(header) > cfg.max_chars:
            break

        parts.append(header)
        total += len(header)

        for pos in poses:
            meta = dict(chunk_meta_list[pos])
            chunk_key = _safe_int(meta.get("chunk_id"), -1)
            if chunk_key >= 0:
                try:
                    meta.update(meta_by_chunk_id.get(chunk_key, {}))
                except Exception:
                    pass
            text = chunk_texts[pos]

            flags = []
            if meta.get("answer_rescue"):
                flags.append("answer_rescue")
            if meta.get("doc_date_bridge"):
                flags.append("doc_date_bridge")
            if meta.get("date_match"):
                flags.append("date_match")
            if meta.get("id_match"):
                flags.append("id_match")
            if meta.get("object_term_hits", 0):
                flags.append(f"obj={meta.get('object_term_hits')}")
            if meta.get("entity_match_count", 0):
                flags.append(f"ent={meta.get('entity_match_count')}")
            if meta.get("keyword_match_count", 0):
                flags.append(f"kw={meta.get('keyword_match_count')}")
            if meta.get("matched_ids"):
                flags.append(f"ids={','.join(map(str, meta.get('matched_ids')))}")
            flags_str = f" flags={','.join(flags)}" if flags else ""

            block = (
                f"[chunk{flags_str}] "
                f"(chunk_index={meta.get('chunk_index')}, chars={meta.get('start_char')}-{meta.get('end_char')})\n"
                f"{text}\n\n"
            )

            if total + len(block) > cfg.max_chars:
                break

            parts.append(block)
            total += len(block)

        docs_used += 1

    # fallback: если получилось пусто (например бюджет слишком маленький)
    if not parts and cfg.allow_full_doc_fallback:
        # берём самый релевантный документ и даём отрывок
        best_doc_id = doc_order[0]
        doc = doc_processor.get_document(best_doc_id)
        if doc and getattr(doc, "content", None):
            content = doc.content[: max(200, cfg.full_doc_max_chars)]
            return (
                f"📄 ДОКУМЕНТ: {doc.name}\n"
                f"📅 Изменен: {time.ctime(doc.modified_time)}\n"
                + "-" * 70 + "\n"
                f"{content}\n"
            )

    return "\n".join(parts).strip()
