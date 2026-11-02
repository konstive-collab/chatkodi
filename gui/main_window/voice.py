"""Voice mode: speak a request (whisper.cpp), hear the answer (Piper).

🎤 starts listening and stops by itself after a pause in speech (or on a
second click); the recognised text goes to the assistant. 🔊 reads answers
aloud. With «Голосовой диалог» the microphone opens again after each answer.
"""

from __future__ import annotations

import array
import tempfile
import time
import wave
from pathlib import Path

from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QMessageBox, QPushButton

from core.logging_utils import get_logger

logger = get_logger(__name__)

_RATE = 16000
_SILENCE_RMS = 450  # 16-bit PCM level below which a 100 ms frame counts as silence
_SILENCE_STOP_S = 1.4
_MAX_SECONDS = 90


class _SpeechWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, recognizer, path: Path):
        super().__init__()
        self.recognizer = recognizer
        self.path = path

    def run(self) -> None:
        try:
            self.done.emit(self.recognizer.transcribe_file(self.path))
        except Exception as exc:
            logger.exception("Speech recognition failed")
            self.failed.emit(str(exc))


class _SayWorker(QThread):
    done = pyqtSignal()

    def __init__(self, synthesizer, text: str):
        super().__init__()
        self.synthesizer = synthesizer
        self.text = text

    def run(self) -> None:
        try:
            import winsound

            out = Path(tempfile.gettempdir()) / "chatkodi_voice" / f"reply_{int(time.time() * 1000)}.wav"
            self.synthesizer.synthesize(self.text, out)
            winsound.PlaySound(str(out), winsound.SND_FILENAME)  # blocks this thread only
        except Exception as exc:
            logger.warning("Speech synthesis failed: %s", exc)
        self.done.emit()


class _LiveText(QObject):
    """Partial transcripts from a worker thread to the input field."""

    text = pyqtSignal(str)


class ChatKodiVoiceMixin:
    def _build_voice_widgets(self, entry_row: QHBoxLayout) -> None:
        self.voice_btn = QPushButton("", self.input_frame)
        self.voice_btn.setObjectName("voiceButton")
        self.voice_btn.setCheckable(True)
        self.voice_btn.setFixedWidth(44)
        self.voice_btn.setCursor(Qt.PointingHandCursor)
        self.voice_btn.setToolTip("Сказать голосом (остановится сам после паузы)")
        self.voice_btn.clicked.connect(self._toggle_listening)
        self.speak_btn = QPushButton("", self.input_frame)
        self.speak_btn.setObjectName("voiceButton")
        self.speak_btn.setCheckable(True)
        self.speak_btn.setFixedWidth(44)
        self.speak_btn.setCursor(Qt.PointingHandCursor)
        self.speak_btn.setToolTip("Озвучивать ответы")
        self.speak_btn.toggled.connect(self._toggle_voice_replies)
        index = entry_row.indexOf(self.input_field) + 1
        entry_row.insertWidget(index, self.voice_btn)
        entry_row.insertWidget(index + 1, self.speak_btn)
        self.voice_reply_enabled = False
        self.voice_dialog_enabled = False
        from core.theme_manager import theme_palette
        from gui.components.wave import VoiceWave

        self.voice_wave = VoiceWave(theme_palette(getattr(self, "current_theme", "classic")).get("accent", "#22c55e"), self.input_frame)
        self.voice_wave.hide()
        self.input_frame.layout().insertWidget(0, self.voice_wave)
        self._live_text = _LiveText()
        self._live_text.text.connect(self._on_partial_text)
        settings = getattr(self, "assistant_settings", None)
        if settings is not None and not settings.speech:
            self.voice_btn.hide()
        self._refresh_assistant_icons()

    # ------------------------------------------------------------ engines

    def _recognizer(self):
        if getattr(self, "_speech", None) is None:
            from core.speech import SpeechRecognizer

            from core.speech import code_vocabulary

            self._speech = SpeechRecognizer()
            processor = getattr(self, "doc_processor", None)
            docs = list(getattr(processor, "documents", None) or [])[:3000]
            self._speech.vocabulary = code_vocabulary(str(getattr(d, "content", ""))[:20000] for d in docs)
        return self._speech

    def _synthesizer(self):
        if getattr(self, "_tts", None) is None:
            from core.speech import SpeechSynthesizer

            self._tts = SpeechSynthesizer(voice=str(getattr(self, "voice_name", "") or ""))
        return self._tts

    # ------------------------------------------------------------ listening

    def _toggle_listening(self, checked: bool) -> None:
        if checked:
            self._start_listening()
        else:
            self._stop_listening(send=True)

    def _start_listening(self) -> None:
        from PyQt5.QtMultimedia import QAudio, QAudioDeviceInfo, QAudioFormat, QAudioInput

        recognizer = self._recognizer()
        if not recognizer.available:
            self.voice_btn.setChecked(False)
            QMessageBox.information(self, "Голосовой ввод недоступен", recognizer.unavailable_reason())
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(_RATE)
        fmt.setChannelCount(1)
        fmt.setSampleSize(16)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)
        device = QAudioDeviceInfo.defaultInputDevice()
        if device.isNull():
            self.voice_btn.setChecked(False)
            QMessageBox.information(self, "Нет микрофона", "Микрофон не найден.")
            return
        if not device.isFormatSupported(fmt):
            fmt = device.nearestFormat(fmt)
        self._audio_format = fmt
        self._audio_bytes = QByteArray()
        self._audio_buffer = QBuffer(self._audio_bytes)
        self._audio_buffer.open(QIODevice.WriteOnly)
        self._audio_input = QAudioInput(device, fmt, self)
        self._audio_input.start(self._audio_buffer)
        self._voice_heard = False
        self._voice_last_sound = time.monotonic()
        self._voice_started = time.monotonic()
        self._voice_read_pos = 0
        self._voice_timer = QTimer(self)
        self._voice_timer.timeout.connect(self._check_silence)
        self._voice_timer.start(100)
        self._partial_busy = False
        self._partial_size = 0
        self._partial_timer = QTimer(self)
        self._partial_timer.timeout.connect(self._partial_transcribe)
        self._partial_timer.start(1200)
        self.voice_wave.start()
        self.input_field.clear()
        self.input_field.setPlaceholderText("Говорите — текст появится здесь…")
        self._refresh_assistant_icons()
        self.status.showMessage("Слушаю… говорите. Остановлюсь сам после паузы.")
        # start the recognizer while the user is speaking
        QTimer.singleShot(0, lambda: self._warm_recognizer())

    def _warm_recognizer(self) -> None:
        import threading

        threading.Thread(target=lambda: self._safe_start(self._recognizer()), daemon=True).start()

    @staticmethod
    def _safe_start(recognizer) -> None:
        try:
            recognizer.start()
        except Exception as exc:
            logger.warning("whisper-server warm-up failed: %s", exc)

    def _check_silence(self) -> None:
        data = bytes(self._audio_bytes)
        frame = data[self._voice_read_pos:]
        self._voice_read_pos = len(data)
        if self._audio_format.sampleSize() == 16 and len(frame) >= 2:
            samples = array.array("h", frame[: len(frame) // 2 * 2])
            rms = (sum(s * s for s in samples[::4]) / max(1, len(samples[::4]))) ** 0.5
            self.voice_wave.set_level(rms / 2600.0)
            if rms > _SILENCE_RMS:
                self._voice_heard = True
                self._voice_last_sound = time.monotonic()
        now = time.monotonic()
        if (self._voice_heard and now - self._voice_last_sound > _SILENCE_STOP_S) or now - self._voice_started > _MAX_SECONDS:
            self._stop_listening(send=True)
        elif not self._voice_heard and now - self._voice_started > 8:
            self._stop_listening(send=False)
            self.status.showMessage("Не услышал речь. Нажмите 🎤 и говорите.")

    def _stop_listening(self, send: bool) -> None:
        for name in ("_voice_timer", "_partial_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()
        audio = getattr(self, "_audio_input", None)
        if audio is None:
            return
        self.voice_wave.stop()
        if hasattr(self, "_update_mode_display"):
            self._update_mode_display()  # the usual placeholder back
        audio.stop()
        self._audio_input = None
        self._audio_buffer.close()
        self.voice_btn.setChecked(False)
        self._refresh_assistant_icons()
        if not send or not getattr(self, "_voice_heard", False):
            return
        fmt = self._audio_format
        path = Path(tempfile.gettempdir()) / "chatkodi_voice" / f"request_{int(time.time() * 1000)}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(fmt.channelCount())
            out.setsampwidth(max(1, fmt.sampleSize() // 8))
            out.setframerate(fmt.sampleRate())
            out.writeframes(bytes(self._audio_bytes))
        self.status.showMessage("Распознаю речь…")
        self.voice_btn.setEnabled(False)
        worker = _SpeechWorker(self._recognizer(), path)
        self._speech_worker = worker
        worker.done.connect(self._on_speech_text)
        worker.failed.connect(lambda error: (self.voice_btn.setEnabled(True), QMessageBox.warning(self, "Распознавание речи", error)))
        worker.start()

    def _wav_of_buffer(self, name: str) -> Path:
        fmt = self._audio_format
        path = Path(tempfile.gettempdir()) / "chatkodi_voice" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(fmt.channelCount())
            out.setsampwidth(max(1, fmt.sampleSize() // 8))
            out.setframerate(fmt.sampleRate())
            out.writeframes(bytes(self._audio_bytes))
        return path

    def _partial_transcribe(self) -> None:
        """While the user speaks, what has been said so far appears in the field."""
        if getattr(self, "_audio_input", None) is None or self._partial_busy or not self._voice_heard:
            return
        size = len(self._audio_bytes)
        if size - self._partial_size < self._audio_format.sampleRate():  # < 0.5 s of new speech
            return
        self._partial_size = size
        self._partial_busy = True
        path = self._wav_of_buffer("partial.wav")
        recognizer = self._recognizer()

        def work():
            try:
                text = recognizer.transcribe_file(path)
            except Exception:
                text = ""
            self._partial_busy = False
            if text:
                self._live_text.text.emit(text)

        import threading

        threading.Thread(target=work, daemon=True).start()

    def _on_partial_text(self, text: str) -> None:
        if getattr(self, "_audio_input", None) is not None:
            self.input_field.setText(text)
            self.input_field.setCursorPosition(len(text))

    def _on_speech_text(self, text: str) -> None:
        self.voice_btn.setEnabled(True)
        if not text.strip():
            self.status.showMessage("Речь не распознана")
            return
        self.input_field.setText(text)
        if not self.speak_btn.isChecked():
            # a spoken request usually wants a spoken answer
            self.speak_btn.setChecked(True)
        self._on_send()

    # ------------------------------------------------------------ speaking

    def _toggle_voice_replies(self, checked: bool) -> None:
        self.voice_reply_enabled = checked
        self._refresh_assistant_icons()
        if checked and not self._synthesizer().available:
            self.speak_btn.setChecked(False)
            QMessageBox.information(self, "Озвучивание недоступно", "Не найден Piper или голос в папке speech.")

    def _toggle_voice_dialog(self, checked: bool) -> None:
        self.voice_dialog_enabled = checked
        if checked and hasattr(self, "speak_btn"):
            self.speak_btn.setChecked(True)

    def _speak_text(self, text: str) -> None:
        from core.speech import speakable

        spoken = speakable(text)
        if not spoken:
            return
        worker = _SayWorker(self._synthesizer(), spoken)
        self._say_worker = worker
        worker.done.connect(self._after_speaking)
        worker.start()

    def _after_speaking(self) -> None:
        if getattr(self, "voice_dialog_enabled", False) and not self.voice_btn.isChecked():
            self.voice_btn.setChecked(True)
            self._start_listening()
