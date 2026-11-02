"""Other programs: a mail draft with attachments (Outlook, or any mail program
through an unsent .eml), a file opened in Word/Excel, a program started.

Mail is never sent by ChatKodi: the draft opens for the user, who sends it.
The PowerShell for Outlook is built by the program (not by the model); the
texts reach it as JSON on stdin, so nothing in them is executed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from email.message import EmailMessage
from pathlib import Path
from typing import List, Tuple

from core.agent.outputs import outputs_dir

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_OUTLOOK = r"""
[Console]::InputEncoding = [Text.Encoding]::UTF8
$d = [Console]::In.ReadToEnd() | ConvertFrom-Json
$outlook = New-Object -ComObject Outlook.Application
$mail = $outlook.CreateItem(0)
$mail.To = $d.to
$mail.Subject = $d.subject
$mail.Body = $d.body
foreach ($f in $d.files) { [void]$mail.Attachments.Add($f) }
$mail.Display()
"ok"
"""

PROGRAMS = {
    "блокнот": "notepad", "notepad": "notepad", "калькулятор": "calc", "calc": "calc",
    "проводник": "explorer", "explorer": "explorer", "word": "winword", "ворд": "winword",
    "excel": "excel", "эксель": "excel", "outlook": "outlook", "аутлук": "outlook", "почта": "outlook",
    "paint": "mspaint", "пейнт": "mspaint", "powerpoint": "powerpnt",
}


def email_draft(to: str, subject: str, body: str, files: List[Path]) -> Tuple[str, Path | None]:
    """Open a draft; returns (how, eml path when the fallback was used)."""
    payload = json.dumps({"to": to or "", "subject": subject or "", "body": body or "",
                          "files": [str(Path(f)) for f in files if Path(f).is_file()]}, ensure_ascii=False)
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _OUTLOOK],
            input=payload.encode("utf-8"), capture_output=True, timeout=60, creationflags=_NO_WINDOW,
        )
        if proc.returncode == 0 and b"ok" in proc.stdout:
            return "outlook", None
    except Exception:
        pass
    # any mail program: an unsent message opens as a new draft
    message = EmailMessage()
    message["To"] = to or ""
    message["Subject"] = subject or ""
    message["X-Unsent"] = "1"
    message.set_content(body or "")
    for file in files:
        path = Path(file)
        if path.is_file():
            message.add_attachment(path.read_bytes(), maintype="application", subtype="octet-stream", filename=path.name)
    eml = outputs_dir() / f"Черновик письма {time.strftime('%H-%M-%S')}.eml"
    eml.write_bytes(bytes(message))
    os.startfile(str(eml))  # type: ignore[attr-defined]
    return "eml", eml


def open_in(path: Path, program: str = "") -> str:
    exe = PROGRAMS.get((program or "").lower().strip(), "")
    if exe:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", exe, str(path)], creationflags=_NO_WINDOW)
            return exe
        except OSError:
            pass
    os.startfile(str(path))  # type: ignore[attr-defined]
    return "по умолчанию"


def launch(program: str) -> str:
    exe = PROGRAMS.get((program or "").lower().strip())
    if not exe:
        raise ValueError(f"Не знаю программу «{program}» — попросите выполнить команду, и я предложу её")
    subprocess.Popen(["cmd", "/c", "start", "", exe], creationflags=_NO_WINDOW)
    return exe
