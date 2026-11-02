"""Console actions: the model proposes PowerShell commands, the user confirms
each run in the window, the program executes and shows the output."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple

# Commands that can destroy data or the system are shown with a warning; the
# user still decides, but never by a single careless click.
_DANGEROUS = re.compile(
    r"\b(?:Remove-Item|rm|rmdir|del|erase|rd|Format-Volume|format(?![-\w])|diskpart|Clear-Disk|Stop-Computer|Restart-Computer|"
    r"shutdown|reg\s+delete|Remove-ItemProperty|Set-ExecutionPolicy|bcdedit|cipher\s+/w|Clear-Content|"
    r"Uninstall-|net\s+user|takeown|icacls)\b",
    re.IGNORECASE,
)
OUTPUT_LIMIT = 8000


@dataclass
class CommandProposal:
    """An action waiting for the user's «Выполнить»: PowerShell commands, file
    operations or an application step (``kind``), with its program-side risk."""

    commands: List[str]
    explanation: str
    task: str = ""
    dangerous: List[str] = field(default_factory=list)
    risk: int = 0
    kind: str = "powershell"  # powershell | files | app
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        self.commands = [c.strip() for c in self.commands if c and c.strip()][:5]
        self.dangerous = [c for c in self.commands if _DANGEROUS.search(c)]


@dataclass
class CommandResult:
    command: str
    exit_code: int
    output: str


_WRAPPER = re.compile(r'^\s*(?:powershell(?:\.exe)?|pwsh)\s+(?:-\w+\s+)*-Command\s+["\']?(.*?)["\']?\s*$', re.IGNORECASE | re.S)

_VALIDATOR = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
[Console]::InputEncoding = [Text.Encoding]::UTF8
$items = [Console]::In.ReadToEnd() | ConvertFrom-Json
$out = @()
foreach ($code in $items) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseInput($code, [ref]$tokens, [ref]$errors)
    $problems = @($errors | ForEach-Object { $_.Message })
    $names = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true) |
        ForEach-Object { $_.GetCommandName() } | Where-Object { $_ } | Select-Object -Unique
    foreach ($name in $names) {
        if (-not (Get-Command $name -ErrorAction SilentlyContinue)) { $problems += "Команда «$name» не существует" }
    }
    $out += [pscustomobject]@{ p = @($problems); n = @($names) }
}
ConvertTo-Json -InputObject @($out) -Depth 4 -Compress
"""


def unwrap(command: str) -> str:
    """«powershell -Command "…"» around a command is redundant here: the inner part runs as is."""
    match = _WRAPPER.match(command or "")
    return match.group(1).strip() if match else (command or "").strip()


def analyze_powershell(commands: List[str]) -> List[Tuple[List[str], List[str]]]:
    """(problems, called command names) per command — PowerShell parses them,
    nothing is executed. No problems means the command is valid."""
    import json

    if not commands:
        return []
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", _VALIDATOR],
            input=json.dumps(list(commands), ensure_ascii=False).encode("utf-8"),
            capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        data = json.loads(proc.stdout.decode("utf-8", errors="replace") or "[]")
    except Exception:
        return [([], []) for _ in commands]  # the check is a helper, not a gate
    if isinstance(data, dict):
        data = [data]
    result = []
    for index in range(len(commands)):
        item = data[index] if isinstance(data, list) and index < len(data) and isinstance(data[index], dict) else {}
        listify = lambda v: [str(x) for x in (v if isinstance(v, list) else [v]) if x]
        result.append((listify(item.get("p")), listify(item.get("n"))))
    return result


def validate_powershell(commands: List[str]) -> List[List[str]]:
    """Parse errors and unknown cmdlets per command."""
    return [problems for problems, _names in analyze_powershell(commands)]

def run_powershell(command: str, *, timeout: int = 120, cwd: str | None = None) -> CommandResult:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    wrapped = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference='SilentlyContinue'; " + command
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", wrapped],
            capture_output=True,
            timeout=timeout,
            cwd=cwd or os.path.expanduser("~"),
            creationflags=creationflags,
        )
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        text = (out + ("\n" + err if err.strip() else "")).strip()
        if len(text) > OUTPUT_LIMIT:
            text = text[:OUTPUT_LIMIT] + "\n…(вывод сокращён)"
        return CommandResult(command, proc.returncode, text)
    except subprocess.TimeoutExpired:
        return CommandResult(command, -1, f"Команда не завершилась за {timeout} с и была остановлена.")
    except Exception as exc:
        return CommandResult(command, -1, f"Не удалось запустить команду: {exc}")
