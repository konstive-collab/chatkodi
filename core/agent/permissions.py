"""How much the assistant may do without asking.

Every action gets a risk class from the program (not only from the model):

    READ         — looks only: search, read, show            → never asks
    CHANGE       — creates or changes something reversible   → asks in «Только с разрешения»
    DANGER       — deletes to the bin, overwrites, system    → asks in «Автомат» too
    IRREVERSIBLE — permanent deletion, formatting, disks     → always asks, at every level

Levels (settings): ask — «Только с разрешения», auto — «Автомат»,
full — «Полная самостоятельность».
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

READ, CHANGE, DANGER, IRREVERSIBLE = 0, 1, 2, 3
RISK_NAMES = {READ: "только чтение", CHANGE: "изменение", DANGER: "опасно", IRREVERSIBLE: "необратимо"}
LEVELS = {"ask": "Только с разрешения", "auto": "Автомат", "full": "Полная самостоятельность"}

_READ_VERBS = {
    "get", "select", "measure", "test", "find", "search", "show", "format", "where", "sort", "group",
    "compare", "resolve", "write", "convertto", "convertfrom", "join", "split", "foreach", "read", "trace",
}
_CHANGE_VERBS = {"new", "copy", "move", "rename", "set", "add", "start", "export", "compress", "expand",
                 "invoke", "out", "push", "pop", "update", "save", "send", "import", "mount", "register"}
_DANGER_VERBS = {"remove", "clear", "stop", "restart", "uninstall", "disable", "enable", "unregister",
                 "reset", "suspend", "dismount", "revoke", "grant", "block", "unblock", "install", "lock", "unlock"}
_READ_ALIASES = {
    "dir", "ls", "gci", "cat", "type", "gc", "echo", "where", "select", "sort", "measure", "ft", "fl", "%", "?",
    "foreach", "cd", "pwd", "sl", "gl", "findstr", "more",
    "out-string", "out-host", "out-null", "set-location", "push-location", "pop-location",
}
_READ_PROGRAMS = {
    "ipconfig", "systeminfo", "hostname", "whoami", "tasklist", "ping", "nslookup", "tracert", "ver", "getmac",
    "nvidia-smi", "driverquery", "query", "netstat", "route", "arp",
}
_IRREVERSIBLE = re.compile(
    r"\b(?:Format-Volume|Clear-Disk|Initialize-Disk|diskpart|cipher\s+/w|vssadmin\s+delete|bcdedit|"
    r"Remove-Partition|Clear-RecycleBin|wevtutil\s+cl)\b",
    re.IGNORECASE,
)
_PERMANENT_DELETE = re.compile(r"\b(?:Remove-Item|rm|rmdir|rd|del|erase|ri)\b", re.IGNORECASE)
_SYSTEM_TARGET = re.compile(r"HKLM:|HKEY_LOCAL_MACHINE|\\Windows\\|System32|Program Files", re.IGNORECASE)


def command_risk(command: str, names: Iterable[str]) -> int:
    """Risk of one PowerShell command from the commands it calls (PowerShell AST)."""
    text = command or ""
    if _IRREVERSIBLE.search(text):
        return IRREVERSIBLE
    if _PERMANENT_DELETE.search(text):
        return IRREVERSIBLE  # PowerShell deletes past the recycle bin
    risk = READ
    for name in names:
        low = name.lower().strip()
        base = low.rsplit("\\", 1)[-1].removesuffix(".exe")
        verb = low.split("-", 1)[0] if "-" in low else ""
        if low in _READ_ALIASES or base in _READ_PROGRAMS:
            continue
        if verb in _DANGER_VERBS:
            risk = max(risk, DANGER)
        elif verb in _CHANGE_VERBS:
            risk = max(risk, CHANGE)
        elif verb in _READ_VERBS:
            continue
        elif low in _READ_ALIASES or base in _READ_PROGRAMS:
            continue
        else:
            risk = max(risk, DANGER)  # an unknown program: assume it can change things
    if re.search(r"(?<![>\d])>{1,2}(?!&)", text):  # redirection writes a file
        risk = max(risk, CHANGE)
    if risk >= CHANGE and _SYSTEM_TARGET.search(text):
        risk = max(risk, DANGER)
    return risk


def needs_confirmation(risk: int, level: str) -> bool:
    if risk <= READ:
        return False
    if risk >= IRREVERSIBLE:
        return True
    if level == "full":
        return False
    if level == "auto":
        return risk >= DANGER
    return True  # «Только с разрешения»


def describe(risk: int) -> str:
    return RISK_NAMES.get(risk, "изменение")


def worst(risks: List[int]) -> int:
    return max(risks) if risks else READ
