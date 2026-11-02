"""Agent actions beyond answers: permissions, files with undo, memory, plans."""

from __future__ import annotations

import pytest

from core.agent.agent import Agent, AgentResult
from core.agent.attachments import AttachmentStore
from core.agent.commands import CommandProposal
from core.agent.permissions import CHANGE, DANGER, IRREVERSIBLE, READ, command_risk, needs_confirmation


class _NoLLM:
    def is_available(self):
        return False


class _Manager:
    llm_model = _NoLLM()
    hybrid_search = None


@pytest.mark.parametrize(
    "command, names, risk",
    [
        ("Get-ChildItem C:\\ | Sort-Object Length", ["Get-ChildItem", "Sort-Object"], READ),
        ("ipconfig /all", ["ipconfig"], READ),
        ("Copy-Item a.txt b.txt", ["Copy-Item"], CHANGE),
        ("Get-Process > out.txt", ["Get-Process"], CHANGE),
        ("Stop-Process -Name notepad", ["Stop-Process"], DANGER),
        ("Set-ItemProperty HKLM:\\Software\\X -Name a -Value 1", ["Set-ItemProperty"], DANGER),
        ("somethingunknown.exe", ["somethingunknown.exe"], DANGER),
        ("Remove-Item C:\\temp\\old -Recurse", ["Remove-Item"], IRREVERSIBLE),
        ("Format-Volume -DriveLetter E", ["Format-Volume"], IRREVERSIBLE),
    ],
)
def test_command_risk(command, names, risk):
    assert command_risk(command, names) == risk


def test_autonomy_levels():
    assert not any(needs_confirmation(READ, level) for level in ("ask", "auto", "full"))
    assert needs_confirmation(CHANGE, "ask") and not needs_confirmation(CHANGE, "auto")
    assert needs_confirmation(DANGER, "auto") and not needs_confirmation(DANGER, "full")
    assert all(needs_confirmation(IRREVERSIBLE, level) for level in ("ask", "auto", "full"))


def test_find_files_prefers_names_and_types(tmp_path):
    from core.agent.files import find_files

    library, desktop = tmp_path / "lib", tmp_path / "desk"
    (library / "Акты").mkdir(parents=True)
    desktop.mkdir()
    for path in (library / "Акты" / "Акт_приемки_октябрь.docx", desktop / "акт приемки.pdf", desktop / "Счёт 15.pdf", desktop / "notes.bin"):
        path.write_bytes(b"x")
    roots = [(library, "библиотека"), (desktop, "Рабочий стол")]
    found = find_files("найди акт приёмки", roots)
    assert {f.path.name for f in found} == {"Акт_приемки_октябрь.docx", "акт приемки.pdf"}
    assert [f.path.name for f in find_files("акт приемки pdf", roots)] == ["акт приемки.pdf"]
    assert find_files("найди файл", roots) == []


def test_moves_never_overwrite_and_undo(tmp_path, monkeypatch):
    from core.agent import files

    monkeypatch.setattr(files, "undo_dir", lambda: tmp_path)
    a, b, taken = tmp_path / "a.docx", tmp_path / "b.docx", tmp_path / "c.docx"
    a.write_text("a")
    taken.write_text("c")
    target = tmp_path / "Архив" / "a.docx"
    done, errors, log = files.apply_moves([(str(a), str(target)), (str(b), str(tmp_path / "x.docx")), (str(taken), str(target))])
    assert len(done) == 1 and target.exists() and not a.exists() and taken.exists()
    assert len(errors) == 2  # a missing file and a would-be overwrite
    count, problems = files.undo_moves(log)
    assert count == 1 and not problems and a.exists() and not target.exists() and not log.exists()


def test_memory_dedupes_and_survives_reload(tmp_path):
    from core.agent.memory import MemoryStore

    path = tmp_path / "memory.json"
    memory = MemoryStore(path)
    assert memory.add("Пользователь — инженер ПТО ООО «СтройЛаб»")
    assert not memory.add("пользователь инженер ПТО ООО СтройЛаб")
    assert not memory.add("коротко")
    assert memory.add("Письма подписывать от имени Петровой А. С.")
    again = MemoryStore(path)
    assert [f.text for f in again.facts] == [f.text for f in memory.facts]
    assert "СтройЛаб" in again.prompt_block()
    again.remove(again.facts[0].id)
    assert len(MemoryStore(path).facts) == 1


def test_route_builds_plan_from_steps(tmp_path):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    step = {"source": "none", "output": "chat", "language": "", "search_query": ""}
    raw = {
        "results": ["письмо", "черновик в почте"],
        "action": "write", "source": "none", "output": "chat", "language": "", "task": "t", "search_query": "",
        "steps": [dict(step, action="write", task="напиши письмо"), dict(step, action="chat", task="привет"),
                  dict(step, action="app", task="открой черновик в почте")],
    }
    agent._json = lambda prompt, schema, max_tokens=0: raw
    route = agent.route("напиши письмо и открой черновик", [])
    assert [s["action"] for s in route["steps"]] == ["write", "app"]
    raw["results"] = ["письмо"]  # one result asked for: the steps are only preparation
    assert "steps" not in agent.route("напиши письмо", [])


def test_plan_pauses_for_permission_and_continues(tmp_path):
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path))
    agent._do_write = lambda task, step, items, say: AgentResult(f"готово: {task}", action="write")
    waiting = CommandProposal(["Черновик"], "", kind="app", payload={"op": "nothing"}, risk=CHANGE)
    agent._do_app = lambda task, step, items, say: AgentResult("письмо готово", action="app", proposal=waiting)
    route = {"steps": [{"action": "write", "task": "текст"}, {"action": "app", "task": "почта"}, {"action": "write", "task": "итог"}]}
    first = agent._run_plan("m", route, [], lambda _t: None)
    assert "### 2." in first.text and "### 3." not in first.text and first.proposal is waiting
    assert "Дальше по плану" in first.text
    final = agent.run_commands(waiting)
    assert "### 3." in final.text and "готово: итог" in final.text and agent._pending_plan is None


def test_permitted_move_runs_and_can_be_undone(tmp_path, monkeypatch):
    from core.agent import files

    monkeypatch.setattr(files, "undo_dir", lambda: tmp_path)
    agent = Agent(_Manager(), store=AttachmentStore(tmp_path / "att"))
    source = tmp_path / "отчёт.docx"
    source.write_text("x")
    moves = [(str(source), str(tmp_path / "Отчёт_октябрь.docx"))]

    agent.settings.autonomy = "ask"
    asked = agent._permit(CommandProposal(["move"], "", kind="files", payload={"op": "move", "moves": moves}, risk=CHANGE), "Переименую.", lambda _t: None)
    assert asked.proposal is not None and source.exists()

    agent.settings.autonomy = "auto"
    done = agent._permit(CommandProposal(["move"], "", kind="files", payload={"op": "move", "moves": moves}, risk=CHANGE), "Переименую.", lambda _t: None)
    assert done.proposal is None and done.undo_log is not None and not source.exists()
    assert "Отменено" in agent.undo(done.undo_log) and source.exists()
