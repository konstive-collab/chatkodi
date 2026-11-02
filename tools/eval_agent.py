"""Run the assistant scenarios (tools/agent_cases.json) end to end.

Each scenario is a conversation with a fresh assistant: files are attached
from <corpus>/inbox, the library is <corpus>/docs. A turn passes when the
chosen action, produced files and answer text match the gold checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict


def check_turn(turn: Dict[str, Any], result) -> tuple[bool, Dict[str, Any]]:
    text = result.text or ""
    # read-only commands run without asking: then they are listed in the answer itself
    commands = " ".join(result.proposal.commands) if result.proposal else text
    lowered = text.lower()
    details: Dict[str, Any] = {}
    ok = True
    if turn.get("expect_action"):
        details["action"] = result.action
        ok = ok and result.action == turn["expect_action"]
    if turn.get("expect_files"):
        got = [f.path.suffix.lower() for f in result.files]
        details["files"] = got
        ok = ok and all(ext in got for ext in turn["expect_files"])
    for key in sorted(k for k in turn if k.startswith("expect_any")):
        if turn.get(key):
            hits = [item for item in turn[key] if item.lower() in lowered]
            details[key] = hits
            ok = ok and bool(hits)
    if turn.get("forbid_any"):
        bad = [item for item in turn["forbid_any"] if item.lower() in lowered]
        details["forbidden"] = bad
        ok = ok and not bad
    if "expect_proposal" in turn:
        # an action waiting for «Выполнить» (nothing is done by the eval itself)
        details["proposal"] = result.proposal is not None
        ok = ok and details["proposal"] == bool(turn["expect_proposal"])
    if turn.get("expect_found_any"):
        names = " ".join(Path(p).name for p in getattr(result, "found", []) or []).lower()
        hits = [item for item in turn["expect_found_any"] if item.lower() in names]
        details["found_hits"] = hits
        ok = ok and bool(hits)
    if turn.get("expect_command_any"):
        hits = [item for item in turn["expect_command_any"] if item.lower() in commands.lower()]
        details["command_hits"] = hits
        ok = ok and bool(hits)
    if turn.get("expect_no_files"):
        details["files"] = [f.path.name for f in result.files]
        ok = ok and not result.files
    if turn.get("expect_file_text_any") or turn.get("expect_slides_min") or turn.get("expect_pptx_background"):
        # what is inside the produced file, not only the answer
        made = next((f.path for f in result.files if f.path.suffix.lower() in {".pptx", ".docx"}), None)
        inside, slides, background = "", 0, ""
        if made is not None and made.suffix.lower() == ".pptx":
            from pptx import Presentation

            from core.agent.slides import pptx_outline

            inside = pptx_outline(made, limit=100000)
            prs = Presentation(str(made))
            slides = len(prs.slides)
            fill = prs.slide_master.background.fill
            try:
                background = str(fill.fore_color.rgb)
            except Exception:
                background = ""
        elif made is not None:
            from core.agent.outputs import docx_to_markdown

            inside = docx_to_markdown(made)
        if turn.get("expect_file_text_any"):
            hits = [item for item in turn["expect_file_text_any"] if item.lower() in inside.lower()]
            details["file_hits"] = hits
            ok = ok and bool(hits)
        if turn.get("expect_slides_min"):
            details["slides"] = slides
            ok = ok and slides >= int(turn["expect_slides_min"])
        if turn.get("expect_pptx_background"):
            details["background"] = background
            ok = ok and background.upper() == turn["expect_pptx_background"].upper()
    if not text.strip():
        ok = False
        details["empty"] = True
    return ok, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--only", default="", help="substring of scenario names")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from core.agent.agent import Agent
    from core.agent.attachments import AttachmentStore
    from core.chat_mode import ChatMode
    from tools.eval_synthetic import build_manager

    corpus = Path(args.corpus).resolve()
    scenarios = json.loads((corpus / "agent_cases.json").read_text(encoding="utf-8"))["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if args.only.lower() in s["name"].lower()]
    started = time.perf_counter()
    manager, llm = build_manager(corpus / "docs", True)
    manager.set_mode(ChatMode.RAG)
    setup = time.perf_counter() - started

    from core.agent.instructions import save_instructions
    from core.agent.memory import get_memory

    rows = []
    for index, scenario in enumerate(scenarios):
        save_instructions(scenario.get("instructions", ""))  # personal instructions of this scenario
        if not scenario.get("keep_memory"):
            get_memory().clear()  # a scenario starts with an empty memory unless it tests memory between chats
        agent = Agent(manager, store=AttachmentStore(corpus / "_attachments" / f"s{index}"))
        print(f"== {scenario['name']}", flush=True)
        for turn in scenario["turns"]:
            files = [corpus / "inbox" / name for name in turn.get("attach", [])]
            t0 = time.perf_counter()
            try:
                result = agent.handle(turn.get("message", ""), files)
            except Exception as exc:  # report, do not abort
                import traceback

                traceback.print_exc()
                from core.agent.agent import AgentResult

                result = AgentResult(f"EXCEPTION: {exc!r}", action="error")
            elapsed = time.perf_counter() - t0
            ok, details = check_turn(turn, result)
            if turn.get("remember"):
                # the app does this in the background after every answer
                details["remembered"] = agent.remember(turn.get("message", ""))
                ok = ok and bool(details["remembered"])
            if turn.get("run"):
                # read-only system questions: the commands must really work
                import re as _re

                runs = None
                if result.proposal is None and "Выполнено без запроса" in result.text:
                    # already run by the assistant (reading never asks): exit codes and output are in the answer
                    runs = [(int(code), out.strip()) for code, out in _re.findall(r"\(код (-?\d+)\)\n(.*?)(?=\n> |\n```|\Z)", result.text, _re.S)]
                elif result.proposal is not None and not result.proposal.dangerous:
                    from core.agent.commands import run_powershell

                    runs = [(r.exit_code, r.output) for r in (run_powershell(c, timeout=60) for c in result.proposal.commands)]
                if not runs:
                    ok, details["run"] = False, "no safe commands"
                else:
                    details["run"] = [(code, out[:160]) for code, out in runs]
                    ok = ok and all(code == 0 and out.strip() and out.strip() != "[нет вывода]" for code, out in runs)
                    if turn.get("expect_output_regex"):
                        joined = "\n".join(out for _code, out in runs)
                        hit = bool(_re.search(turn["expect_output_regex"], joined))
                        details["output_check"] = hit
                        ok = ok and hit
            rows.append({
                "scenario": scenario["name"],
                "message": turn.get("message", ""),
                "attach": turn.get("attach", []),
                "ok": ok,
                "seconds": round(elapsed, 2),
                "action": result.action,
                "route": result.route,
                "files": [str(f.path) for f in result.files],
                "details": details,
                "answer": result.text[:5000],
            })
            label = turn.get("message") or f"[{', '.join(turn.get('attach', []))}]"
            print(f"  [{'OK ' if ok else 'BAD'}] {elapsed:6.2f}s {result.action:<14} {label}", flush=True)

    times = [r["seconds"] for r in rows]
    payload = {
        "setup_seconds": round(setup, 1),
        "passed": sum(r["ok"] for r in rows),
        "total": len(rows),
        "accuracy": round(sum(r["ok"] for r in rows) / max(1, len(rows)), 3),
        "median_s": round(statistics.median(times), 2) if times else 0,
        "max_s": round(max(times), 2) if times else 0,
        "rows": rows,
    }
    report = Path(args.report) if args.report else corpus / "report_agent.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    get_memory().clear()
    print(json.dumps({k: payload[k] for k in ("setup_seconds", "passed", "total", "accuracy", "median_s", "max_s")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CHATKODI_DISABLE_TELEMETRY", "1")
    raise SystemExit(main())
