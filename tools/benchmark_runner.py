#!/usr/bin/env python3
"""Небольшой регрессионный прогон для контрольных кейсов ChatKodi 1.7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.query_understanding import analyze_question
from core.fact_extraction import build_evidence_records
from core.analytics import build_statistics
from core.answer_builders import build_structured_answer


def _load_cases(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    parser = argparse.ArgumentParser(description='Прогон benchmark-кейсов без GUI')
    parser.add_argument('cases', nargs='?', default='benchmark_cases.sample.json')
    args = parser.parse_args()
    path = (Path(args.cases).resolve() if Path(args.cases).exists() else (_ROOT / args.cases).resolve())
    cases = _load_cases(path)
    report = {'cases': [], 'summary': {'total': 0, 'passed': 0}}
    for case in cases.get('cases', []):
        question = case['question']
        intent = analyze_question(question)
        candidates = [
            (
                1.0 + idx * 0.1,
                {
                    'doc_id': str(idx),
                    'source': frag.get('source', f'doc{idx}.txt'),
                    'chunk_id': idx,
                    'text': frag['text'],
                    'date_match': frag.get('date_match', False),
                    'answer_rescue': frag.get('answer_rescue', False),
                },
            )
            for idx, frag in enumerate(case.get('fragments', []), 1)
        ]
        records = build_evidence_records(intent, candidates, limit=50)
        stats = build_statistics(records)
        answer = build_structured_answer(intent, records, stats)
        text = answer.text if answer else ''
        passed = True
        for token in case.get('must_contain', []):
            if token.lower() not in text.lower():
                passed = False
                break
        report['cases'].append({
            'question': question,
            'passed': passed,
            'answer': text,
            'records': len(records),
            'mode': getattr(answer, 'mode', ''),
        })
        report['summary']['total'] += 1
        report['summary']['passed'] += int(passed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['summary']['passed'] == report['summary']['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
