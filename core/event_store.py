"""Лёгкое on-disk хранилище событий для структурированных ответов ChatKodi 1.7."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import re
from typing import Dict, Iterable, List

from core.entity_resolution import entity_overlap_score
from core.domain_lexicon import get_domain_lexicon
from core.fact_extraction import IndexedEvent, extract_indexed_events, indexed_event_to_evidence, infer_dates_from_source_for_intent
from core.logging_utils import get_logger
from core.paths import ensure_parent, resolve_project_path
from core.query_understanding import QueryUnderstandingResult, normalize_question

logger = get_logger(__name__)
_DOMAIN = get_domain_lexicon()


def _strict_uav_scope(intent: QueryUnderstandingResult) -> bool:
    q = normalize_question(getattr(intent, 'raw_question', '') or '')
    return bool((getattr(intent, 'identifiers', []) or []) and _DOMAIN.mentions_family(q, 'рбпла'))


def _event_matches_strict_uav_scope(event: IndexedEvent, intent: QueryUnderstandingResult) -> bool:
    """True only when the event belongs to the requested UAV/RBPLA section/object."""
    if not _strict_uav_scope(intent):
        return True
    normalized = normalize_question(event.sentence)
    meta = event.metadata or {}
    section_family = normalize_question(str(meta.get('section_family') or ''))
    section_number = str(meta.get('section_number') or '').lstrip('0')
    section_header = normalize_question(str(meta.get('section_header') or ''))
    section_terms = [normalize_question(str(x)) for x in (meta.get('section_terms') or []) if x]
    for ident in (getattr(intent, 'identifiers', []) or []):
        ident = str(ident).lstrip('0') or '0'
        if _DOMAIN.family_matches(section_family, 'рбпла') and section_number == ident:
            return True
        if re.search(rf"\b(?:р\s*)?бпла\s*(?:№|#)?\s*0*{re.escape(ident)}\b", normalized, flags=re.IGNORECASE):
            return True
        for term in (getattr(intent, 'object_terms', []) or []):
            term_norm = normalize_question(str(term))
            if term_norm and (term_norm in normalized or term_norm in section_header or term_norm in section_terms):
                return True
    return False


class EventStore:
    def __init__(
        self,
        db_path: str = 'events.db',
        meta_path: str = 'events_meta.json',
        max_sync_chunks: int = 0,
        max_sync_seconds: float = 0.0,
    ) -> None:
        self.db_path = resolve_project_path(db_path)
        self.meta_path = resolve_project_path(meta_path)
        self.max_sync_chunks = int(max_sync_chunks or 0)
        self.max_sync_seconds = float(max_sync_seconds or 0.0)
        ensure_parent(self.db_path)
        ensure_parent(self.meta_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS chunk_index (
                    chunk_key TEXT PRIMARY KEY,
                    doc_id TEXT,
                    source TEXT,
                    chunk_id TEXT,
                    signature TEXT NOT NULL,
                    indexed_at REAL NOT NULL
                )
                '''
            )
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_key TEXT NOT NULL,
                    doc_id TEXT,
                    source TEXT,
                    chunk_id TEXT,
                    sentence TEXT,
                    normalized_sentence TEXT,
                    entities_json TEXT,
                    methods_json TEXT,
                    locations_json TEXT,
                    directions_json TEXT,
                    dates_json TEXT,
                    quantities_json TEXT,
                    target TEXT,
                    action TEXT,
                    event_type TEXT,
                    event_time TEXT,
                    polarity TEXT,
                    confidence REAL,
                    metadata_json TEXT,
                    FOREIGN KEY(chunk_key) REFERENCES chunk_index(chunk_key) ON DELETE CASCADE
                )
                '''
            )
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(events)")}
            if "event_type" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN event_type TEXT")
            if "event_time" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN event_time TEXT")
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_doc ON events(doc_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_chunk ON events(chunk_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_norm ON events(normalized_sentence)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_action ON events(action)')
            conn.commit()

    def _chunk_key(self, doc_id: str, chunk_id: object) -> str:
        return f'{doc_id}:{chunk_id}'

    def _signature(self, text: str) -> str:
        payload = 'universal-claim-v10-structural-context-scope\n' + (text or '')
        return hashlib.sha1(payload.encode('utf-8', errors='ignore')).hexdigest()

    def sync_chunks(self, chunks: Iterable[dict]) -> Dict[str, int]:
        chunks = list(chunks or [])
        chunks_total = len(chunks)
        partial = False
        if self.max_sync_chunks > 0 and len(chunks) > self.max_sync_chunks:
            chunks = chunks[: self.max_sync_chunks]
            partial = True
        started = time.time()
        inserted = 0
        updated = 0
        unchanged = 0
        removed = 0
        processed = 0
        with self._connect() as conn:
            existing = {row['chunk_key']: row['signature'] for row in conn.execute('SELECT chunk_key, signature FROM chunk_index')}
            incoming_keys = set()
            for chunk in chunks:
                if self.max_sync_seconds > 0 and (time.time() - started) > self.max_sync_seconds:
                    partial = True
                    logger.warning("Event store sync stopped by time budget: processed=%s total=%s", processed, chunks_total)
                    break
                processed += 1
                doc_id = str(chunk.get('doc_id') or '')
                chunk_id = chunk.get('chunk_id')
                text = str(chunk.get('text') or '')
                source = str(chunk.get('source') or chunk.get('doc_name') or chunk.get('name') or '')
                if not doc_id or chunk_id is None or not text.strip():
                    continue
                key = self._chunk_key(doc_id, chunk_id)
                incoming_keys.add(key)
                sig = self._signature(text)
                old_sig = existing.get(key)
                if old_sig == sig:
                    unchanged += 1
                    continue
                conn.execute('DELETE FROM events WHERE chunk_key = ?', (key,))
                conn.execute(
                    'INSERT OR REPLACE INTO chunk_index(chunk_key, doc_id, source, chunk_id, signature, indexed_at) VALUES(?,?,?,?,?,?)',
                    (key, doc_id, source, str(chunk_id), sig, time.time()),
                )
                events = extract_indexed_events(
                    doc_id=doc_id,
                    source=source,
                    chunk_id=chunk_id,
                    text=text,
                    metadata={
                        'chunk_index': chunk.get('chunk_index'),
                        'start_char': chunk.get('start_char'),
                        'end_char': chunk.get('end_char'),
                        'line_start': chunk.get('line_start'),
                        'line_end': chunk.get('line_end'),
                        'span_key': chunk.get('span_key'),
                        'format_family': chunk.get('format_family'),
                        'document_date': chunk.get('document_date'),
                    },
                )
                for event in events:
                    conn.execute(
                        '''
                        INSERT INTO events(
                            chunk_key, doc_id, source, chunk_id, sentence, normalized_sentence,
                            entities_json, methods_json, locations_json, directions_json, dates_json, quantities_json,
                            target, action, event_type, event_time, polarity, confidence, metadata_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ''',
                        (
                            key,
                            event.doc_id,
                            event.source,
                            str(event.chunk_id),
                            event.sentence,
                            event.normalized_sentence,
                            json.dumps(event.entities, ensure_ascii=False),
                            json.dumps(event.methods, ensure_ascii=False),
                            json.dumps(event.locations, ensure_ascii=False),
                            json.dumps(event.directions, ensure_ascii=False),
                            json.dumps(event.event_dates, ensure_ascii=False),
                            json.dumps(event.quantities, ensure_ascii=False),
                            event.target,
                            event.action,
                            event.event_type,
                            event.event_time,
                            event.polarity,
                            float(event.confidence),
                            json.dumps(event.metadata, ensure_ascii=False),
                        ),
                    )
                if old_sig is None:
                    inserted += 1
                else:
                    updated += 1
            stale = set(existing.keys()) - incoming_keys
            if stale and not partial:
                removed = len(stale)
                conn.executemany('DELETE FROM events WHERE chunk_key = ?', [(key,) for key in stale])
                conn.executemany('DELETE FROM chunk_index WHERE chunk_key = ?', [(key,) for key in stale])
            conn.commit()
        meta = {
            'chunks_total': chunks_total,
            'chunks_processed': processed,
            'partial': bool(partial),
            'inserted': inserted,
            'updated': updated,
            'unchanged': unchanged,
            'removed': removed,
            'synced_at': time.time(),
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info('Event store sync complete: processed=%s/%s partial=%s inserted=%s updated=%s unchanged=%s removed=%s', processed, chunks_total, partial, inserted, updated, unchanged, removed)
        return meta

    def _row_to_indexed_event(self, row: sqlite3.Row) -> IndexedEvent:
        return IndexedEvent(
            doc_id=str(row['doc_id'] or ''),
            source=str(row['source'] or ''),
            chunk_id=row['chunk_id'],
            sentence=str(row['sentence'] or ''),
            normalized_sentence=str(row['normalized_sentence'] or ''),
            entities=list(json.loads(row['entities_json'] or '[]')),
            methods=list(json.loads(row['methods_json'] or '[]')),
            locations=list(json.loads(row['locations_json'] or '[]')),
            directions=list(json.loads(row['directions_json'] or '[]')),
            event_dates=list(json.loads(row['dates_json'] or '[]')),
            quantities=list(json.loads(row['quantities_json'] or '[]')),
            target=str(row['target'] or ''),
            action=str(row['action'] or ''),
            event_type=str(row['event_type'] or 'any'),
            event_time=str(row['event_time'] or ''),
            polarity=str(row['polarity'] or 'neutral'),
            confidence=float(row['confidence'] or 0.0),
            metadata=json.loads(row['metadata_json'] or '{}'),
        )

    def query(self, intent: QueryUnderstandingResult, limit: int = 60) -> List[dict]:
        candidates: List[dict] = []
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM events ORDER BY confidence DESC, event_id DESC LIMIT 4000'
            ).fetchall()
        obj_terms = [normalize_question(x) for x in intent.object_terms + intent.identifiers if x]
        keywords = [normalize_question(x) for x in intent.keywords + intent.answer_terms if x]
        date_values = set()
        for dt in intent.dates:
            try:
                date_values.add(dt.strftime('%Y-%m-%d'))
            except Exception:
                continue
        if intent.start_date:
            date_values.add(intent.start_date.strftime('%Y-%m-%d'))
        if intent.end_date:
            date_values.add(intent.end_date.strftime('%Y-%m-%d'))

        for row in rows:
            event = self._row_to_indexed_event(row)
            if not _event_matches_strict_uav_scope(event, intent):
                continue
            score = float(event.confidence)
            normalized = event.normalized_sentence
            object_hits = sum(1 for term in obj_terms if term and term in normalized)
            meta = event.metadata or {}
            section_terms = [normalize_question(str(x)) for x in (meta.get('section_terms') or []) if x]
            section_header = normalize_question(str(meta.get('section_header') or ''))
            section_number = str(meta.get('section_number') or '').lstrip('0')
            for term in obj_terms:
                if term and (term in section_header or term in section_terms):
                    object_hits += 1
            for ident in (getattr(intent, 'identifiers', []) or []):
                ident_s = str(ident).lstrip('0')
                if ident_s and section_number and ident_s == section_number and _DOMAIN.mentions_family(intent.normalized_question, 'рбпла'):
                    object_hits += 1
            if _strict_uav_scope(intent) and _event_matches_strict_uav_scope(event, intent):
                object_hits = max(object_hits, 1)
            kw_hits = sum(1 for kw in keywords if kw and kw in normalized)
            struct_score, struct_matches = entity_overlap_score(intent.raw_question, event.sentence)
            entity_hits = len(struct_matches)
            if object_hits > 0 and meta.get('section_direction') and meta.get('section_direction') not in event.directions:
                event.directions = [str(meta.get('section_direction'))] + list(event.directions)
            score += 0.62 * object_hits + 0.42 * struct_score + 0.08 * kw_hits
            date_match = False
            if not event.event_dates:
                inferred_dates = infer_dates_from_source_for_intent(intent, event.source)
                if inferred_dates:
                    event.event_dates = inferred_dates
            has_event_dates = bool(event.event_dates)
            if date_values:
                event_dates = set(event.event_dates)
                if event_dates & date_values:
                    date_match = True
                    score += 0.4
                elif intent.start_date and intent.end_date and event_dates:
                    try:
                        parsed = [d for d in event.event_dates if d]
                        start_s = intent.start_date.strftime('%Y-%m-%d')
                        end_s = intent.end_date.strftime('%Y-%m-%d')
                        if any(start_s <= d <= end_s for d in parsed):
                            date_match = True
                            score += 0.35
                        else:
                            score -= 0.35
                    except Exception:
                        pass
                elif has_event_dates:
                    score -= 0.25
            if intent.question_type == 'location' and (event.locations or event.directions):
                score += 0.25
            if intent.question_type == 'method' and event.methods:
                score += 0.25
            if intent.question_type in {'count', 'frequency', 'summary', 'comparison', 'timeline', 'range_summary', 'activity_report'}:
                if event.action or event.target or event.methods:
                    score += 0.12
            if intent.statistics_focus == 'direction' and (event.directions or event.locations):
                score += 0.18
            elif intent.statistics_focus == 'method' and event.methods:
                score += 0.18
            elif intent.statistics_focus == 'target' and event.target:
                score += 0.18
            elif intent.statistics_focus == 'date' and event.event_dates:
                score += 0.18
            if intent.has_date_constraint and not date_match:
                # Любая указанная дата является жёсткой границей для фактологического evidence.
                continue
            has_object_query = bool(getattr(intent, 'object_terms', []) or getattr(intent, 'identifiers', []))
            if has_object_query and object_hits <= 0 and entity_hits <= 0:
                # Если пользователь спрашивает конкретную сущность (например рБпЛА №11),
                # не считаем весь документ только потому, что он попал в период.
                continue
            if object_hits <= 0 and entity_hits <= 0 and kw_hits <= 0 and not date_match:
                continue
            candidates.append({
                'event': event,
                'score': score,
                'date_match': date_match,
                'object_hits': object_hits,
                'entity_hits': entity_hits,
                'keyword_hits': kw_hits,
            })
        candidates.sort(key=lambda item: (item['score'], item['event'].confidence), reverse=True)
        return candidates[:limit]

    def query_as_evidence(self, intent: QueryUnderstandingResult, limit: int = 40):
        records = []
        for item in self.query(intent, limit=limit):
            records.append(indexed_event_to_evidence(
                item['event'],
                score=float(item['score']),
                date_match=bool(item['date_match']),
                object_hits=int(item['object_hits']),
                entity_hits=int(item['entity_hits']),
                keyword_hits=int(item['keyword_hits']),
            ))
        return records

    def get_stats(self) -> Dict[str, object]:
        with self._connect() as conn:
            chunk_count = int(conn.execute('SELECT COUNT(*) FROM chunk_index').fetchone()[0])
            event_count = int(conn.execute('SELECT COUNT(*) FROM events').fetchone()[0])
            top_actions = conn.execute(
                'SELECT action, COUNT(*) AS cnt FROM events WHERE action IS NOT NULL AND action != "" GROUP BY action ORDER BY cnt DESC LIMIT 5'
            ).fetchall()
        meta = {}
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding='utf-8'))
            except Exception:
                meta = {}
        return {
            'db_path': str(self.db_path),
            'chunk_count': chunk_count,
            'event_count': event_count,
            'top_actions': [(str(row['action']), int(row['cnt'])) for row in top_actions],
            'meta': meta,
        }
