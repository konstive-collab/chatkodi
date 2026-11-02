import re
import zlib
from types import SimpleNamespace

import numpy as np

from core.chat.manager import ChatManager


class _HashingEmbedding:
    """Small domain-neutral test double for sentence similarity."""

    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for text in texts:
            normalized = re.sub(r"\s+", " ", str(text).lower().replace("ё", "е"))
            vector = np.zeros(512, dtype=np.float32)
            for token in re.findall(r"[a-zа-я0-9]+", normalized):
                vector[zlib.crc32(token.encode("utf-8")) % vector.size] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors, dtype=np.float32)


def test_focused_fact_answer_preserves_source_negation_without_domain_rules():
    search = SimpleNamespace(
        embedding_model=_HashingEmbedding(),
        config=SimpleNamespace(data={}),
    )
    manager = ChatManager(hybrid_search=search)
    manager._collect_structured_candidates = lambda *args, **kwargs: [
        (
            0.95,
            {
                "doc_id": "status",
                "source": "status.txt",
                "text": "Сведения о запуске реактора в сменном отчете не представлены.",
            },
        ),
        (
            0.80,
            {
                "doc_id": "shipping",
                "source": "shipping.txt",
                "text": "Партия упаковки передана на склад готовой продукции.",
            },
        ),
    ]

    evidence = manager._collect_focused_fact_evidence(
        "Есть ли сведения о запуске реактора в сменном отчете?",
        {"relevant_docs": [(1.0, {"doc_id": "status"})]},
    )
    answer = manager._format_focused_fact_answer(evidence)

    assert "не представлены" in answer
    assert "status.txt" in answer
    assert "Партия упаковки" not in answer


def test_narrative_summary_uses_scoped_document_fragments_not_statistics():
    manager = ChatManager()
    question = "Что происходило на каховском направлении на 6 июня 2025 года?"
    content = (
        "РАЗВЕДЫВАТЕЛЬНАЯ ИНФОРМАЦИЯ\n"
        "НА 6 ИЮНЯ 2025 ГОДА\n"
        "На Каховском направлении ведет воздушную разведку приграничных районов.\n"
        "На Каховском направлении противник сосредотачивает усилия на недопущении продвижения.\n"
        "На Приморском ТН провел одну контратаку с целью вернуть утраченные позиции:\n"
        "в 15.45 6 июня с.г. – шгр в зап. части НОВОФЕДОРОВКА. Успеха не имел.\n"
        "рБпЛА №33 (Каховское направление):\n"
        "10.20 – БМП, северная окраина ЛЬВОВО;\n"
        "20.10 – антенный пост, сев-зап окр ЧЕРНОБАЕВКА.\n"
        "Таким образом, на Каховском направлении контратакует и ведет воздушную разведку."
    )
    search_result = {
        "relevant_docs": [
            (
                1.0,
                {
                    "name": "06.docx",
                    "filename": "06.docx",
                    "content": content,
                    "date_match": True,
                },
            )
        ],
        "relevant_docs_all": [
            (
                1.0,
                {
                    "name": "06.docx",
                    "filename": "06.docx",
                    "content": content,
                    "date_match": True,
                },
            )
        ],
    }

    answer = manager._build_direct_rag_answer(question, search_result)

    assert answer
    assert "воздушную разведку" in answer
    assert "контратаку" in answer
    assert "рБпЛА №33" in answer
    assert "ЛЬВОВО" in answer
    assert "Краткий вывод" not in answer
    assert "Ответ намеренно сокращён" not in answer
