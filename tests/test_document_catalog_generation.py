from pathlib import Path
import shutil
import zipfile

from docx import Document

from core.document_catalog import DocumentCatalog


def _docx(path: Path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(path)


def test_catalog_classifies_deduplicates_and_blocks_unsafe_templates(tmp_path):
    sport = tmp_path / "Методика ФП.docx"
    _docx(sport, "Физическая подготовка. Упражнения, бег и подтягивания.")
    duplicate = tmp_path / "Копия методики.docx"
    shutil.copy2(sport, duplicate)
    legal = tmp_path / "Рапорт на командировку.docx"
    _docx(legal, "РАПОРТ Прошу направить в командировку.")
    broken = tmp_path / "broken.docx"
    broken.write_text("not a docx", encoding="utf-8")
    (tmp_path / "helper.py").write_text("pass", encoding="utf-8")

    catalog = DocumentCatalog(tmp_path, tmp_path / "catalog.json")
    records = catalog.scan()
    by_name = {item.name: item for item in records}

    assert by_name[sport.name].topic == "sport"
    assert bool(by_name[duplicate.name].duplicate_of) != bool(by_name[sport.name].duplicate_of)
    assert by_name[legal.name].template_eligible
    assert not by_name[broken.name].valid
    assert not by_name["helper.py"].source_eligible
    assert catalog.summary()["invalid_docx"] == 1
    assert "text" not in (tmp_path / "catalog.json").read_text(encoding="utf-8")


def test_external_relationship_disqualifies_template(tmp_path):
    path = tmp_path / "Рапорт.docx"
    _docx(path, "РАПОРТ")
    rebuilt = tmp_path / "rebuilt.docx"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            data = source.read(info)
            if info.filename == "word/_rels/document.xml.rels":
                data = data.replace(
                    b"</Relationships>",
                    b'<Relationship Id="rExt" Type="http://example/link" Target="https://example.test" TargetMode="External"/></Relationships>',
                )
            target.writestr(info, data)
    rebuilt.replace(path)
    record = DocumentCatalog(tmp_path).scan()[0]
    assert record.external_relationships == 1
    assert not record.template_eligible
