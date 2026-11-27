"""Evidence Gate: замечание проходит, только если его доказательства проверяются по документу.

Принцип — fail closed: всё, что нельзя надёжно проверить, отклоняется с причиной; любая
внутренняя ошибка проверки — тоже отказ, а не подтверждение.

Документ и сегментация (до проверки замечаний):
- сегментация согласована с документом (verify_segmentation: цитаты, места, покрытие,
  нормализованный текст); иначе EvidenceGateError — проверять нечем.

Замечание:
- поля нужных типов, обязательные поля заполнены, важность, категория, критерий, вид
  доказательства и происхождение — из допустимых значений;
- привязано к этому документу: document_sha256 совпадает (другой файл или другая версия
  документа не принимаются — замечание нужно построить заново);
- замечание правила (origin="rule") совпадает с записью каталога запущенных правил
  (ID, версия, важность, категория, критерий, вид доказательства, название);
- каждое требование существует, у каждого есть доказательство, каждое доказательство
  относится к указанному в замечании требованию.

Доказательство-цитата:
- отрезок внутри текста фрагмента, цитата — ровно fragment.text[start:end] (точный текст,
  а не нормализованный: кавычки, пробелы и регистр не «подгоняются»), не только пробелы;
- цитата не начинается и не обрывается посреди слова и не отрезает отрицание перед собой
  («не», «ни», «без», «not», «no»): иначе цитата меняет смысл;
- ссылки на источник совпадают с span_sources (блок, смещения, место), место определено
  (строка, страница, абзац или таблица исходного файла).

Доказательство отсутствия (evidence_kind="absence"):
- обязательна ограниченная область (AbsenceClaim): таблица, раздел, требование, реестр ID
  документа или нумерация разделов; утверждение перепроверяется по документу;
- пустая цитата допустима только для фрагмента, у которого в документе действительно нет
  текста (пустая строка таблицы), и только внутри области.

Статус: правило — «verified»; LLM — «unverified» даже при доказанной цитате: точная цитата
доказывает происхождение текста, но не смысл замечания. Подтверждает гипотезы LLM только
отдельная смысловая проверка (следующие этапы).

Что считается текстом документа: текст, извлечённый из файла (разметка Markdown, колонтитулы
PDF и т. п. в него не входят). Цитата сверяется с ним символ в символ, а место — с
исходным файлом (строки, страница, абзац, таблица).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Literal, Protocol

from validator.criteria import CRITERIA
from validator.errors import EXIT_INTERNAL_ERROR, InputError, ValidatorError
from validator.findings import (
    ABSENCE_CLAIMS,
    EVIDENCE_KINDS,
    ORIGINS,
    REVIEW_CATEGORIES,
    RULE_CATEGORIES,
    SEVERITIES,
    AbsenceClaim,
    Evidence,
    EvidenceKind,
    EvidenceStatus,
    Finding,
    Origin,
    RejectedFinding,
    RuleCategory,
    Severity,
    is_probability,
)
from validator.ingest.common import one_line
from validator.model import Block, Document, Location
from validator.segmentation import Requirement, Segmentation, SourceRef, span_sources
from validator.segmentation.text import ID_PATTERN, is_document_id, parse_prefix
from validator.segmentation.verify import body_start, verify_segmentation

_REQUIRED_TEXT = ("id", "rule_id", "rule_version", "title", "explanation", "recommendation")
_TEXT_FIELDS = (*_REQUIRED_TEXT, "criterion", "document_sha256", "severity", "category")
_CATALOG_FIELDS = ("version", "severity", "category", "criterion", "evidence_kind", "title")
_NEGATION = re.compile(r"(?<![\w-])(?:не|ни|без|not|no|never)\s+$", re.IGNORECASE)
_ID_TOKEN = re.compile(rf"(?<![\w-]){ID_PATTERN}(?![\w-])")
_EMPTY_CELLS = frozenset({"", "—", "–", "-"})
_EMPTY_VALUE_QUOTE = re.compile(r"(?::[ \t]*|\t+)[—–-][ \t]*$")
_HEADING_NUMBER = re.compile(r"\s*(\d+(?:\.\d+)*)\.?\s")
_SECTION_SEPARATOR = " › "

EvidenceBasis = Literal["text", "structure", "absence"]


class EvidenceGateError(ValidatorError):
    """Проверка доказательств невозможна: сегментация не согласована с документом."""

    exit_code = EXIT_INTERNAL_ERROR


class RuleSpec(Protocol):
    """Запись каталога правил, по которой проверяется замечание origin="rule"."""

    @property
    def rule_id(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def severity(self) -> Severity: ...
    @property
    def category(self) -> RuleCategory: ...
    @property
    def criterion(self) -> str: ...
    @property
    def evidence_kind(self) -> EvidenceKind: ...
    @property
    def title(self) -> str: ...


@dataclass(frozen=True)
class TrustedRule:
    """Запись каталога без функции проверки (для Gate достаточно метаданных)."""

    rule_id: str
    version: str
    severity: Severity
    category: RuleCategory
    criterion: str
    evidence_kind: EvidenceKind
    title: str


@dataclass(frozen=True)
class CheckedEvidence:
    """Проверенное доказательство: что именно и где подтверждено."""

    requirement_id: str
    quote: str
    basis: EvidenceBasis  # текст, структурный элемент (заголовок, строка таблицы), отсутствие
    places: tuple[str, ...]  # места в исходном файле


@dataclass(frozen=True)
class GateVerdict:
    finding_id: str
    origin: Origin
    status: EvidenceStatus  # verified | unverified (LLM) | rejected
    reasons: tuple[str, ...]  # причины отказа
    requirement_ids: tuple[str, ...]
    checked: tuple[CheckedEvidence, ...]


@dataclass(frozen=True)
class GateResult:
    accepted: tuple[Finding, ...]  # «verified» — только замечания правил
    rejected: tuple[RejectedFinding, ...]  # «rejected» и причины
    unverified: tuple[Finding, ...] = ()  # LLM: доказательства верны, смысл не проверен
    verdicts: tuple[GateVerdict, ...] = ()  # по одному на каждое замечание, в порядке входа
    document_name: str = ""
    document_sha256: str = ""


# --- построение доказательств ------------------------------------------------------------


def build_evidence(fragment: Requirement, start: int, end: int, document: Document) -> Evidence:
    """Доказательство-цитата fragment.text[start:end]. ValueError, если отрезок нельзя
    привязать к документу (см. span_sources)."""
    sources = span_sources(fragment, start, end, document)
    return Evidence(fragment.id, start, end, fragment.text[start:end], sources)


def absence_evidence(fragment: Requirement) -> Evidence:
    """Доказательство отсутствия содержания: пустой фрагмент и его места в документе."""
    if fragment.text:
        raise ValueError(
            f"Доказательство отсутствия допустимо только для пустого фрагмента; "
            f"у {fragment.id} есть текст."
        )
    return Evidence(fragment.id, 0, 0, "", fragment.sources)


def ensure_source_unchanged(document: Document) -> None:
    """Файл не изменился после чтения: иначе доказательства построены для прежней версии."""
    try:
        data = document.source.read_bytes()
    except OSError as error:
        raise InputError(
            f"Документ {document.source.name} недоступен для повторной проверки: "
            f"{error.strerror or error}."
        ) from None
    if hashlib.sha256(data).hexdigest() != document.sha256:
        raise InputError(
            f"Документ {document.source.name} изменён во время проверки: доказательства "
            "построены для прежней версии. Запустите проверку заново."
        )


# --- проверка --------------------------------------------------------------------------------


def check_finding(
    finding: Finding,
    segmentation: Segmentation,
    document: Document,
    *,
    rules: Iterable[RuleSpec] | None = None,
) -> tuple[str, ...]:
    """Причины, по которым доказательства замечания не принимаются; пустой кортеж — доказаны.

    rules — каталог запущенных правил; без него сверка с каталогом не выполняется (проверка
    только доказательств). Статус присваивает gate_findings, где каталог обязателен.
    """
    catalog = None if rules is None else {rule.rule_id: rule for rule in rules}
    try:
        return tuple(_problems(finding, segmentation, document, catalog))
    except Exception as error:  # fail closed: сбой проверки — отказ, а не подтверждение
        return (f"Проверка не выполнена (внутренняя ошибка): {type(error).__name__}: {error}",)


def gate_findings(
    findings: Iterable[Finding],
    segmentation: Segmentation,
    document: Document,
    *,
    rules: Iterable[RuleSpec],
) -> GateResult:
    """Проверить замечания и присвоить статус. Порядок сохраняется."""
    integrity = verify_segmentation(segmentation, document)
    if integrity:
        raise EvidenceGateError(
            "Сегментация не согласована с документом — доказательства проверить нельзя: "
            + "; ".join(integrity[:5])
        )
    catalog = tuple(rules)
    accepted: list[Finding] = []
    unverified: list[Finding] = []
    rejected: list[RejectedFinding] = []
    verdicts: list[GateVerdict] = []
    for finding in findings:
        problems = check_finding(finding, segmentation, document, rules=catalog)
        status: EvidenceStatus
        if problems:
            status = "rejected"
            rejected.append(RejectedFinding(replace(finding, evidence_status=status), problems))
        elif finding.origin == "rule":
            status = "verified"
            accepted.append(replace(finding, evidence_status=status))
        else:  # LLM: доказана цитата, но не смысл замечания
            status = "unverified"
            unverified.append(replace(finding, evidence_status=status))
        verdicts.append(_verdict(finding, status, problems, segmentation, document))
    return GateResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        unverified=tuple(unverified),
        verdicts=tuple(verdicts),
        document_name=document.source.name,
        document_sha256=document.sha256,
    )


def _verdict(
    finding: Finding,
    status: EvidenceStatus,
    problems: tuple[str, ...],
    segmentation: Segmentation,
    document: Document,
) -> GateVerdict:
    checked: tuple[CheckedEvidence, ...] = ()
    if not problems:
        checked = tuple(_checked(evidence, document) for evidence in finding.evidence)
    origin: Origin = finding.origin if finding.origin in ORIGINS else "llm"
    ids = finding.requirement_ids if isinstance(finding.requirement_ids, tuple) else ()
    return GateVerdict(finding.id, origin, status, problems, ids, checked)


def _checked(evidence: Evidence, document: Document) -> CheckedEvidence:
    places = tuple(dict.fromkeys(ref.location.describe() for ref in evidence.sources))
    basis: EvidenceBasis
    if not evidence.quote:
        basis = "absence"
    elif document.blocks[evidence.sources[0].block].kind in (
        "heading",
        "table_header",
        "table_row",
    ):
        basis = "structure"
    else:
        basis = "text"
    return CheckedEvidence(evidence.requirement_id, evidence.quote, basis, places)


def _problems(
    finding: Finding,
    segmentation: Segmentation,
    document: Document,
    catalog: Mapping[str, RuleSpec] | None,
) -> list[str]:
    type_problems = _type_problems(finding)
    if type_problems:
        return type_problems
    problems = [
        f"Не заполнено обязательное поле {name}."
        for name in _REQUIRED_TEXT
        if not getattr(finding, name).strip()
    ]
    if finding.severity not in SEVERITIES:
        problems.append(f"Недопустимая важность (severity): {finding.severity!r}.")
    if finding.category not in RULE_CATEGORIES:
        problems.append(f"Недопустимая категория (category): {finding.category!r}.")
    if finding.criterion not in CRITERIA:
        problems.append(f"Неизвестный критерий качества (criterion): {finding.criterion!r}.")
    if finding.evidence_kind not in EVIDENCE_KINDS:
        problems.append(f"Недопустимый вид доказательства: {finding.evidence_kind!r}.")
    if finding.origin not in ORIGINS:
        problems.append(f"Недопустимое происхождение (origin): {finding.origin!r}.")
    if finding.origin == "rule" and (
        finding.review_category is not None or finding.confidence is not None
    ):
        problems.append(
            "У замечания правила не может быть полей LLM-обзора (review_category, confidence)."
        )
    binding = _binding_problems(finding, segmentation, document)
    if binding:  # смещения относятся к другому тексту — дальше проверять бессмысленно
        return problems + binding
    if catalog is not None and finding.origin == "rule":
        problems += _catalog_problems(finding, catalog)
    if not finding.requirement_ids:
        problems.append("Не указаны требования (requirement_ids).")
    repeated = [rid for rid, count in Counter(finding.requirement_ids).items() if count > 1]
    if repeated:
        problems.append(f"ID требований повторяются: {', '.join(repeated)}.")
    if not finding.evidence:
        problems.append("Нет доказательств (evidence).")
    for evidence in finding.evidence:
        problems += _check_evidence(evidence, finding, segmentation, document)
    covered = {evidence.requirement_id for evidence in finding.evidence}
    problems += [
        f"У требования {rid} нет доказательства."
        for rid in dict.fromkeys(finding.requirement_ids)
        if rid not in covered
    ]
    if not problems:
        problems += _absence_problems(finding, segmentation, document)
    return problems


def _type_problems(finding: Finding) -> list[str]:
    """Поля нужных типов: данные из JSON или от модели могут быть любыми."""
    problems = [
        f"Поле {name}: ожидается строка."
        for name in _TEXT_FIELDS
        if not isinstance(getattr(finding, name), str)
    ]
    if not isinstance(finding.requirement_ids, tuple) or not all(
        isinstance(rid, str) for rid in finding.requirement_ids
    ):
        problems.append("Поле requirement_ids: ожидается кортеж строк.")
    if finding.review_category is not None and finding.review_category not in REVIEW_CATEGORIES:
        problems.append(f"Поле review_category: недопустимое значение {finding.review_category!r}.")
    if finding.confidence is not None and not is_probability(finding.confidence):
        problems.append("Поле confidence: ожидается число от 0 до 1 — оценка модели.")
    if finding.absence is not None and not (
        isinstance(finding.absence, AbsenceClaim)
        and all(isinstance(getattr(finding.absence, f.name), str) for f in fields(AbsenceClaim))
    ):
        problems.append("Поле absence: ожидается AbsenceClaim из строк.")
    if not isinstance(finding.evidence, tuple):
        return [*problems, "Поле evidence: ожидается кортеж доказательств."]
    for index, evidence in enumerate(finding.evidence, 1):
        if not isinstance(evidence, Evidence):
            problems.append(f"Доказательство № {index}: ожидается Evidence.")
            continue
        if not isinstance(evidence.requirement_id, str) or not isinstance(evidence.quote, str):
            problems.append(f"Доказательство № {index}: requirement_id и quote — строки.")
        if not _is_int(evidence.start) or not _is_int(evidence.end):
            problems.append(f"Доказательство № {index}: start и end — целые числа.")
        if not isinstance(evidence.sources, tuple) or not all(
            isinstance(ref, SourceRef)
            and _is_int(ref.block)
            and _is_int(ref.start)
            and _is_int(ref.end)
            and isinstance(ref.location, Location)
            for ref in evidence.sources
        ):
            problems.append(f"Доказательство № {index}: sources — кортеж SourceRef.")
    return problems


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _binding_problems(
    finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    if segmentation.document_sha256 != document.sha256:
        return ["Сегментация относится к другому документу (SHA-256 не совпадает)."]
    if not finding.document_sha256:
        return ["Замечание не привязано к документу: не указан SHA-256 документа."]
    if finding.document_sha256 != document.sha256:
        return [
            "Замечание построено для другой версии документа или другого файла "
            f"(SHA-256 {finding.document_sha256[:12]}…, а проверяется {document.sha256[:12]}…): "
            "его нужно построить заново."
        ]
    return []


def _catalog_problems(finding: Finding, catalog: Mapping[str, RuleSpec]) -> list[str]:
    rule = catalog.get(finding.rule_id)
    if rule is None:
        return [
            f"Правило {finding.rule_id} не входит в каталог запущенных правил: замечание "
            "с origin=rule без записи каталога не подтверждается."
        ]
    actual = {
        "version": finding.rule_version,
        "severity": finding.severity,
        "category": finding.category,
        "criterion": finding.criterion,
        "evidence_kind": finding.evidence_kind,
        "title": finding.title,
    }
    differences = [
        f"{name} ({actual[name]!r} вместо {getattr(rule, name)!r})"
        for name in _CATALOG_FIELDS
        if actual[name] != getattr(rule, name)
    ]
    if differences:
        return [
            f"Поля замечания не совпадают с каталогом правила {rule.rule_id}: "
            + ", ".join(differences)
            + "."
        ]
    return []


def _check_evidence(
    evidence: Evidence, finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    label = f"Доказательство для {evidence.requirement_id}"
    problems: list[str] = []
    if evidence.requirement_id not in finding.requirement_ids:
        problems.append(f"{label}: требование не указано в requirement_ids.")
    try:
        fragment = segmentation.by_id(evidence.requirement_id)
    except KeyError:
        return [*problems, f"{label}: требование {evidence.requirement_id} не найдено."]
    if evidence.start == evidence.end == 0 and not evidence.quote:
        return problems + _check_empty(evidence, finding, fragment, document, label)
    start, end, text = evidence.start, evidence.end, fragment.text
    if not 0 <= start < end <= len(text):
        return [
            *problems,
            f"{label}: отрезок {start}–{end} вне текста требования (длина {len(text)}).",
        ]
    actual = text[start:end]
    if actual != evidence.quote:
        problems.append(
            f"{label}: цитата не совпадает с текстом требования ({start}–{end}) — "
            "требуется точный текст, без пересказа и нормализации."
        )
    if not actual.strip():
        problems.append(f"{label}: цитата пуста — только пробельные символы.")
    if start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        problems.append(f"{label}: цитата начинается посреди слова — смысл искажается.")
    if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        problems.append(f"{label}: цитата обрывается посреди слова — смысл искажается.")
    negation = _NEGATION.search(text, max(0, start - 12), start)
    if negation:
        problems.append(
            f"{label}: цитата отрезает отрицание «{negation.group(0).strip()}» перед собой — "
            "смысл меняется на противоположный."
        )
    try:
        expected = span_sources(fragment, start, end, document)
    except ValueError as error:
        return [*problems, f"{label}: {error}"]
    if expected != evidence.sources:
        problems.append(f"{label}: ссылки на источник не совпадают с документом.")
    else:
        problems += _location_problems(evidence.sources, label)
    return problems


def _check_empty(
    evidence: Evidence, finding: Finding, fragment: Requirement, document: Document, label: str
) -> list[str]:
    if finding.evidence_kind != "absence":
        return [f"{label}: пустая цитата допустима только в замечании об отсутствии содержания."]
    problems: list[str] = []
    if fragment.text:
        problems.append(f"{label}: доказательство отсутствия, но фрагмент не пуст.")
    blocks = document.blocks
    if not fragment.sources or not all(
        0 <= ref.block < len(blocks) and blocks[ref.block].text == "" for ref in fragment.sources
    ):
        problems.append(f"{label}: блоки документа этого фрагмента не пусты или не найдены.")
    if evidence.sources != fragment.sources:
        problems.append(f"{label}: ссылки на источник не совпадают с документом.")
    else:
        problems += _location_problems(evidence.sources, label)
    return problems


def _location_problems(sources: tuple[SourceRef, ...], label: str) -> list[str]:
    for ref in sources:
        place = ref.location
        if (place.page, place.line_start, place.paragraph, place.table) == (None,) * 4:
            return [
                f"{label}: место доказательства в исходном файле не определено "
                "(нет строки, страницы, абзаца или таблицы)."
            ]
    return []


# --- отсутствие в ограниченной области ---------------------------------------------------


def _absence_problems(
    finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    claim = finding.absence
    if finding.evidence_kind != "absence":
        if claim is not None:
            return ["Указана область отсутствия, но вид доказательства не absence."]
        return []
    if claim is None:
        return [
            "Доказательство отсутствия без ограниченной и проверяемой области (absence): "
            "утверждение «этого нет» не принимается."
        ]
    if (claim.scope, claim.missing) not in ABSENCE_CLAIMS:
        return [f"Недопустимая область отсутствия: {claim.scope}/{claim.missing}."]
    checks = {
        "table": _table_absence,
        "section": _section_absence,
        "requirement": _requirement_absence,
        "id_registry": _id_absence,
        "heading_registry": _heading_absence,
    }
    return checks[claim.scope](claim, finding, segmentation, document)


def _not_confirmed(detail: str) -> list[str]:
    return [f"Утверждение об отсутствии не подтверждается документом: {detail}."]


def _evidence_blocks(finding: Finding) -> list[int]:
    return [ref.block for evidence in finding.evidence for ref in evidence.sources]


def _table_absence(
    claim: AbsenceClaim, finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    if not claim.target.isdigit():
        return [f"Область «таблица»: ожидается номер таблицы, а не «{claim.target}»."]
    number = int(claim.target)
    members = [
        index
        for index, block in enumerate(document.blocks)
        if block.location.table == number and block.kind in ("table_header", "table_row")
    ]
    if not members:
        return [f"Область отсутствия: таблица {number} в документе не найдена."]
    if any(index not in members for index in _evidence_blocks(finding)):
        return [f"Доказательства лежат вне таблицы {number}, указанной как область отсутствия."]
    rows = [document.blocks[i] for i in members if document.blocks[i].kind == "table_row"]
    if any(_row_filled(row) for row in rows):
        return _not_confirmed(f"в таблице {number} есть заполненные строки")
    return []


def _row_filled(block: Block) -> bool:
    cells = block.cells or (block.text,)
    return any(cell.strip() not in _EMPTY_CELLS for cell in cells)


def _section_absence(
    claim: AbsenceClaim, finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    path = tuple(claim.target.split(_SECTION_SEPARATOR))
    headings = [
        index
        for index, block in enumerate(document.blocks)
        if block.kind == "heading" and (*block.location.section_path, one_line(block.text)) == path
    ]
    if not headings:
        return [f"Область отсутствия: раздел «{claim.target}» в документе не найден."]
    if not set(_evidence_blocks(finding)) & set(headings):
        return [f"Доказательство не указывает на заголовок раздела «{claim.target}»."]
    for block in document.blocks:
        if block.kind != "heading" and block.location.section_path[: len(path)] == path:
            return _not_confirmed(f"в разделе «{claim.target}» есть текст")
    return []


def _requirement_absence(
    claim: AbsenceClaim, finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    if claim.target not in finding.requirement_ids:
        return [f"Требование области отсутствия {claim.target} не входит в замечание."]
    fragment = segmentation.by_id(claim.target)
    if claim.missing == "document_id":
        if fragment.doc_id is not None:
            return _not_confirmed(f"у требования есть идентификатор «{fragment.doc_id}»")
        return []
    if claim.missing == "number":
        body = _ID_TOKEN.sub(" ", fragment.text[body_start(fragment, document) :])
        if re.search(r"\d", body):
            return _not_confirmed("в требовании есть число")
        return []
    block = document.blocks[fragment.sources[0].block]
    two_cells = block.kind == "table_row" and len(block.cells) == 2
    if (
        two_cells
        and block.cells[0].strip() not in _EMPTY_CELLS
        and block.cells[1].strip() in _EMPTY_CELLS
    ):
        return []  # «параметр | —»: значение пусто или прочерк
    quotes = [e.quote for e in finding.evidence if e.requirement_id == claim.target]
    if any(_EMPTY_VALUE_QUOTE.search(quote) for quote in quotes):
        return []
    return _not_confirmed("значение в требовании заполнено")


def _id_absence(
    claim: AbsenceClaim, finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    if not is_document_id(claim.target):
        return [f"Область «реестр ID»: «{claim.target}» — не идентификатор требования."]
    if not any(claim.target in evidence.quote for evidence in finding.evidence):
        return [f"Доказательство не упоминает идентификатор «{claim.target}»."]
    declared = {f.doc_id for f in segmentation.fragments if f.doc_id}
    declared |= {
        doc_id
        for block in document.blocks
        if block.kind == "heading" and (doc_id := parse_prefix(block.text, "heading").doc_id)
    }
    if claim.target in declared:
        return _not_confirmed(f"идентификатор «{claim.target}» в документе объявлен")
    return []


def _heading_absence(
    claim: AbsenceClaim, finding: Finding, segmentation: Segmentation, document: Document
) -> list[str]:
    if not re.fullmatch(r"\d+(?:\.\d+)*", claim.target):
        return [f"Область «нумерация разделов»: «{claim.target}» — не номер раздела."]
    if not any(claim.target in evidence.quote for evidence in finding.evidence):
        return [f"Доказательство не упоминает номер раздела {claim.target}."]
    for block in document.blocks:
        number = _HEADING_NUMBER.match(one_line(block.text) + " ")
        if block.kind == "heading" and number and number.group(1) == claim.target:
            return _not_confirmed(f"раздел {claim.target} в документе есть")
    return []
