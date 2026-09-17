"""인식된 텍스트에서 날짜 후보를 추출하는 모듈."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

CONFUSION = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "、": ".", "·": ".", ",": ".", "•": ".",
})

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
MONTHS["SEPT"] = 9
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))

YEAR_MIN, YEAR_MAX = 2018, 2032


def normalize(text: str) -> str:
    """OCR 혼동 문자를 1:1로 정규화한다."""
    chars = list(text)
    last = len(text) - 1
    for i, ch in enumerate(text):
        mapped = CONFUSION.get(ord(ch))
        if mapped is None:
            continue
        prev_alpha = i > 0 and text[i - 1].isalpha()
        next_alpha = i < last and text[i + 1].isalpha()
        if prev_alpha or next_alpha:
            continue
        chars[i] = mapped
    return "".join(chars)


@dataclass(frozen=True)
class Candidate:
    """텍스트 한 조각에서 나온 날짜 후보 하나."""

    year: str | None
    month: str | None
    day: str | None
    text: str          # 매치된 원문 조각
    context: str       # 문맥 줄 전체
    pattern: str
    embedded: bool     # 연속 숫자열 포함 여부
    span: tuple[int, int]
    source: int = 0

    @property
    def final_date(self) -> str | None:
        if self.year and self.month and self.day:
            return f"{self.year}-{self.month}-{self.day}"
        return None

    @property
    def complete(self) -> bool:
        return self.final_date is not None


def fuzzy_month(name: str) -> int | None:
    """영문 월 명칭의 1글자 오독을 보정한다."""
    key = name.upper()[:3]
    if key in MONTHS:
        return MONTHS[key]
    best, best_cost = None, 2
    for cand, num in MONTHS.items():
        cand = cand[:3]
        cost = sum(a != b for a, b in zip(key, cand))
        if len(key) == len(cand) and cost < best_cost:
            best, best_cost = num, cost
    return best


def _year4(value: str) -> str | None:
    """2자리/4자리 연도를 4자리로 변환한다."""
    y = int(value)
    if len(value) == 2:
        y += 2000
    return str(y) if YEAR_MIN <= y <= YEAR_MAX else None


def _valid_md(month: int, day: int | None) -> bool:
    if not 1 <= month <= 12:
        return False
    if day is None:
        return True
    return 1 <= day <= 31


def _build(year, month, day, *, raw, span, context, pattern, source):
    """검증 후 Candidate 객체를 생성한다."""
    if month is not None and not _valid_md(month, day):
        return None
    if year is not None and day is not None and month is not None:
        if day > calendar.monthrange(int(year), month)[1]:
            return None
    return Candidate(
        year=year,
        month=f"{month:02d}" if month else None,
        day=f"{day:02d}" if day else None,
        text=raw[span[0]:span[1]],
        context=raw,
        pattern=pattern,
        embedded=_embedded(raw, *span),
        span=span,
        source=source,
    )


def _embedded(raw: str, start: int, end: int) -> bool:
    """매치가 구분자 없는 연속 숫자열의 일부인지 판별한다."""
    if any(sep in raw[start:end] for sep in ".-/ "):
        return False
    before = raw[start - 1] if start > 0 else ""
    after = raw[end] if end < len(raw) else ""
    return before.isdigit() or after.isdigit()


# ── 날짜 정규식 패턴 정의 ───────────────────────────────────────────
_SEP = r"[.\-/]"

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("korean", re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")),
    ("ymd4",   re.compile(rf"(20\d{{2}})\s*({_SEP})\s*(\d{{1,2}})\s*\2\s*(\d{{1,2}})")),
    ("dmy4",   re.compile(rf"(\d{{1,2}})\s*({_SEP})\s*(\d{{1,2}})\s*\2\s*(20\d{{2}})")),
    ("ymd4_space", re.compile(r"(20\d{2})\s+(\d{1,2})\s+(\d{1,2})")),
    ("dmy4_space", re.compile(r"(\d{1,2})\s+(\d{1,2})\s+(20\d{2})")),
    # 구분자가 섞이거나 공백이 구분자 역할을 하는 표기(`2022 03. 05`)도 받는다.
    # 숫자 경계가 없으면 박스 병합 공백 때문에 `2021.12` + `300` 이 `2021-12-30` 이 된다.
    ("ymd4_cross", re.compile(rf"(?<!\d)(20\d{{2}})(?:\s*{_SEP}\s*|\s+)(\d{{1,2}})(?:\s*{_SEP}\s*|\s+)(\d{{1,2}})(?!\d)")),
    ("dmy4_cross", re.compile(rf"(\d{{1,2}})\s*{_SEP}\s*(\d{{1,2}})\s*{_SEP}\s*(20\d{{2}})")),
    ("ymd4_colon", re.compile(r"(?<!\d)(20\d{2})\s*[:]\s*(\d{1,2})\s*[.:\-/]\s*(\d{1,2})(?!\d)")),
    ("ymd4_colon2", re.compile(r"(?<!\d)(20\d{2})\s*[.:\-/]\s*(\d{1,2})\s*[:]\s*(\d{1,2})(?!\d)")),
    ("ymd9",   re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d(?!\d)")),
    ("ymd8",   re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})")),
    ("y_mmdd", re.compile(r"(20\d{2})[.\-/ ](\d{2})(\d{2})(?!\d)")),
    ("d_mmy",  re.compile(r"(?<!\d)(\d{1,2})[.\-/ ](\d{2})(20\d{2})(?!\d)")),
    ("dmy8",   re.compile(r"(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)")),
    ("ymd2",   re.compile(rf"(\d{{2}})\s*({_SEP})\s*(\d{{1,2}})\s*\2\s*(\d{{1,2}})")),
    ("ymd2_space", re.compile(r"(\d{2})\s+(\d{1,2})\s+(\d{1,2})")),
    ("ymd2_cross", re.compile(rf"(\d{{2}})\s*{_SEP}\s*(\d{{1,2}})\s*{_SEP}\s*(\d{{1,2}})")),
    # 월 이름 꼬리는 [A-Za-z]* 로만 받는다. \w* 는 숫자까지 삼켜 `2021JUN12` 의 일을
    # `2` 로 자르고, `FEB042021` 의 일을 버린 채 연·월만 남겼다.
    ("y_mon_d", re.compile(rf"(20\d{{2}})\s*{_SEP}?\s*({_MON})[A-Za-z]*\s*{_SEP}?\s*(\d{{1,2}})", re.I)),
    ("d_mon_y", re.compile(rf"(\d{{1,2}})\s*{_SEP}?\s*({_MON})[A-Za-z]*\s*{_SEP}?\s*(\d{{2,4}})", re.I)),
    # 월-일-연 (`MAY/15/20`, `APR-28-2023`, `AUG 13 2021`). 일·연 사이 구분자가 있으면 연도 2·4자리
    ("mon_d_y", re.compile(rf"({_MON})[A-Za-z]*\.?\s*[.\-/,]?\s*(\d{{1,2}})(?:\s*[.\-/,]\s*|\s+)(\d{{4}}|\d{{2}})(?!\d)", re.I)),
    # 붙은 표기(`AUG122020`, `AUG 132021`)는 4자리 연도만 — `MAY2022` 를 5월 20일로 읽지 않게
    ("mon_d_y", re.compile(rf"({_MON})[A-Za-z]*\.?\s*[.\-/,]?\s*(\d{{1,2}})(20\d{{2}})(?!\d)", re.I)),
    ("mon_y",  re.compile(rf"({_MON})[A-Za-z]*\.?\s*(20\d{{2}})", re.I)),
    ("d_fuzz_y", re.compile(r"(\d{1,2})\s*[./\- ]\s*([A-Za-z]{3,4})\s*[./\- ]\s*(\d{2,4})")),
    ("fuzz_y",  re.compile(r"(?<![A-Za-z])([A-Za-z]{3,4})\.?\s*(20\d{2})")),
    ("ymmd",   re.compile(r"(20\d{2})(\d{2})[./\- ](\d{1,2})(?![\d])")),
    ("ymmd2",  re.compile(r"(?<!\d)(\d{2})(\d{2})\.(\d{1,2})(?![\d])")),
    ("y2_mmdd", re.compile(r"(?<!\d)(\d{2})\.(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")),
    ("ymd6",   re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")),
    ("ym4",    re.compile(rf"(20\d{{2}})\s*({_SEP})\s*(\d{{1,2}})(?!{_SEP}?\d)")),
    ("m_y",    re.compile(r"(?<!\d)(\d{1,2})\s*[./\-]\s*(20\d{2})(?!\d)")),
    ("ym_space_d", re.compile(r"(?<!\d)(20\d{2})[.:\-/]\s*([01]?\d)\s+([0-3]?\d)(?![:\d])")),
    ("d_space_my", re.compile(r"(?<![:\d])([0-3]?\d)\s+([01]?\d)[.:\-/](20\d{2})(?!\d)")),
    ("y1m1d",  re.compile(r"(?<!\d)(20\d{2})1([01]\d)1([0-3]\d)(?!\d)")),
    ("d1m1y",  re.compile(r"(?<!\d)([0-3]\d)1([01]\d)1(20\d{2})(?!\d)")),
    ("y1m1d2", re.compile(r"(?<!\d)(\d{2})1([01]\d)1([0-3]\d)(?!\d)")),
    ("md",     re.compile(rf"(?<!\d)(?<![\d][.\-/])(\d{{1,2}})\s*({_SEP})\s*(\d{{1,2}})(?!{_SEP}?\d)")),
]


def _interpret(name, m, raw, source):
    """패턴 매치 → 가능한 해석들. 모호하면 여러 개를 내고 달력 검증으로 거른다.

    ⚠️ **텍스트에 연도가 찍혀 있는데 창([2018,2032]) 밖이면 후보 자체를 버린다.**
    연도만 None으로 두고 남기면 보정 단계가 그럴듯한 연도를 붙여 되살리기 때문에,
    ``1986.08.02`` 이나 품목보고번호가 유효한 소비기한으로 둔갑한다.
    연도 None이 허용되는 건 텍스트에 애초에 연도가 없는 ``md`` 뿐이다.
    """
    g, span = m.groups(), m.span()
    kw = dict(raw=raw, span=span, context=raw, pattern=name, source=source)

    def dated(year, month, day, pattern=None):
        """연도가 창 밖이면(=None) 후보를 만들지 않는다."""
        if not year:
            return []
        return [_build(year, month, day, **{**kw, "pattern": pattern or name})]

    def or_mdy(found, year, month, day, pattern):
        """일-월-연·연-월-일이 모두 무효일 때만 미국식 월-일-연(`08/18/21`)으로 읽는다.

        유효한 해석이 하나라도 있으면 추가하지 않는다 — `01/12/21` 같은
        기존 일-월-연 정답에 경쟁 후보를 만들지 않기 위해서다.
        """
        if any(c is not None for c in found):
            return found
        # 미국식은 `/` 로 찍힌다. `01.30.18` 같은 점 표기는 국내 월.일 + 시각이라 제외한다
        # (통합셋: `/` 6장 전부 정답, `.` 2장 전부 오탐)
        if "/" not in raw[span[0]:span[1]]:
            return found
        if raw[span[1]:].lstrip().startswith(":"):   # `01/30/18:04` 의 18 은 시각이다
            return found
        return dated(year, month, day, pattern)

    if name == "ymd4":
        return dated(_year4(g[0]), int(g[2]), int(g[3]))
    if name == "dmy4":
        return or_mdy(dated(_year4(g[3]), int(g[2]), int(g[0])),
                      _year4(g[3]), int(g[0]), int(g[2]), "mdy4")
    if name == "ymd4_space":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "dmy4_space":
        return dated(_year4(g[2]), int(g[1]), int(g[0]))
    if name == "ymd4_cross":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "dmy4_cross":
        return or_mdy(dated(_year4(g[2]), int(g[1]), int(g[0])),
                      _year4(g[2]), int(g[0]), int(g[1]), "mdy4")
    if name in ("ymd4_colon", "ymd4_colon2", "ymd9"):
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "ymd8":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "y_mmdd":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "d_mmy":
        return dated(_year4(g[2]), int(g[1]), int(g[0]))
    if name == "dmy8":
        return dated(_year4(g[2]), int(g[1]), int(g[0]))
    if name == "y2_mmdd":
        return dated(_year4(g[0]), int(g[1]), int(g[2]), "ymd2")
    if name == "korean":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "ymd2":
        return or_mdy(dated(_year4(g[0]), int(g[2]), int(g[3]), "ymd2") +
                      dated(_year4(g[3]), int(g[2]), int(g[0]), "dmy2"),
                      _year4(g[3]), int(g[0]), int(g[2]), "mdy2")
    if name == "ymd2_space":
        return dated(_year4(g[0]), int(g[1]), int(g[2]), "ymd2") + \
               dated(_year4(g[2]), int(g[1]), int(g[0]), "dmy2")
    if name == "ymd2_cross":
        return or_mdy(dated(_year4(g[0]), int(g[1]), int(g[2]), "ymd2") +
                      dated(_year4(g[2]), int(g[1]), int(g[0]), "dmy2"),
                      _year4(g[2]), int(g[0]), int(g[1]), "mdy2")
    if name == "y_mon_d":
        return dated(_year4(g[0]), MONTHS[g[1].upper()[:3]], int(g[2]))
    if name == "d_mon_y":
        return dated(_year4(g[2]), MONTHS[g[1].upper()[:3]], int(g[0]))
    if name == "mon_d_y":
        return dated(_year4(g[2]), MONTHS[g[0].upper()[:3]], int(g[1]))
    if name == "mon_y":
        return dated(_year4(g[1]), MONTHS[g[0].upper()[:3]], None)
    if name == "d_fuzz_y":
        month = fuzzy_month(g[1])
        return dated(_year4(g[2]), month, int(g[0])) if month else []
    if name == "fuzz_y":
        month = fuzzy_month(g[0])
        return dated(_year4(g[1]), month, None) if month else []
    if name == "ymmd":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "ymmd2":
        return dated(_year4(g[0]), int(g[1]), int(g[2]), "ymmd2")
    if name == "ymd6":
        return dated(_year4(g[0]), int(g[1]), int(g[2]), "ymd6") + \
               dated(_year4(g[2]), int(g[1]), int(g[0]), "dmy6")
    if name == "m_y":
        return dated(_year4(g[1]), int(g[0]), None)
    if name == "ym_space_d":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "d_space_my":
        return dated(_year4(g[2]), int(g[1]), int(g[0]))
    if name == "y1m1d":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "d1m1y":
        return dated(_year4(g[2]), int(g[1]), int(g[0]))
    if name == "y1m1d2":
        return dated(_year4(g[0]), int(g[1]), int(g[2]), "y1m1d2")
    if name == "ym4":
        return dated(_year4(g[0]), int(g[2]), None)
    if name == "md":
        return [_build(None, int(g[0]), int(g[2]), **kw),
                _build(None, int(g[2]), int(g[0]), **{**kw, "pattern": "dm"})]
    return []


def parse(raw: str, source: int = 0) -> list[Candidate]:
    """한 줄의 텍스트에서 정규식 패턴을 적용해 날짜 후보를 추출한다."""
    if not raw:
        return []
    norm = normalize(raw)
    out, claimed = [], []

    for name, pattern in _PATTERNS:
        for m in pattern.finditer(norm):
            start, end = m.span()
            if any(s <= start and end <= e for s, e in claimed):
                continue
            found = [c for c in _interpret(name, m, raw, source) if c is not None]
            if found:
                claimed.append((start, end))
                out.extend(found)
    return out


def _filter_overlapping_boxes(row, overlap_thr: float = 0.3):
    """수평으로 겹치는 중복 박스를 정리한다."""
    if len(row) <= 1:
        return row

    sorted_row = sorted(row, key=lambda p: (p[1][1], -(p[1][3] - p[1][1])))
    kept = []
    for item in sorted_row:
        idx, box = item
        bx0, bx1 = box[1], box[3]
        bw = max(1.0, bx1 - bx0)
        duplicate = False
        for k_idx, (k_i, k_box) in enumerate(kept):
            kx0, kx1 = k_box[1], k_box[3]
            kw = max(1.0, kx1 - kx0)
            inter = max(0.0, min(bx1, kx1) - max(bx0, kx0))
            if inter / min(bw, kw) >= overlap_thr:
                duplicate = True
                if len(box[0]) > len(k_box[0]) or (len(box[0]) == len(k_box[0]) and bw > kw):
                    kept[k_idx] = item
                break
        if not duplicate:
            kept.append(item)
    return sorted(kept, key=lambda p: p[1][1])


def _is_just_time(text: str) -> bool:
    norm = normalize(text).strip()
    return bool(re.search(r"^\s*\d{1,2}:\d{2}", norm))


_TWO_DIGIT_PATS = {"ymd2", "ymd2_space", "ymd2_cross", "dmy2", "ymd6", "ymmd2", "y1m1d2"}


def merge_lines(items, y_tol: float = 0.6) -> list[tuple[str, int]]:
    """박스 기하 정보를 기반으로 같은 행의 텍스트 조각들을 병합한다."""
    if not items:
        return []
    boxes = [(t, float(x0), float(y0), float(x1), float(y1))
             for t, x0, y0, x1, y1 in items if t.strip()]
    if not boxes:
        return []

    heights = [b[4] - b[2] for b in boxes]
    med_h = float(sorted(heights)[len(heights) // 2])
    band = max(med_h * y_tol, 10.0)

    rows: list[list[tuple[int, tuple]]] = []
    for idx, b in enumerate(boxes):
        cy = (b[2] + b[4]) / 2.0
        placed = False
        for row in rows:
            rcy = sum((b_[2] + b_[4]) / 2 for _, b_ in row) / len(row)
            if abs(cy - rcy) <= band:
                row.append((idx, b))
                placed = True
                break
        if not placed:
            rows.append([(idx, b)])

    lines = []
    for row in rows:
        row = _filter_overlapping_boxes(row, overlap_thr=0.3)
        if len(row) > 1:
            row.sort(key=lambda p: p[1][1])
            texts = [b[0] for _, b in row]
            head = row[0][0]
            lines.append((" ".join(texts), head))
            lines.append(("".join(texts), head))

    # 수직 2줄 결합
    if len(rows) > 1:
        for r_top in rows:
            top_text = " ".join(b[0] for _, b in r_top)
            if any(c.complete for c in parse(top_text)) or _is_just_time(top_text):
                continue
            top_cy = sum((b[2] + b[4]) / 2 for _, b in r_top) / len(r_top)
            top_x0 = min(b[1] for _, b in r_top)
            top_x1 = max(b[3] for _, b in r_top)
            for r_bot in rows:
                if r_top is r_bot:
                    continue
                bot_text = " ".join(b[0] for _, b in r_bot)
                if any(c.complete for c in parse(bot_text)) or _is_just_time(bot_text):
                    continue
                bot_cy = sum((b[2] + b[4]) / 2 for _, b in r_bot) / len(r_bot)
                if 0 < bot_cy - top_cy <= band * 4.0:
                    bot_x0 = min(b[1] for _, b in r_bot)
                    bot_x1 = max(b[3] for _, b in r_bot)
                    x_overlap = max(0, min(top_x1, bot_x1) - max(top_x0, bot_x0))
                    if x_overlap > 0:
                        lines.append((f"{top_text} {bot_text}", r_top[0][0]))
    return lines


def parse_boxes(items) -> list[Candidate]:
    """검출 박스 목록 및 병합 줄에서 모든 날짜 후보를 추출한다."""
    out = []
    for i, box in enumerate(items):
        out.extend(parse(box[0], source=i))
    for line, head in merge_lines(items):
        out.extend(parse(line, source=head))

    # 180도 반전 문자열 fallback 파싱
    if not out:
        for i, box in enumerate(items):
            for cand in parse(box[0][::-1], source=i):
                out.append(Candidate(**{**cand.__dict__,
                                        "pattern": cand.pattern + "_rev",
                                        "context": box[0]}))

    # 동일 (연,월,일) 중복 정리
    best: dict[tuple, Candidate] = {}
    for c in out:
        key = (c.year, c.month, c.day)
        if key not in best:
            best[key] = c
        else:
            curr = best[key]
            c_is_2 = c.pattern in _TWO_DIGIT_PATS or c.pattern.endswith("2") or c.pattern.endswith("6")
            curr_is_2 = curr.pattern in _TWO_DIGIT_PATS or curr.pattern.endswith("2") or curr.pattern.endswith("6")
            if curr_is_2 and not c_is_2:
                best[key] = c
            elif not curr_is_2 and c_is_2:
                pass
            elif len(c.context) > len(curr.context):
                best[key] = c
    return list(best.values())
