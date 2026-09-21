"""후보 날짜 중 최적의 소비기한을 선별하는 모듈."""

from __future__ import annotations

import calendar
import re

from .parse import Candidate

NEGATIVE = (
    "품목보고번호", "품목보고", "보고번호", "제조일자", "제조년월", "제조",
    "생산일자", "생산", "등록번호", "허가번호", "전화", "상담", "고객", "문의",
    "LOT", "L0T", "MFG", "MFD", "PROD", "BATCH", "SINCE", "TEL", "FAX",
    "PRO.DATE", "PRO DATE", "PROD DATE", "P.D.", "P.D", "PRD",
    "MANUFACTURE", "MANUFACTURING", "MFR", "DOM", "PKD",
)

POSITIVE = (
    "소비기한", "유통기한", "품질유지기한", "까지", "기한", "유통", "소비",
    "EXP", "EXPIRY", "EXPIRES", "BBE", "BBD", "BEST BEFORE", "BEST BY",
    "USE BY", "USEBY", "정확한", "표시일",
)

PATTERN_PRIOR = {
    "korean": 12, "ymd4": 10, "dmy4": 8, "mon_d_y": 8, "d_mon_y": 8, "y_mon_d": 8,
    "ymd4_space": 7, "dmy4_space": 6,
    "ymd4_cross": 6, "dmy4_cross": 5,
    "ymd4_colon": 9, "ymd4_colon2": 9,
    "ym_space_d": 7, "d_space_my": 6,
    "ymd9": 4, "y_mmdd_tail": 4,
    "ymd8": 4,
    "ymmd": 4,
    "ymmd2": -2,
    "y_mmdd": 4, "d_mmy": 4, "dm_y": 4,
    "y2_mmdd": 3,
    "dmy8": 1,
    "ymd6": 1, "dmy6": -2,
    "d_fuzz_y": 1, "fuzz_y": -3, "m_y": -1,
    "y1m1d": 3, "d1m1y": 2, "y1m1d2": 0,
    "ymd2": 3, "dmy2": -1,
    "mon_y": 2, "ym4": 1,
    "md": -2, "dm": -4,
}

ANCHOR_YEAR = 2026

_DAY_RULE_LAST = re.compile(r"말일|마지막\s*날|end\s+of\s+(the\s+)?month", re.I)
_DAY_RULE_FIRST = re.compile(r"0?1\s*일\s*까지|1st\s+of", re.I)


def _has(text: str, words) -> bool:
    upper = text.upper()
    if any(w.upper() in upper for w in words):
        return True
    cleaned = re.sub(r"[.:\-_/]", " ", upper)
    return any(w.upper() in cleaned for w in words)


POSITIVE_BONUS = 40
POSITIVE_FULL_TEXT_BONUS = 20


def score(cand: Candidate, full_text: str = "", suppress_positive: bool = False) -> float:
    """후보 날짜의 신뢰도 점수를 계산한다."""
    s = 0.0

    # 1. 연속 숫자열(품목보고번호 등) 감점
    if cand.embedded:
        ctx = cand.context
        st, en = cand.span
        while st > 0 and ctx[st - 1].isdigit():
            st -= 1
        while en < len(ctx) and ctx[en].isdigit():
            en += 1
        digit_len = en - st
        if digit_len >= 12:
            s -= 1000
        elif digit_len >= 11:
            s -= 500
        else:
            s -= 15

    # 2. 문맥 키워드 점수 반영
    if _has(cand.context, NEGATIVE):
        s -= 60
    elif _has(full_text, NEGATIVE):
        s -= 20

    if not suppress_positive:
        if _has(cand.context, POSITIVE):
            s += POSITIVE_BONUS
        elif _has(full_text, POSITIVE):
            s += POSITIVE_FULL_TEXT_BONUS

    # 3. 패턴 형식 사전 점수
    pattern = cand.pattern
    if pattern.endswith("_rev"):
        s += PATTERN_PRIOR.get(pattern[:-4], 0) - 8
    else:
        s += PATTERN_PRIOR.get(pattern, 0)

    # 4. 연도 범위 보정
    if cand.year and cand.complete:
        try:
            y_int = int(cand.year)
            if ("2" in cand.pattern or "6" in cand.pattern) and y_int >= 2028:
                s -= 15
            elif y_int < 2020:
                s -= 30
        except (ValueError, TypeError):
            pass

    # 5. 완전한 날짜(연·월·일 구비) 가산점
    if cand.complete:
        s += 35
    elif not cand.year:
        s -= 15

    return s


def _impute_day(cand: Candidate, full_text: str) -> str:
    """일자가 없을 때(`OCT. 2021`) 채워 넣는다. NONE이면 0점, 채우면 최소 10점."""
    if _DAY_RULE_LAST.search(full_text):
        return f"{calendar.monthrange(int(cand.year), int(cand.month))[1]:02d}"
    return "01"   # 기본값. '말일' 가정은 실물로 반증됐다 — §PIPELINE 보정표


def _impute_year(cand: Candidate) -> str:
    """연도가 없을 때(`02.18까지`) 추정한다.

    EXIF 촬영일시 폴백은 쓰지 않는다 — 확인된 결측 사례 2건(`000995`,`001955`)
    **모두 EXIF에 촬영일시가 없었다.** 발동하지 않는 분기다.
    """
    month = int(cand.month) if cand.month else 12
    return str(ANCHOR_YEAR + 1 if month <= 6 else ANCHOR_YEAR)


def impute(cand: Candidate, full_text: str = "") -> Candidate:
    """결측 필드를 보정한다."""
    year, month, day = cand.year, cand.month, cand.day
    if year and month and not day:
        day = _impute_day(cand, full_text)
    elif month and day and not year:
        year = _impute_year(cand)
    if (year, month, day) == (cand.year, cand.month, cand.day):
        return cand
    if year and month and day and int(day) > calendar.monthrange(int(year), int(month))[1]:
        return cand
    return Candidate(**{**cand.__dict__, "year": year, "month": month, "day": day})


def rank(candidates, full_text: str = "") -> list[tuple[float, Candidate]]:
    """후보 목록을 점수 및 우선순위에 따라 정렬한다."""
    has_complete = any(c.complete for c in candidates)
    has_4digit_complete = any(
        c.complete and c.year and len(c.year) == 4 and ("2" not in c.pattern and "6" not in c.pattern)
        for c in candidates
    )

    scored = []
    for c in candidates:
        suppress = has_complete and not c.complete
        base = score(c, full_text, suppress_positive=suppress)
        if has_4digit_complete and c.complete and ("2" in c.pattern or "6" in c.pattern):
            base -= 45
        scored.append((base, c))
    scored.sort(key=lambda p: (p[0], p[1].final_date or "", p[1].source, p[1].year or ""), reverse=True)
    return scored


MIN_SCORE = -500
STOP_SCORE = 19


def select(candidates, full_text: str = "", impute_missing: bool = False) -> Candidate | None:
    """후보 날짜 중 최적의 소비기한을 선별한다. 후보가 없으면 None."""
    ranked = rank(candidates, full_text)
    if not ranked or ranked[0][0] < MIN_SCORE:
        return None
    best = ranked[0][1]
    return impute(best, full_text) if impute_missing else best


def to_row(cand: Candidate | None, image_id: str) -> dict:
    """제출 스키마 행을 생성한다."""
    if cand is None:
        return {"image_id": image_id, "year": "NONE", "month": "NONE",
                "day": "NONE", "final_date": "NONE"}
    return {
        "image_id": image_id,
        "year": cand.year or "NONE",
        "month": cand.month or "NONE",
        "day": cand.day or "NONE",
        "final_date": cand.final_date or "NONE",
    }
