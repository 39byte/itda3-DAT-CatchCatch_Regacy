"""웹 데모(ITDA3-CatchCatch-Web)용 합성 사용 내역과, 같은 입력에 대한 Python recommend() 결과를 만든다.

  python -m pharma.export_web <웹 저장소 경로>

  src/mock/usage.json            약품 정보 + 1년치 일별 불출량 (웹이 이걸로 모델을 돌린다)
  src/lib/recommend.fixture.json 웹 mock 선반 로트에 대한 Python 결과 (TS 포팅 대조 테스트의 기준)

약품 이름은 웹 mock 선반(A-03)의 21종이고, 수요 특성은 전부 시연용 가정이다.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pharma.model import recommend

AS_OF = pd.Timestamp("2026-09-25")      # 웹 mock 의 기준일 (InventoryContext CURRENT_DATE)
DAYS = 364

# 이름, 포장당 단위, 하루 발생 확률, 발생 시 평균 수량, 크기 산포(작을수록 몰림), 단가(원/단위), 필수 비축
DRUGS = [
    ("아세트아미노펜정 500mg", 100, 0.98, 60, 5, 30, False),
    ("이부프로펜시럽 100ml", 1, 0.60, 1.5, 5, 2500, False),
    ("세파클러캡슐 250mg", 100, 0.80, 8, 2, 250, False),
    ("아목시실린 캡슐 500mg", 100, 0.95, 30, 5, 150, False),
    ("푸로세미드정 40mg", 100, 0.85, 6, 5, 40, False),
    ("알마겔정 500mg", 100, 0.95, 25, 5, 50, False),
    ("트라마돌주사액 50mg", 10, 0.08, 2, 0.5, 900, False),
    ("페니실린주사 100만단위", 10, 0.03, 1.5, 0.5, 1800, False),
    ("헤파린나트륨주사", 10, 0.03, 1.2, 5, 6000, True),
    ("디클로페낙주 75mg", 10, 0.90, 5, 5, 400, False),
    ("니트로글리세린 설하정", 25, 0.02, 1, 5, 300, True),
    ("아스피린장용정 100mg", 100, 0.98, 40, 5, 20, False),
    ("와파린칼륨정 2mg", 100, 0.60, 3, 5, 30, False),     # 아래에서 처방 전환으로 수요 감소
    ("메트포르민정 500mg", 100, 0.98, 50, 5, 20, False),
    ("글리메피리드정 2mg", 100, 0.70, 4, 5, 60, False),
    ("심바스타틴정 20mg", 100, 0.90, 15, 5, 80, False),
    ("암로디핀베실산염정 5mg", 100, 0.98, 30, 5, 70, False),
    ("발사르탄정 80mg", 30, 0.80, 5, 5, 250, False),
    ("로수바스타틴정 10mg", 100, 0.95, 20, 5, 200, False),
    ("옴세프캡슐 100mg", 100, 0.40, 3, 1, 400, False),
    ("레보플록사신정 500mg", 100, 0.10, 3, 0.5, 800, False),
]
SWITCHED = {"와파린칼륨정 2mg": 200}    # 이 날부터 발생 확률 1/6 (신규 경구 항응고제로 처방 전환)


def make_usage(seed=0):
    rng = np.random.default_rng(seed)
    daily = {}
    for name, _, p, size, k, _, _ in DRUGS:
        pp = np.full(DAYS, p)
        pp[SWITCHED.get(name, DAYS):] /= 6
        mu = size - 1
        qty = (rng.random(DAYS) < pp) * (1 + rng.negative_binomial(k, k / (k + mu), DAYS))
        daily[name] = qty.astype(int).tolist()
    return daily


def web_lots(web):
    """웹 mock 선반의 제품을 (약품, 유효기한) 로트로 묶는다. 유효기한 미판독 제품은 뺀다."""
    src = (web / "src/mock/products.ts").read_text(encoding="utf-8")
    pack = {d[0]: d[1] for d in DRUGS}
    items = re.findall(r'id: "([^"]+)"[\s\S]*?productName: "([^"]+)"[\s\S]*?(?:expiryDate: "([^"]+)"[\s\S]*?)?status:', src)
    lots = {}
    for pid, name, exp in items:
        if exp:
            lots.setdefault((name, exp), []).append(pid)
    return [{"drug": n, "lot": ", ".join(ids), "expiry": e, "qty": len(ids) * pack[n]} for (n, e), ids in lots.items()]


def main(web):
    web = Path(web)
    daily = make_usage()
    (web / "src/mock/usage.json").write_text(json.dumps({
        "asOf": AS_OF.date().isoformat(),
        "note": "시연용 합성 데이터 — ITDA_26 pharma/export_web.py 로 생성",
        "drugs": [{"name": n, "packUnits": u, "unitPrice": pr, "essential": e} for n, u, _, _, _, pr, e in DRUGS],
        "daily": daily,
    }, ensure_ascii=False), encoding="utf-8")

    dates = [AS_OF - pd.Timedelta(days=DAYS - i) for i in range(DAYS)]
    usage = pd.DataFrame([(d, n, q) for n, qs in daily.items() for d, q in zip(dates, qs)], columns=["date", "drug", "qty"])
    lots = web_lots(web)
    out = recommend(usage, pd.DataFrame(lots), AS_OF, essential={d[0] for d in DRUGS if d[6]})
    (web / "src/lib/recommend.fixture.json").write_text(json.dumps({
        "returnDays": 180, "alpha": 0.5, "lots": lots,
        "expected": json.loads(out.assign(유효기한=out["유효기한"].astype(str)).replace(np.inf, 1e9).to_json(orient="records", force_ascii=False)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
