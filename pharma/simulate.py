"""합성 병원(약제부 중앙 창고 1곳)에서 유효기한 경고 규칙을 비교한다.

2년을 돌려 1년차는 이력 쌓기, 2년차만 집계한다. 수요·입고 유효기한·발주 규칙은 모든 경고 규칙이
같은 난수를 쓴다. 규칙마다 다른 것은 두 가지다 — 어떤 로트에 경고하느냐(일괄 D-T / 약품별 동적),
그리고 반품 기한 직전에 남을 양을 무엇으로 추정하느냐(평균 사용량 / 모델).

  python -m pharma.simulate

합성 병원의 수치(품목 구성·포장 단위·임박 입고 비율·단가)는 전부 가정이다. 실측 근거가 없는 대신
기준 폐기율이 문헌 범위(에티오피아 공공병원 평균 4.87%, 허용 기준 2% — Getahun et al. 2024)에
들어오는지로 현실성을 점검한다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharma.model import classify, demand_paths, lot_waste, recommend, tsb_step

N, DAYS, EVAL = 300, 730, 365     # 품목 수, 시뮬레이션 일수, 집계 시작일
LEAD = 3                          # 발주 → 입고 (일)
REFUND = 0.9                      # 반품 정산율 (가정)
PACKS = np.array([10, 30, 100, 300, 1000])
ALPHA = 0.5


@dataclass
class Lot:
    drug: int
    exp: int          # 실제 유효기한 (일)
    rec: int          # 시스템에 기록된 유효기한 (OCR 판독)
    qty: float
    alerted: bool = False
    returned: float = 0
    expired: float = 0


def make_hospital(seed=0, ocr_err=0.0):
    """약품 N 개의 2년치 일별 수요와 입고 조건을 만든다."""
    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), N))
    kind = rng.choice(["fast", "slow", "essential"], N, p=[0.4, 0.52, 0.08])
    fast, slow = kind == "fast", kind == "slow"
    p = np.select([fast, slow], [rng.uniform(0.7, 1, N), lu(0.01, 0.5)], lu(0.002, 0.02))
    size = np.select([fast, slow], [lu(2, 50), lu(1, 10)], lu(1, 3))
    k = rng.choice([0.5, 5.0], N)                     # 크기 산포: 0.5 = 몰림, 5 = 고름
    occ = rng.random((N, DAYS)) < p[:, None]
    drop = np.where((kind != "essential") & (rng.random(N) < 0.1), rng.integers(180, 700, N), DAYS)
    occ &= (np.arange(DAYS) < drop[:, None]) | (rng.random((N, DAYS)) < 0.1)   # 처방 중단 → 수요 90% 감소
    mu = (size - 1)[:, None]
    demand = occ * (1 + rng.negative_binomial(k[:, None], k[:, None] / (k[:, None] + mu), (N, DAYS)))

    need = 14 * p * size                              # 2주치 사용량보다 큰 가장 작은 포장, 30% 는 한 단계 큼
    pack = PACKS[np.minimum(np.searchsorted(PACKS, need) + (rng.random(N) < 0.3), len(PACKS) - 1)]
    price = rng.lognormal(np.log(np.select([fast, slow], [100, 2000], 20000)), 1.0)   # 다빈도 약일수록 싸다 (원/단위)
    R = 120                                           # 품목당 입고 횟수 상한
    shelf = np.where(rng.random((N, R)) < 0.1, rng.integers(90, 270, (N, R)), rng.integers(540, 1095, (N, R)))
    ocr = np.where(rng.random((N, R)) < ocr_err,
                   rng.choice([-1, 1], (N, R)) * 30 * rng.integers(1, 13, (N, R)), 0)   # 월·연 한 자리 오독
    return dict(kind=kind, demand=demand.astype(float), pack=pack, price=price, shelf=shelf, ocr=ocr, drop=drop,
                p0=1 - (1 - p) ** 7, z0=7 * p * size / (1 - (1 - p) ** 7))


def run(h, rule, est="model", return_days=180, n=300, seed=0):
    """rule  None(경고 없음) · 정수 T(잔여 T일 이하 일괄 경고) · "dyn"(폐기 확률 ≥ ALPHA 인 로트만)
    est   반품할 잉여분 추정 — "model"(TSB + 몬테카를로 중앙값) · "avg"(최근 1년 평균 사용량)"""
    rng = np.random.default_rng(seed)
    dem, pack, price = h["demand"], h["pack"], h["price"]
    essential = h["kind"] == "essential"
    p, z = h["p0"].copy(), h["z0"].copy()
    weekly = np.zeros((N, DAYS // 7 + 1))
    lots = [[] for _ in range(N)]
    every, orders = [], {}
    on_order = np.zeros(N)
    nrec = np.zeros(N, int)
    m = dict(demand=0.0, served=0.0, expired=0.0, returned=0.0, bought=0.0, alerts=0)

    def receive(i, q, t):
        j = nrec[i] = nrec[i] + 1
        lot = Lot(i, t + h["shelf"][i, j], t + h["shelf"][i, j] + h["ocr"][i, j], q)
        lots[i].append(lot)
        lots[i].sort(key=lambda l: l.rec)            # FEFO — 시스템이 아는 유효기한 순
        every.append(lot)
        if t >= EVAL:
            m["bought"] += q * price[i]

    def model_waste(i, y, left, qty):
        sizes = y[y > 0] if (y > 0).any() else [max(z[i], 1.0)]
        return lot_waste(demand_paths(p[i], sizes, left.max() // 7, n, rng), left // 7, qty)

    def review(i, t):
        """경고 규칙으로 로트에 경고를 붙이고, 경고된 로트는 반품 기한 직전 주에 남을 양만 반품한다.
        반품을 일찍 하면 재고가 줄어 곧바로 재발주되고 새 로트가 다시 위험해지는 순환이 생긴다."""
        L = lots[i]
        left = np.array([l.rec - t for l in L])
        qty = np.array([l.qty for l in L])
        y = weekly[i, max(0, t // 7 - 51):t // 7 + 1]
        W = None
        if rule == "dyn" and qty.sum() > 0.5 * p[i] * z[i] * (left.min() // 7):   # 명백히 다 쓰는 약은 건너뜀
            W = model_waste(i, y, left, qty)
        for k, l in enumerate(L):
            # 일괄 규칙은 이번 주 안에 D-T 에 닿을 로트를 잡는다 (주 1회 점검이 기한을 놓치지 않게)
            hit = (W is not None and (W[:, k] > 0).mean() >= ALPHA) if rule == "dyn" else left[k] < rule + 7
            if hit and not l.alerted:
                l.alerted = True
                m["alerts"] += 1
        due = [k for k, l in enumerate(L) if l.alerted and return_days <= left[k] < return_days + 7]
        if not due:
            return
        if est == "avg":    # 담당자 판단 모사: 최근 1년 평균 사용량으로 소진을 계산
            surplus = lot_waste((y.mean() * np.arange(left.max() // 7 + 1))[None], left // 7, qty)[0]
        else:
            surplus = np.median(W if W is not None else model_waste(i, y, left, qty), 0)
        for k in due:
            r = np.floor(surplus[k])
            if r > 0:
                L[k].qty -= r
                L[k].returned += r
                m["returned"] += r * price[i]

    for i in range(N):
        receive(i, max(pack[i], np.ceil(4 * p[i] * z[i] / pack[i]) * pack[i]), 0)

    for t in range(DAYS):
        for i, q in orders.pop(t, []):
            on_order[i] -= q
            receive(i, q, t)
        for i in np.flatnonzero(dem[:, t]):
            need = dem[i, t]
            for lot in lots[i]:
                take = min(lot.qty, need)
                lot.qty -= take
                need -= take
            if t >= EVAL:
                m["demand"] += dem[i, t]
                m["served"] += dem[i, t] - need
        for i in range(N):
            if lots[i] and min(l.exp for l in lots[i]) <= t or any(l.qty <= 0 for l in lots[i]):
                for lot in lots[i]:
                    if lot.exp <= t and lot.qty > 0:
                        lot.expired = lot.qty
                        if t >= EVAL:
                            m["expired"] += lot.qty * price[i]
                lots[i] = [l for l in lots[i] if l.exp > t and l.qty > 0]

        if t % 7 != 6:
            continue
        w = t // 7
        weekly[:, w] = dem[:, t - 6:t + 1].sum(1)
        p, z = tsb_step(p, z, weekly[:, w])
        if t >= EVAL and rule is not None:
            for i in range(N):
                if lots[i]:
                    review(i, t)
        f = p * z                                     # 주간 예측
        s = np.where(essential, np.maximum(2 * f, pack), 2 * f)
        S = np.where(essential, np.maximum(4 * f, pack), 4 * f)
        ip = np.array([sum(l.qty for l in L) for L in lots]) + on_order
        for i in np.flatnonzero(ip < s):
            q = max(1, np.ceil((S[i] - ip[i]) / pack[i])) * pack[i]
            orders.setdefault(t + LEAD, []).append((i, q))
            on_order[i] += q

    judged = [l for l in every if l.alerted and (l.returned or l.expired or l.qty <= 0)]
    wasted = sum(1 for l in judged if not l.returned and not l.expired)
    return {"폐기(만원)": m["expired"] / 1e4, "반품손실(만원)": (1 - REFUND) * m["returned"] / 1e4,
            "총손실(만원)": (m["expired"] + (1 - REFUND) * m["returned"]) / 1e4,
            "폐기율(%)": 100 * m["expired"] / m["bought"], "충족률(%)": 100 * m["served"] / m["demand"],
            "경보(건)": m["alerts"], "헛경보(%)": 100 * wasted / max(len(judged), 1)}, lots


def forecast_table(h):
    """1년차로 분류, 2년차 매주 1주 앞 예측의 RMSSE (Hyndman & Koehler 2006 의 제곱 버전, M5 채택)."""
    Y = h["demand"][:, :DAYS // 7 * 7].reshape(N, -1, 7).sum(2)
    T = Y.shape[1] // 2
    tr = Y[:, :T]
    a = 0.1
    naive = np.zeros(N)
    ses = tr.mean(1)
    cz = np.array([r[r > 0].mean() if (r > 0).any() else 0 for r in tr])
    cx = np.array([len(r) / max((r > 0).sum(), 1) for r in tr])
    q = np.ones(N)
    p, z = (tr > 0).mean(1), cz.copy()
    err = {k: [] for k in ["Naive", "SES", "Croston", "SBA", "TSB"]}
    for w in range(Y.shape[1]):
        y = Y[:, w]
        if w >= T:
            for k, f in zip(err, [naive, ses, cz / cx, (1 - a / 2) * cz / cx, p * z]):
                err[k].append((y - f) ** 2)
        naive = y
        ses = ses + a * (y - ses)
        occ = y > 0
        cz, cx = np.where(occ, cz + a * (y - cz), cz), np.where(occ, cx + a * (q - cx), cx)
        q = np.where(occ, 1, q + 1)
        p, z = tsb_step(p, z, y)
    scale = (np.diff(tr, axis=1) ** 2).mean(1)
    ok = scale > 0
    df = pd.DataFrame({k: np.sqrt(np.mean(v, 0)[ok] / scale[ok]) for k, v in err.items()})
    df["유형"] = [classify(r) for r in tr[ok]]
    stopped = (h["drop"][ok] >= T * 7) & (h["drop"][ok] < DAYS - 28)   # 2년차에 처방이 끊긴 약
    num = df.drop(columns="유형")
    return pd.concat([df.groupby("유형").mean(), num.mean().to_frame("전체").T,
                      num[stopped].mean().to_frame(f"처방 중단 {stopped.sum()}종").T]).round(3), df["유형"].value_counts()


def demo(h, per_kind=2, return_days=180):
    """경고 없이 2년을 돈 창고의 마지막 날 재고에 recommend() 를 돌린다 — 발표용 예시표.
    입력은 실제 운영과 같은 모양이다: 불출 기록(date, drug, qty) + 로트 재고(drug, lot, expiry, qty)."""
    _, lots = run(h, None)
    t, day0 = DAYS - 1, pd.Timestamp("2025-01-01")
    drugs = [i for kind in ("fast", "slow", "essential")
             for i in [i for i in range(N) if h["kind"][i] == kind and lots[i]][:per_kind]]
    usage = pd.DataFrame([(day0 + pd.Timedelta(days=d), i, h["demand"][i, d])
                          for i in drugs for d in range(DAYS) if h["demand"][i, d] > 0],
                         columns=["date", "drug", "qty"])
    stock = pd.DataFrame([(i, f"L{i}-{j}", day0 + pd.Timedelta(days=int(l.rec)), l.qty)
                          for i in drugs for j, l in enumerate(lots[i])], columns=["drug", "lot", "expiry", "qty"])
    ess = set(np.flatnonzero(h["kind"] == "essential"))
    return recommend(usage, stock, day0 + pd.Timedelta(days=t), return_days, ALPHA, ess)


def main(seeds=range(5)):
    pd.set_option("display.width", 200)
    h = make_hospital(0)
    table, counts = forecast_table(h)
    print("## 수요 유형 (1년차 주별, ADI/CV²)\n", counts.to_string(), "\n")
    print("## 예측 정확도 RMSSE (2년차, 1주 앞, 낮을수록 좋음)\n", table.to_string(), "\n")

    rules = [(None, "model", "경고 없음"), (90, "avg", "일괄 D-90 · 평균"), (180, "avg", "일괄 D-180 · 평균"),
             (180, "model", "일괄 D-180 · 모델"), ("dyn", "model", "약품별 동적 · 모델")]
    for rd in (180, 90, 0):
        res = []
        for s in seeds:
            hs = make_hospital(s)
            for rule, est, name in rules:
                res.append({"규칙": name, **run(hs, rule, est, return_days=rd, seed=s)[0]})
        df = pd.DataFrame(res).groupby("규칙", sort=False).mean().round(1)
        base = df.loc["경고 없음", "총손실(만원)"]
        df["손실절감(%)"] = (100 * (1 - df["총손실(만원)"] / base)).round(1)
        print(f"## 반품 가능 잔여일 ≥ {rd}일 (시드 {len(seeds)}개 평균)\n", df.to_string(), "\n")

    res = []
    for e in (0, 0.05, 0.15, 0.3):
        for s in seeds:
            res.append({"OCR 미검출 오독률": e, **run(make_hospital(s, ocr_err=e), "dyn", seed=s)[0]})
    print("## OCR 오독이 동적 경고에 미치는 영향 (반품 ≥180일)\n",
          pd.DataFrame(res).groupby("OCR 미검출 오독률").mean().round(1).to_string(), "\n")

    print("## 추천 결과 예시\n", demo(h).to_string(index=False))


if __name__ == "__main__":
    main()
