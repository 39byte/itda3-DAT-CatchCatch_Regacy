"""약제부 유효기한 관리 모델 — 사용 내역으로 약품별 회전 속도를 추정하고,
로트마다 "유효기한 안에 다 쓸 수 있는가"를 확률로 계산해 사용 순서·경고·조치를 낸다.

방법과 근거:
  수요 분류  ADI/CV² 4분류 — Syntetos, Boylan & Croston (2005)
  수요 예측  TSB — Teunter, Syntetos & Babai (2011). 수요가 없는 주에도 발생 확률을 갱신하므로
             처방이 끊긴 약의 예측이 0 으로 내려간다 (Croston·SBA 는 내려가지 않는다).
  폐기 위험  P(남은 기한 안에 다 못 씀) — Meraklı & Küçükyavuz (2019). 원 공식(얼랑 CDF)은 포아송
             수요 가정이라, 몰려서 나오는 병원 수요에 맞게 같은 확률을 몬테카를로로 센다.
  사용 순서  FEFO — Nahmias (1982)

시간 단위는 주(週)다. 일 단위로는 저회전 약의 발생 확률이 0/1 로만 튀어 추정이 불안정하다.
"""
import numpy as np
import pandas as pd

SBC_ADI, SBC_CV2 = 1.32, 0.49


def classify(y):
    """주별 사용량 → smooth / erratic / intermittent / lumpy (사용 이력이 없으면 none)."""
    y = np.asarray(y, float)
    nz = y[y > 0]
    if len(nz) == 0:
        return "none"
    adi = len(y) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2
    return ("smooth", "erratic", "intermittent", "lumpy")[2 * (adi >= SBC_ADI) + (cv2 >= SBC_CV2)]


def tsb_step(p, z, d, a=0.1, b=0.1):
    """TSB 한 주 갱신. p = 수요 발생 확률, z = 발생했을 때 크기, 예측 = p*z. 약품 배열도 받는다."""
    occ = d > 0
    return p + b * (occ - p), np.where(occ, z + a * (d - z), z)


def tsb(y, a=0.1, b=0.1):
    y = np.asarray(y, float)
    occ = y > 0
    p, z = occ.mean(), (y[occ].mean() if occ.any() else 0.0)
    for d in y:
        p, z = tsb_step(p, z, d, a, b)
    return float(p), float(z)


def demand_paths(p, sizes, weeks, n=500, rng=None):
    """앞으로의 누적 수요 표본 (n, weeks+1). D[:, w] = 앞으로 w 주 동안 쓸 양.
    매주 확률 p 로 수요가 생기고, 크기는 최근 실제 발생 크기에서 재표집한다."""
    rng = rng or np.random.default_rng(0)
    weeks = max(int(weeks), 1)
    d = (rng.random((n, weeks)) < p) * rng.choice(np.asarray(sizes, float), (n, weeks))
    return np.concatenate([np.zeros((n, 1)), d.cumsum(1)], 1)


def lot_waste(D, weeks_left, qty):
    """로트별 폐기량 표본 (n, 로트 수). 로트는 FEFO 순서(weeks_left 오름차순)로 준다.

    FEFO 에서 로트 1..j 가 j 의 만료 시점까지 쓰이는 누적량은 U_j = min(U_{j-1} + q_j, D(w_j)) 이고
    폐기_j = q_j - (U_j - U_{j-1}) 이다. 앞 로트가 만료로 남기고 간 몫도 그대로 반영된다.
    """
    used = np.zeros(len(D))
    out = []
    for w, q in zip(weeks_left, qty):
        take = np.minimum(q, D[:, max(int(w), 0)] - used)
        used += take
        out.append(q - take)
    return np.stack(out, 1)


def weeks_to_use(D, stock):
    """현재고를 다 쓰는 데 걸리는 주 수 표본. 표본 기간 안에 못 쓰면 inf."""
    hit = D >= stock
    return np.where(hit.any(1), hit.argmax(1), np.inf)


def recommend(usage, lots, today, return_days=180, alpha=0.5, essential=(), n=500, seed=0):
    """사용 내역 + 로트 재고 → 로트별 사용 순서·폐기 위험·조치.

    usage       date, drug, qty       (불출 기록)
    lots        drug, lot, expiry, qty (현재 재고, 유효기한은 입고 때 OCR 로 읽은 값)
    return_days 반품이 받아지는 최소 잔여일. 법정 기준이 없고 제약사마다 달라 병원이 정한다.
    alpha       폐기 확률이 이 값 이상이면 경고
    essential   사용량과 무관하게 보유해야 하는 약품 (반품 대신 교환 요청)

    약품별 경고 임계일 = 현재고를 다 쓰는 데 걸리는 일수의 (1-alpha) 분위수.
    잔여 유효일이 이보다 짧은 로트가 경고 대상이다 — 회전이 빠른 약은 짧고 느린 약은 길다.
    """
    today = pd.Timestamp(today)
    rng = np.random.default_rng(seed)
    ago = (today - pd.to_datetime(usage["date"])).dt.days // 7
    hist = usage.assign(ago=ago)[ago >= 0].pivot_table("qty", "drug", "ago", aggfunc="sum", fill_value=0)
    hist = hist.reindex(columns=range(int(ago.max()) + 1), fill_value=0).iloc[:, ::-1]  # 오래된 주 → 최근 주

    rows = []
    for drug, g in lots.assign(expiry=pd.to_datetime(lots["expiry"])).sort_values("expiry").groupby("drug", sort=False):
        y = hist.loc[drug].to_numpy(float) if drug in hist.index else np.zeros(1)
        p, z = tsb(y)
        recent = y[-52:][y[-52:] > 0]
        sizes = recent if len(recent) else (y[y > 0] if (y > 0).any() else [max(z, 1.0)])
        days_left = (g["expiry"] - today).dt.days.to_numpy()
        qty = g["qty"].to_numpy(float)
        D = demand_paths(p, sizes, max(days_left.max() // 7, 156), n, rng)
        W = lot_waste(D, days_left // 7, qty)
        prob, med = (W > 0).mean(0), np.floor(np.median(W, 0))
        rate = p * z / 7
        limit = 7 * np.quantile(weeks_to_use(D, qty.sum()), 1 - alpha, method="higher")  # inf 끼리 보간하면 nan
        for k, (lot, left) in enumerate(zip(g["lot"], days_left)):
            if prob[k] < alpha:
                act = "우선 사용" if k == 0 else ""
            elif left < return_days:
                act = "폐기 예상 — 양도 검토"
            elif med[k] > 0:
                act = f"{'교환 요청' if drug in essential else '반품 권고'} {med[k]:.0f}개"
            else:
                act = "주의"
            rows.append({"약품": drug, "수요유형": classify(y), "일평균사용": round(rate, 2),
                         "재고일수": round(qty.sum() / rate) if rate > 0 else np.inf,
                         "경고임계(일)": limit, "로트": lot, "사용순서": k + 1,
                         "유효기한": g["expiry"].iloc[k].date(), "잔여일": int(left), "수량": int(qty[k]),
                         "폐기확률": round(float(prob[k]), 2), "예상폐기": int(med[k]), "조치": act})
    return pd.DataFrame(rows)
