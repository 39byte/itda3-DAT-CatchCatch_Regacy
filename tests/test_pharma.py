import numpy as np
import pandas as pd

from pharma.model import classify, lot_waste, recommend, tsb


def test_lot_waste_matches_hand_fefo():
    # 매주 3개씩 쓴다. 로트1 은 2주 뒤, 로트2 는 5주 뒤 만료, 각 10개.
    # 로트1: 6개 쓰고 4개 폐기. 로트2: 3~5주차 9개 쓰고 1개 폐기.
    D = np.array([[0, 3, 6, 9, 12, 15]], float)
    assert lot_waste(D, [2, 5], [10, 10]).tolist() == [[4, 1]]
    # 앞 로트가 먼저 떨어지면 뒤 로트가 그만큼 일찍 쓰인다
    assert lot_waste(D, [2, 5], [2, 10]).tolist() == [[0, 0]]


def test_tsb_falls_when_prescribing_stops():
    p, z = tsb([5] * 20 + [0] * 30)
    assert p * z < 0.1 * 5


def test_classify_quadrants():
    assert classify([5, 5, 5, 5]) == "smooth"
    assert classify([1, 20, 1, 20]) == "erratic"
    assert classify([5, 0, 0, 5, 0, 0]) == "intermittent"
    assert classify([1, 0, 0, 30, 0, 0]) == "lumpy"


def test_recommend_flags_slow_drug_not_fast_one():
    today = pd.Timestamp("2026-01-01")
    days = pd.date_range(end=today - pd.Timedelta(days=1), periods=364)
    usage = pd.concat([pd.DataFrame({"date": days, "drug": "fast", "qty": 10}),
                       pd.DataFrame({"date": days[::60], "drug": "slow", "qty": 1})])
    lots = pd.DataFrame({"drug": ["fast", "slow", "slow"], "lot": ["F1", "S1", "S2"],
                         "expiry": today + pd.to_timedelta([120, 400, 300], "D"), "qty": [500, 30, 30]})
    out = recommend(usage, lots, today).set_index("로트")
    assert out.loc["F1", "조치"] == "우선 사용"
    assert out.loc["S2", "사용순서"] == 1                 # FEFO: 기한이 빠른 S2 먼저
    assert out.loc["S1", "조치"].startswith("반품 권고")    # 60일에 1개 쓰는 약 30개는 기한 안에 못 쓴다
    by_drug = out.groupby("약품")["경고임계(일)"].first()
    assert by_drug["fast"] < 90 < by_drug["slow"]          # 약품별 임계: 빨리 도는 약은 짧고 느린 약은 길다
