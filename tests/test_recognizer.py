"""PP-OCRv6 small 인식기 교체가 실제로 적용됐는지 확인한다.

파일이 없거나 손상되면 predict.ipynb 는 번들 PP-OCRv4 로 조용히 돌아가 점수만 떨어진다
(docs/RECOGNIZER_DECISION.md). 그 조용한 폴백을 여기서 잡는다.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from itda_ocr.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "weights" / "ppocrv6_rec_small.onnx"
REC_SHA256 = "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"


def test_weight_is_present_and_intact():
    assert REC.exists(), f"{REC} 가 없습니다 — .gitignore 예외 또는 저장소 포함 여부를 확인하세요"
    assert hashlib.sha256(REC.read_bytes()).hexdigest() == REC_SHA256


def test_notebook_points_to_the_weight():
    nb = json.loads((ROOT / "predict.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert '"ppocrv6_rec_small.onnx"' in code and "rec_onnx=CFG.rec_onnx" in code


def test_engine_loads_v6_charset_and_reads_a_date():
    engine = Engine(nanodet_onnx=str(ROOT / "weights" / "date_detector_ema.onnx"), rec_onnx=str(REC))
    sess = engine._rec.session.session
    assert sess.get_outputs()[0].shape[-1] == len(engine._rec.postprocess_op.character) == 18710

    img = np.full((48, 260, 3), 255, np.uint8)
    cv2.putText(img, "2024.05.17", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    assert engine.recognize([img])[0][0] == "2024.05.17"


def test_notebook_uses_crop_tta_instead_of_v4_fallback():
    """크롭 TTA 를 켜고 v4 폴백은 끈다.

    TTA 위에서 폴백의 기여는 +0.051 로 줄고(단독 +0.133) 꼬리 지연만 커진다 —
    1,473장 실측은 README §3 의 "[3] 인식 — 크롭 TTA" 참조.
    """
    nb = json.loads((ROOT / "predict.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "crop_tta=0.20" in code
    assert "rec_fallback=False" in code and "rec_fallback=CFG.rec_fallback" in code


def test_fallback_recognizer_is_bundled_v4():
    engine = Engine(nanodet_onnx=str(ROOT / "weights" / "date_detector_ema.onnx"),
                    rec_onnx=str(REC), rec_fallback=True)
    assert engine.has_fallback
    assert len(engine._rec_fallback.postprocess_op.character) != 18710   # v6 사전이 아니다

    img = np.full((48, 260, 3), 255, np.uint8)
    cv2.putText(img, "2024.05.17", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    assert engine.recognize([img], fallback=True)[0][0] == "2024.05.17"


class _FakeEngine:
    """주 인식기와 폴백 인식기가 정해진 글자를 돌려주는 가짜 엔진."""

    has_fallback = True

    def __init__(self, primary, fallback):
        self.reads = {False: primary, True: fallback}
        self.calls = []

    def detect_and_filter(self, img):
        return [(1.0, 0, np.array([[0, 0], [50, 0], [50, 10], [0, 10]], np.float32))]

    def crop(self, img, box, expand=0.0):
        return np.ones((10, 50, 3), np.uint8)

    def recognize(self, crops, fallback=False):
        self.calls.append(fallback)
        return [(self.reads[fallback], 0.9)] * len(crops)


def _run(primary, fallback):
    from itda_ocr.pipeline import Config, process_image
    engine = _FakeEngine(primary, fallback)
    row = process_image(engine, np.zeros((20, 60, 3), np.uint8), Config())
    return row, engine.calls


def test_fallback_fills_only_when_primary_has_no_full_date():
    row, calls = _run("2024.05.17", "2023.01.01")
    assert row["final_date"] == "2024-05-17" and calls == [False]            # 폴백 안 부름

    row, calls = _run("es.01.250s", "2025.10.29")
    assert row["final_date"] == "2025-10-29" and row["_diag"]["fallback"]    # 빈자리를 채움

    row, _ = _run("2021.11", "no date")
    assert (row["year"], row["month"], row["final_date"]) == ("2021", "11", "NONE")  # 부분 추출 보존


def test_fallback_does_not_overwrite_matching_partial_date():
    """주 인식기가 연·월을 냈으면 어긋나는 폴백 날짜는 버린다 (test_00108, test_00593 퇴행)."""
    row, calls = _run("2021.12", "2021-12-25")       # 연·월 일치 → 채택
    assert row["final_date"] == "2021-12-25" and calls == [False, True]

    row, calls = _run("2021.12", "2023-01-05")       # 연·월 불일치 → 부분 답 유지
    assert (row["year"], row["month"], row["final_date"]) == ("2021", "12", "NONE")
    assert calls == [False, True] and not row["_diag"]["fallback"]


class _TTAFakeEngine:
    """여백에 따라 다른 글자를 돌려주는 가짜 엔진 — 좁은 크롭은 끝 글자가 잘린다."""

    has_fallback = False

    def __init__(self, narrow, wide):
        self.by_marker = {0: narrow, 1: wide}
        self.expands = []

    def detect_and_filter(self, img):
        return [(1.0, 0, np.array([[0, 0], [50, 0], [50, 10], [0, 10]], np.float32))]

    def crop(self, img, box, expand=0.0):
        self.expands.append(expand)
        return np.full((10, 50, 3), 1 if expand else 0, np.uint8)

    def recognize(self, crops, fallback=False):
        return [(self.by_marker[int(c[0, 0, 0])], 0.9) for c in crops]


def test_crop_tta_reads_each_box_twice_and_prefers_the_untruncated_reading():
    """절단 판독(`2025.12.2`)이 온전한 판독(`2025.12.24`)에 밀려야 한다."""
    from itda_ocr.pipeline import Config, process_image

    engine = _TTAFakeEngine("2025.12.2", "2025.12.24")
    row = process_image(engine, np.zeros((20, 60, 3), np.uint8), Config(crop_tta=0.20))
    assert engine.expands == [0.0, 0.20]                  # 박스당 크롭 2개
    assert row["final_date"] == "2025-12-24"

    engine = _TTAFakeEngine("2025.12.2", "2025.12.24")    # 끄면 잘린 판독만 본다
    row = process_image(engine, np.zeros((20, 60, 3), np.uint8), Config())
    assert engine.expands == [0.0]
    assert row["final_date"] == "2025-12-02"


def test_crop_expand_widens_the_patch_and_is_a_noop_at_zero():
    """`Engine.crop(expand=)` 은 여백만 늘리고, 0 이면 기존 동작과 같아야 한다."""
    engine = Engine.__new__(Engine)                       # 모델 로드 없이 crop 만 쓴다
    engine._cv2 = cv2
    img = np.zeros((300, 400, 3), np.uint8)
    box = np.array([[100, 80], [200, 80], [200, 140], [100, 140]], np.float32)

    base = engine.crop(img, box)                          # 높이 60 ≥ REC_HEIGHT → 리사이즈 없음
    assert base.shape[:2] == (60, 100)
    wide = engine.crop(img, box, 0.20)                    # 좌우 ±20px, 상하 ±12px
    assert wide.shape[:2] == (84, 140)
    assert np.array_equal(engine.crop(img, box, 0.0), base)
