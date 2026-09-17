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
