"""NanoDet 전처리를 버퍼 재사용으로 바꿔도 모델 입력이 비트 단위로 같아야 한다."""

from pathlib import Path

import cv2
import numpy as np

from itda_ocr.nanodet_det import _MEAN, _STD, NanoDetDetector

NANODET = Path(__file__).resolve().parents[1] / "weights" / "date_detector_ema.onnx"


def test_detect_input_is_bit_identical_to_the_reference_formula():
    det = NanoDetDetector(str(NANODET), score_thr=0.05)
    seen = []
    orig_run = det._sess.run

    class Spy:                                   # InferenceSession.run 은 C 메서드라 감싸서 가로챈다
        def run(self, out, feed):
            seen.append(next(iter(feed.values())).copy())
            return orig_run(out, feed)

    det._sess = Spy()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (600, 800, 3), np.uint8)[:, :, ::-1]   # load_image 처럼 음수 stride 뷰
    for _ in range(2):                           # 두 번째 호출이 첫 호출의 버퍼 잔여물에 오염되지 않는다
        det.detect(img)

    r = cv2.resize(img, (480, 480), interpolation=cv2.INTER_LINEAR)
    ref = np.ascontiguousarray(((r.astype(np.float32) - _MEAN) / _STD).transpose(2, 0, 1)[None])
    assert all(np.array_equal(x, ref) for x in seen) and len(seen) == 2
