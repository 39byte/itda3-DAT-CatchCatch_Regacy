"""OCR 엔진 래퍼 모듈."""

from __future__ import annotations

import os

import numpy as np

from .nanodet_det import DEFAULT_EXPAND, DEFAULT_NMS_IOU

DEFAULT_THREADS = 4


def pin_threads(n: int = DEFAULT_THREADS) -> None:
    """스레드 수를 환경변수로 고정한다."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


def _disable_rapidocr_spinning() -> None:
    """RapidOCR 세션(det/cls/rec)의 intra-op 스레드 spinning 을 끈다.

    RapidOCR 은 SessionOptions 를 내부 정적 메서드에서 만들어 인자로 끌 수 없으므로
    RapidOCR() 생성 전에 그 메서드를 감싼다. 이유는 nanodet_det.py 의 같은 설정 참조.
    """
    from rapidocr_onnxruntime.utils.infer_engine import OrtInferSession

    orig = OrtInferSession._init_sess_opts
    if getattr(orig, "_no_spin", False):
        return

    def init_sess_opts(config):
        opts = orig(config)
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return opts

    init_sess_opts._no_spin = True
    OrtInferSession._init_sess_opts = staticmethod(init_sess_opts)


class DateCTCLabelDecode:
    """CTC 디코딩 단계에서 비라틴/비숫자 노이즈 토큰 마스킹."""

    def __init__(self, original_op, allowed_chars: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-/:, ()[]~年月日"):
        self.original_op = original_op
        self.character = original_op.character
        allowed_set = set(allowed_chars)
        self.disallowed = np.array([
            i for i, c in enumerate(self.character)
            if i != 0 and c != 'blank' and c not in allowed_set
        ], dtype=np.int64)

    def __getattr__(self, name):
        return getattr(self.original_op, name)

    def __call__(self, preds, *args, **kwargs):
        preds[:, :, self.disallowed] = -np.inf
        return self.original_op(preds, *args, **kwargs)


class Engine:
    """검출·방향분류·인식 래퍼."""

    def __init__(self, det_side: int = 640, threads: int = DEFAULT_THREADS,
                 box_thresh: float = 0.5, unclip_ratio: float = 1.6,
                 text_score: float = 0.0, nanodet_onnx: str | None = None,
                 nanodet_score_thr: float = 0.05,
                 nanodet_expand: float | tuple[float, float] = DEFAULT_EXPAND,
                 nanodet_nms_iou: float = DEFAULT_NMS_IOU,
                 rec_onnx: str | None = None):
        pin_threads(threads)
        import cv2
        from rapidocr_onnxruntime import RapidOCR
        from rapidocr_onnxruntime.ch_ppocr_det.utils import DetPreProcess

        cv2.setNumThreads(threads)
        self._cv2 = cv2
        self.det_side = det_side

        self._nanodet = None
        if nanodet_onnx:
            from .nanodet_det import NanoDetDetector
            self._nanodet = NanoDetDetector(nanodet_onnx, threads=threads,
                                            score_thr=nanodet_score_thr,
                                            nms_iou=nanodet_nms_iou,
                                            expand=nanodet_expand)

        _disable_rapidocr_spinning()
        # rec_onnx: 번들 PP-OCRv4 대신 쓸 인식기(PP-OCRv6 small). 문자 사전은 RapidOCR 이
        # ONNX 메타데이터에서 읽는다. 교체 근거는 docs/RECOGNIZER_DECISION.md
        self._ocr = RapidOCR(
            intra_op_num_threads=threads,
            inter_op_num_threads=1,
            det_box_thresh=box_thresh,
            det_unclip_ratio=unclip_ratio,
            text_score=text_score,
            **({"rec_model_path": rec_onnx} if rec_onnx else {}),
        )
        self._det = self._ocr.text_det
        self._cls = self._ocr.text_cls
        self._rec = self._ocr.text_rec
        self._rec.postprocess_op = DateCTCLabelDecode(self._rec.postprocess_op)
        self._pre = DetPreProcess(det_side, "max", self._det.mean, self._det.std)

    # ── 검출 ───────────────────────────────────────────────────────────────
    def detect(self, img: np.ndarray) -> np.ndarray:
        """텍스트 박스 검출."""
        if self._nanodet is not None:
            return self._nanodet.detect(img)
        tensor = self._pre(img)
        if tensor is None:
            return np.empty((0, 4, 2), dtype=np.float32)
        preds = self._det.infer(tensor)[0]
        boxes, _ = self._det.postprocess_op(preds, img.shape[:2])
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32)
        return self._det.filter_tag_det_res(boxes, img.shape[:2])

    def detect_and_filter(self, img: np.ndarray):
        """검출 및 박스 필터링/정렬."""
        boxes = self.detect(img)
        return filter_boxes(boxes, img.shape, rerank=self._nanodet is None)

    # ── 인식 ───────────────────────────────────────────────────────────────
    def recognize(self, crops: list[np.ndarray], use_cls: bool = True):
        """크롭들을 **한 번에 배치로** 인식한다. 반환 ``[(text, score), ...]``.

        기본 패키징의 95% 이상이 정방향(0°)이므로, 1차는 cls 없이 바로 인식하고
        숫자가 전혀 검출되지 않을 때에만 조건부(Lazy)로 방향 분류기(_cls)를 호출한다.
        """
        if not crops:
            return []
        rec_res = self._rec(crops)[0]
        texts = [str(t) for t, _ in rec_res]
        # 크롭들 중 최소 하나라도 숫자 3개 이상(연/월/일 파편)이 잡히면 정상 방향으로 판단
        if any(sum(c.isdigit() for c in t) >= 3 for t in texts) or not use_cls:
            return [(str(t), float(s)) for t, s in rec_res]
        # 180도 역방향 크롭 구제: 숫자가 전혀 안 잡힐 때에만 _cls 실행 후 재인식
        oriented_crops = self._cls(crops)[0]
        rec_res2 = self._rec(oriented_crops)[0]
        return [(str(t), float(s)) for t, s in rec_res2]

    # ── 크롭 ───────────────────────────────────────────────────────────────
    #: 인식기의 입력 높이. 크롭을 이보다 크게 키우는 건 순수한 낭비다 —
    REC_HEIGHT = 48

    def crop(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """박스를 잘라내고 필요 시 최소 높이로 리사이즈한다."""
        h, w = img.shape[:2]
        xs, ys = box[:, 0], box[:, 1]
        x0, x1 = max(int(xs.min()), 0), min(int(np.ceil(xs.max())), w)
        y0, y1 = max(int(ys.min()), 0), min(int(np.ceil(ys.max())), h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return np.empty((0, 0, 3), dtype=img.dtype)

        patch = img[y0:y1, x0:x1]
        ph = patch.shape[0]
        if ph < self.REC_HEIGHT:
            scale = min(4.0, self.REC_HEIGHT / max(ph, 1))
            patch = self._cv2.resize(
                patch, (max(int(patch.shape[1] * scale), 1), max(int(ph * scale), 1)),
                interpolation=self._cv2.INTER_LINEAR)
        MAX_REC_WIDTH = 320
        if patch.shape[1] > MAX_REC_WIDTH:
            scale_w = MAX_REC_WIDTH / patch.shape[1]
            patch = self._cv2.resize(
                patch, (MAX_REC_WIDTH, max(int(patch.shape[0] * scale_w), 16)),
                interpolation=self._cv2.INTER_AREA)
        return np.ascontiguousarray(patch)


def box_metrics(box: np.ndarray) -> tuple[float, float, float]:
    """(너비, 높이, 종횡비) 계산."""
    xs, ys = box[:, 0], box[:, 1]
    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    return w, h, (w / h if h > 0 else 0.0)


def filter_boxes(boxes: np.ndarray, img_shape, *, min_height: float = 6.0,
                 min_ratio: float = 1.2, max_ratio: float = 25.0,
                 max_width_frac: float = 0.95, rerank: bool = True):
    """기하 제약으로 노이즈 박스를 필터링하고 정렬한다."""
    if len(boxes) == 0:
        return []
    ih, iw = img_shape[:2]
    kept = []
    for i, box in enumerate(boxes):
        w, h, ratio = box_metrics(box)
        if h < min_height or w < min_height:
            continue
        if not (min_ratio <= ratio <= max_ratio):
            continue
        if w > iw * max_width_frac:
            continue
        kept.append((_date_prior(w, h, ratio), i, box))
    if rerank:
        kept.sort(key=lambda t: -t[0])
    return kept


DATE_ASPECT = 5.4


def _date_prior(w: float, h: float, ratio: float) -> float:
    """박스 기하 기반 사전 점수 계산."""
    ratio_fit = -abs(ratio - DATE_ASPECT) / DATE_ASPECT
    return ratio_fit + min(h, 60.0) / 120.0
