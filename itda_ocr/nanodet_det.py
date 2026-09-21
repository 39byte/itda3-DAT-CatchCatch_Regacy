"""NanoDet-Plus-m 날짜 전용 검출기 (ONNX) 추론 모듈."""

from __future__ import annotations

import numpy as np

_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)   # BGR
_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
_STRIDES = (8, 16, 32, 64)
_REG_MAX = 7

DEFAULT_EXPAND = 0.10
DEFAULT_NMS_IOU = 0.60


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _center_priors(size: int) -> np.ndarray:
    """``(sum(h*w), 4)`` — (cx, cy, stride, stride). nanodet get_bboxes 와 동일."""
    out = []
    for stride in _STRIDES:
        h = w = -(-size // stride)                       # ceil(size / stride)
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        px = (xs.reshape(-1) * stride).astype(np.float32)
        py = (ys.reshape(-1) * stride).astype(np.float32)
        s = np.full_like(px, stride)
        out.append(np.stack([px, py, s, s], axis=-1))
    return np.concatenate(out, axis=0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_thr]
    return keep


class NanoDetDetector:
    """ONNX NanoDet-Plus-m. ``detect(img) -> (N, 4, 2)`` (점수 내림차순)."""

    def __init__(self, onnx_path: str, input_size: int = 480, threads: int = 4,
                 score_thr: float = 0.35, nms_iou: float = DEFAULT_NMS_IOU, max_det: int = 20,
                 expand: float | tuple[float, float] = DEFAULT_EXPAND):
        import cv2
        import onnxruntime as ort

        self._cv2 = cv2
        self.input_size = input_size
        self.score_thr = score_thr
        self.nms_iou = nms_iou
        self.max_det = max_det
        if isinstance(expand, (int, float)):
            self.expand_x = float(expand)
            self.expand_y = float(expand)
        elif isinstance(expand, (tuple, list)) and len(expand) == 2:
            self.expand_x = float(expand[0])
            self.expand_y = float(expand[1])
        else:
            self.expand_x = DEFAULT_EXPAND_X
            self.expand_y = DEFAULT_EXPAND_Y
        self.expand = expand

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # 검출·인식 세션을 교대로 부르므로 끝난 세션의 스레드가 코어를 붙잡고 돌면
        # 다음 세션이 굶는다. 끄면 예측 불변·전체 2.0배 (docs/SPEED_ANALYSIS.md §2)
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._sess = ort.InferenceSession(onnx_path, sess_options=so,
                                          providers=["CPUExecutionProvider"])
        self._inp = self._sess.get_inputs()[0].name
        self._priors = _center_priors(input_size)                # (4789, 4)
        self._buf = np.empty((1, 3, input_size, input_size), np.float32)   # detect() 입력 버퍼
        self._proj = np.arange(_REG_MAX + 1, dtype=np.float32)   # [0..7]

    # ── Engine.detect 와 같은 시그니처 ────────────────────────────────────
    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        S = self.input_size
        resized = self._cv2.resize(img, (S, S), interpolation=self._cv2.INTER_LINEAR)
        # (x - mean) / std 를 채널별로 사전할당 CHW 버퍼에 바로 쓴다 (BGR, nanodet val 파이프라인과
        # 동일). 통째로 계산하면 2.7MB 임시 배열이 넷 생겨 7.8 ms, 이렇게 하면 0.74 ms.
        # 곱셈(×1/std)으로 바꾸면 마지막 자리가 달라진다 — 나눗셈을 그대로 둬야 비트 단위로 같다.
        x = self._buf
        for c in range(3):
            np.subtract(resized[:, :, c], _MEAN[c], out=x[0, c], dtype=np.float32)
            np.divide(x[0, c], _STD[c], out=x[0, c])

        out = self._sess.run(None, {self._inp: x})[0][0]          # (4789, 33)
        scores = out[:, 0]
        m = scores >= self.score_thr
        if not m.any():
            return np.empty((0, 4, 2), dtype=np.float32)

        priors = self._priors[m]
        reg = out[m, 1:].reshape(-1, 4, _REG_MAX + 1)
        dist = (_softmax(reg, axis=-1) * self._proj).sum(-1) * priors[:, 2:3]   # (n, 4) 픽셀
        cx, cy = priors[:, 0], priors[:, 1]
        boxes = np.stack([cx - dist[:, 0], cy - dist[:, 1],
                          cx + dist[:, 2], cy + dist[:, 3]], axis=-1)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, S)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, S)
        sc = scores[m]

        keep = _nms(boxes, sc, self.nms_iou)[: self.max_det]
        boxes, sc = boxes[keep], sc[keep]

        # 480 프레임 -> 입력 img 프레임
        boxes[:, 0::2] *= w / S
        boxes[:, 1::2] *= h / S

        order = sc.argsort()[::-1]                                # 점수 내림차순
        quad = np.empty((len(order), 4, 2), dtype=np.float32)
        for j, i in enumerate(order):
            x0, y0, x1, y1 = boxes[i]
            if self.expand_x or self.expand_y:
                dx = (x1 - x0) * self.expand_x
                dy = (y1 - y0) * self.expand_y
                x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy
            quad[j] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return quad
