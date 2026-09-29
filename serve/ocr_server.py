"""소비기한 OCR HTTP 서버 — 웹 입력 보조(카메라 스캐너)용.

  python -m serve.ocr_server [--port 8765]

  POST /ocr      본문 = 이미지 바이트(JPEG/PNG) → {"year","month","day","final_date","texts","ms"}
                 인식하지 못한 필드는 "NONE" (대회 제출 형식과 같다)
  POST /ocr?closeup=1   카메라 가이드 박스처럼 날짜가 화면을 채운 근접 크롭 (pad_closeup 참고)
  GET  /health   {"ok": true}

설정은 predict.ipynb 의 CFG 셀과 같은 운영값이다. 표준 라이브러리만 쓴다.
Engine 은 스레드 안전하지 않고(검출 입력 버퍼 공유) 한 장에 4코어를 다 쓰므로 요청은 한 번에 하나씩 처리한다.
"""
import os

THREADS = 4
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)   # cv2 / onnxruntime import 보다 먼저

import argparse
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from itda_ocr.engine import Engine
from itda_ocr.pipeline import Config, load_image, process_image

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
NONE_ROW = {"year": "NONE", "month": "NONE", "day": "NONE", "final_date": "NONE"}

CFG = Config(   # predict.ipynb CFG 셀과 같은 값 — 바꾸면 제출본과 인식 결과가 달라진다
    draft_to=720, det_side=640, top_k=2, max_k=9, threads=THREADS, impute_missing=False,
    box_thresh=0.5, unclip_ratio=1.6,
    nanodet_onnx=str(ROOT / "weights" / "date_detector_ema.onnx"), nanodet_score_thr=0.05, nanodet_expand=0.10,
    crop_tta=0.20, det_fallback=True,
    rec_onnx=str(ROOT / "weights" / "ppocrv6_rec_small_date.onnx"), rec_fallback=False,
)


def make_engine():
    return Engine(det_side=CFG.det_side, threads=CFG.threads, box_thresh=CFG.box_thresh,
                  unclip_ratio=CFG.unclip_ratio, nanodet_onnx=CFG.nanodet_onnx,
                  nanodet_score_thr=CFG.nanodet_score_thr, nanodet_expand=CFG.nanodet_expand,
                  rec_onnx=CFG.rec_onnx, rec_fallback=CFG.rec_fallback)


def pad_closeup(img, f=3):
    """날짜가 화면을 채운 근접 사진을 중앙값 색 캔버스 가운데에 놓는다 (가로 f배, 세로 1.5f배).
    검출기는 포장 전체 사진(날짜가 작게 찍힌)으로 학습돼, 가이드 박스 크롭을 그대로 넣으면
    KIST 100장 완전일치가 99 → 56 으로 떨어진다. 3배 캔버스에 놓으면 97 로 회복된다."""
    h, w = img.shape[:2]
    canvas = np.empty((int(h * f * 1.5), int(w * f), 3), img.dtype)
    canvas[:] = np.median(img.reshape(-1, 3), axis=0)
    y, x = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def recognize(engine, data: bytes, closeup=False) -> dict:
    """이미지 바이트 1장 → 결과. 어떤 입력에서도 예외를 내지 않고 NONE 행을 돌려준다.
    closeup=True 는 카메라 가이드 박스 크롭 (pad_closeup 적용), False 는 포장 전체 사진."""
    t0 = time.perf_counter()
    try:
        img = load_image(io.BytesIO(data), CFG.draft_to)
        row = process_image(engine, pad_closeup(img) if closeup else img, CFG, image_id="req")
        out = {k: row[k] for k in NONE_ROW} | {"texts": row.get("_diag", {}).get("texts", [])}
    except Exception as e:  # 깨진 이미지 등
        out = NONE_ROW | {"texts": [], "error": type(e).__name__}
    return out | {"ms": round((time.perf_counter() - t0) * 1000)}


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            url = urlsplit(self.path)
            if url.path != "/ocr":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BYTES:
                return self._send(413 if n else 400, {"error": "이미지 본문이 없거나 10MB 를 넘습니다"})
            closeup = parse_qs(url.query).get("closeup") == ["1"]
            self._send(200, recognize(engine, self.rfile.read(n), closeup))

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    t = time.time()
    engine = make_engine()
    print(f"engine ready in {time.time() - t:.2f}s — http://{args.host}:{args.port}  (POST /ocr, GET /health)")
    HTTPServer((args.host, args.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
