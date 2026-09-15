"""이미지 로딩, 장당 파이프라인 및 배치 처리 드라이버 모듈."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .nanodet_det import DEFAULT_EXPAND, DEFAULT_NMS_IOU
from .parse import parse_boxes
from .select import STOP_SCORE, score as sel_score, select, to_row

FIELDNAMES = ["image_id", "year", "month", "day", "final_date"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Config:
    """파이프라인 실행 설정."""

    draft_to: int = 720
    det_side: int = 640
    top_k: int = 2
    max_k: int = 9
    threads: int = 4
    box_thresh: float = 0.5
    unclip_ratio: float = 1.6
    impute_missing: bool = False
    per_image_budget: float = 0.15
    flush_every: int = 50
    nanodet_onnx: str | None = None
    nanodet_score_thr: float = 0.05
    nanodet_expand: float | tuple[float, float] = DEFAULT_EXPAND
    nanodet_nms_iou: float = DEFAULT_NMS_IOU


def load_image(path, draft_to: int = 720) -> np.ndarray:
    """BGR 배열로 읽는다. **EXIF 회전 보정이 무조건 먼저.**

    표본 조사에서 "90° 회전 문제"로 지목된 이미지가 전부 EXIF Orientation=6
    이었다(293장, 8.7%). 이 한 줄이 전체 이미지 회전 TTA를 불필요하게 만든다.

    ``draft()`` 는 JPEG를 DCT 단계에서 1/2·1/4·1/8로 **디코딩하며** 줄인다.
    전부 디코딩한 뒤 리사이즈하는 것보다 훨씬 싸다.
    """
    im = Image.open(path)
    if draft_to:
        im.draft("RGB", (draft_to, draft_to))
    exif = im.getexif()
    orientation = exif.get(0x0112) if exif else None
    if orientation not in (1, None):
        im = ImageOps.exif_transpose(im)
    arr = np.asarray(im if im.mode == "RGB" else im.convert("RGB"))
    return arr[:, :, ::-1]


def iter_images(input_dir) -> list[Path]:
    """디렉터리 내 지원 이미지 파일 목록을 정렬하여 반환한다."""
    return sorted(p for p in Path(input_dir).glob("*.*")
                  if p.suffix.lower() in IMAGE_SUFFIXES)


def process_image(engine, path_or_img, cfg: Config, top_k: int | None = None,
                  max_k: int | None = None, image_id: str | None = None) -> dict:
    """이미지 1장 처리: 검출 -> 필터링 -> 배치 인식 -> 파싱 -> 선별."""
    t0 = time.perf_counter()
    if isinstance(path_or_img, np.ndarray):
        img = path_or_img
        image_id = image_id or "image"
        t_load = time.perf_counter()
    else:
        image_id = image_id or Path(path_or_img).stem
        img = load_image(path_or_img, cfg.draft_to)
        t_load = time.perf_counter()

    kept = engine.detect_and_filter(img)
    t_det = time.perf_counter()

    batch = top_k if top_k is not None else cfg.top_k
    limit = max_k if max_k is not None else cfg.max_k

    texts, geoms, items, candidates = [], [], [], []
    for start in range(0, min(len(kept), limit), batch):
        crops, batch_geoms = [], []
        for _, _, box in kept[start:start + batch]:
            patch = engine.crop(img, box)
            if patch.size:
                crops.append(patch)
                xs, ys = box[:, 0], box[:, 1]
                batch_geoms.append((float(xs.min()), float(ys.min()),
                                    float(xs.max()), float(ys.max())))
        if not crops:
            continue
        texts.extend(engine.recognize(crops))
        geoms.extend(batch_geoms)
        items = [(t, g[0], g[1], g[2], g[3]) for (t, _), g in zip(texts, geoms)]
        candidates = parse_boxes(items)
        if any(sel_score(c, "") >= STOP_SCORE for c in candidates):
            break
    t_rec = time.perf_counter()
    full_text = " ".join(t for t, _ in texts)
    winner = select(candidates, full_text, cfg.impute_missing)
    t_end = time.perf_counter()

    row = to_row(winner, image_id)
    row["_diag"] = {
        "n_filtered": len(kept),
        "n_recognized": len(texts),
        "texts": [t for t, _ in texts],
        "candidates": [{"text": c.text, "final_date": c.final_date} for c in candidates],
        "ms": {
            "load": (t_load - t0) * 1000,
            "detect": (t_det - t_load) * 1000,
            "recognize": (t_rec - t_det) * 1000,
            "parse": (t_end - t_rec) * 1000,
            "total": (t_end - t0) * 1000,
        },
    }
    return row


def write_rows(path, rows) -> None:
    """제출 스키마로 저장한다. 인덱스 컬럼은 생기지 않는다."""
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def blank_rows(paths) -> list[dict]:
    return [{"image_id": Path(p).stem, "year": "NONE", "month": "NONE",
             "day": "NONE", "final_date": "NONE"} for p in paths]


def run(input_dir, output_path, cfg: Config | None = None, engine=None,
        deadline: float | None = None, progress_every: int = 200,
        collect_diag: bool = False) -> dict:
    """디렉터리 전체를 순차 처리하여 CSV로 저장한다."""
    cfg = cfg or Config()
    paths = iter_images(input_dir)
    rows = blank_rows(paths)
    write_rows(output_path, rows)

    if engine is None:
        from .engine import Engine
        engine = Engine(det_side=cfg.det_side, threads=cfg.threads,
                        box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio,
                        nanodet_onnx=cfg.nanodet_onnx,
                        nanodet_score_thr=cfg.nanodet_score_thr,
                        nanodet_expand=cfg.nanodet_expand,
                        nanodet_nms_iou=cfg.nanodet_nms_iou)

    diags, degraded, skipped = [], 0, 0
    started = time.time()
    for i, path in enumerate(paths):
        remaining = len(paths) - i
        top_k = cfg.top_k
        max_k = cfg.max_k
        if deadline is not None:
            budget = (deadline - time.time()) / max(remaining, 1)
            if budget <= 0:
                skipped = remaining
                break
            if budget < cfg.per_image_budget * 0.6:
                top_k, max_k, degraded = 1, 1, degraded + 1

        try:
            row = process_image(engine, path, cfg, top_k=top_k, max_k=max_k)
            diag = row.pop("_diag")
            if collect_diag:
                diags.append({"image_id": row["image_id"], **diag})
            rows[i] = row
        except Exception:
            pass

        if (i + 1) % cfg.flush_every == 0:
            write_rows(output_path, rows)
        if progress_every and (i + 1) % progress_every == 0:
            rate = (time.time() - started) / (i + 1)
            print(f"  {i + 1}/{len(paths)}  {rate * 1000:.0f} ms/img", flush=True)

    write_rows(output_path, rows)
    elapsed = time.time() - started
    return {
        "n": len(paths),
        "elapsed": elapsed,
        "ms_per_image": elapsed / len(paths) * 1000 if paths else 0.0,
        "coverage": sum(r["final_date"] != "NONE" for r in rows) / len(paths) if paths else 0.0,
        "degraded": degraded,
        "skipped": skipped,
        "rows": rows,
        "diagnostics": diags,
    }
