"""이미지 로딩, 장당 파이프라인 및 배치 처리 드라이버 모듈."""

from __future__ import annotations

import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
    crop_tta: float = 0.0                  # >0 이면 각 박스를 (원래 여백, +이 비율) 두 번 읽는다
    det_fallback: bool = False             # 검출 박스가 0개인 이미지만 범용 DB 검출로 재시도
    prefetch: bool = False                 # 다음 이미지 디코딩을 보조 스레드로 선행 (run() 전용)
    second_read: bool = False              # 판독이 의심스러울 때만 전처리를 바꿔 다시 읽는다
    rec_onnx: str | None = None
    rec_fallback: bool | str = False
    fallback_max_k: int | None = None      # 폴백이 읽을 크롭 수 상한 (None = max_k 와 같음)
    fallback_budget: float | None = None   # 폴백 배치 경계에서 검사하는 시간 상한(초). None = 무제한


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

    kept = engine.detect_and_filter(img, db_fallback=cfg.det_fallback)
    t_det = time.perf_counter()

    batch = top_k if top_k is not None else cfg.top_k
    limit = max_k if max_k is not None else cfg.max_k

    texts, candidates = _read(engine, img, kept, batch, limit, tta=cfg.crop_tta)
    t_rec = time.perf_counter()
    winner = select(candidates, " ".join(t for t, _ in texts), cfg.impute_missing)
    t_sel = time.perf_counter()

    # 완전한 날짜를 못 냈으면 같은 박스를 폴백 인식기로 다시 읽고, 거기서 완전한 날짜가
    # 나올 때만 바꾼다. 못 내면 원래 결과(연·월 부분 추출 포함)를 그대로 둔다.
    used_fallback = False
    if (winner is None or not winner.complete) and engine.has_fallback and kept:
        fb_texts, fb_cands = _read(engine, img, kept, batch,
                                   min(limit, cfg.fallback_max_k or limit),
                                   fallback=True, budget=cfg.fallback_budget,
                                   tta=cfg.crop_tta)
        fb = select(fb_cands, " ".join(t for t, _ in fb_texts), cfg.impute_missing)
        # 주 인식기가 연·월을 냈다면 그것과 어긋나는 폴백 날짜는 버린다. 인쇄에 연·월만 있어
        # GT 도 연·월뿐인 이미지(통합셋 48장)에서 맞은 부분 답을 덮어쓰는 것을 막는다.
        if fb is not None and winner is not None and winner.year and winner.month:
            if (fb.year, fb.month) != (winner.year, winner.month):
                fb = None
        if fb is not None and fb.complete:
            texts, candidates, winner, used_fallback = fb_texts, fb_cands, fb, True
    if cfg.second_read and kept and _suspect(winner, candidates):
        sr_texts, sr_cands = _read(engine, img, kept, batch, limit, tta=0.0,
                                   modes=("contrast", "invert"))
        sr = select(sr_cands, " ".join(t for t, _ in sr_texts), cfg.impute_missing)
        # 1차가 연·월을 냈다면 그것과 어긋나는 2차 날짜는 버린다 (폴백과 같은 가드).
        if sr is not None and winner is not None and winner.year and winner.month:
            if (sr.year, sr.month) != (winner.year, winner.month):
                sr = None
        if sr is not None and sr.complete:
            texts, candidates, winner, used_fallback = sr_texts, sr_cands, sr, True
    t_end = time.perf_counter()

    row = to_row(winner, image_id)
    row["_diag"] = {
        "n_filtered": len(kept),
        "n_recognized": len(texts),
        "texts": [t for t, _ in texts],
        "candidates": [{"text": c.text, "final_date": c.final_date} for c in candidates],
        "fallback": used_fallback,
        "ms": {
            "load": (t_load - t0) * 1000,
            "detect": (t_det - t_load) * 1000,
            "recognize": (t_rec - t_det) * 1000,
            "parse": (t_sel - t_rec) * 1000,
            "fallback": (t_end - t_sel) * 1000,
            "total": (t_end - t0) * 1000,
        },
    }
    return row


#: 끝이 한 자리 수로 끊긴 판독의 서명. `2025.12.2`(실제 12-24) 처럼 형식은 완벽한 오답을 만든다.
_TRUNC_TAIL = re.compile(r"\d{2}\s*[.\-/]\s*\d\s*$")


def _suspect(winner, candidates) -> bool:
    """2차 판독이 필요한가 — 발동 조건을 '실패 여부'가 아니라 '신뢰도'로 둔다.

    오답 감점의 49% 가 **형식이 완벽한 오답**이라 기존 폴백 조건(완전한 날짜 실패)에 걸리지 않는다.
    세 신호를 본다: ① 완전한 날짜가 없다 ② 판독이 한 자리 수로 끊겼다 ③ 같은 연·월인데 일이 갈린다.
    """
    if winner is None or not winner.complete:
        return True
    if _TRUNC_TAIL.search(winner.text):
        return True
    days = {c.day for c in candidates
            if c.complete and c.year == winner.year and c.month == winner.month}
    return len(days) > 1


def _read(engine, img, kept, batch: int, limit: int, fallback: bool = False,
          budget: float | None = None, tta: float = 0.0, modes=(None,)):
    """상위 박스부터 배치로 잘라 읽고 날짜 후보를 모은다. 확실한 후보가 나오면 멈춘다.

    ``budget`` 이 있으면 배치 경계에서 경과 시간을 보고 넘으면 중단한다 — 느린 폴백
    인식기가 어려운 이미지 하나에 시간을 다 쓰는 것을 막는다.

    ``modes`` 는 크롭마다 적용할 전처리 목록이다 (``None`` = 원본). ``expands × modes`` 조합만큼
    크롭을 만든다.

    ``tta`` > 0 이면 박스마다 크롭을 **두 개** (원래 여백, 추가 ``tta`` 비율) 만들어 같은 배치로
    읽고 **같은 기하 좌표**를 붙인다. 판독이 여백에 민감해서(PP-OCRv6 자체 지표 crop margin
    robustness 75.32%) 여백 하나로는 끝 글자 절단을 못 막는다 — 값을 고르는 대신 둘 다 읽고
    기존 선별기가 고르게 한다. 같은 좌표를 주면 ``_filter_overlapping_boxes`` 가 줄 병합에서
    긴 판독을 남기고, ``rank()`` 동점 규칙(늦은 날짜 우선)이 잘린 판독을 떨어뜨린다.
    근거와 1,473장 A/B 수치는 README §3 의 "[3] 인식 — 크롭 TTA".
    """
    texts, geoms, candidates = [], [], []
    expands = (0.0, tta) if tta else (0.0,)
    variants = [(e, m) for e in expands for m in modes]
    t_start = time.perf_counter()
    for start in range(0, min(len(kept), limit), batch):
        if budget is not None and start and time.perf_counter() - t_start > budget:
            break
        crops, batch_geoms = [], []
        for _, _, box in kept[start:start + batch]:
            xs, ys = box[:, 0], box[:, 1]
            geom = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            for expand, mode in variants:
                patch = engine.crop(img, box, expand)
                if patch.size:
                    crops.append(engine.enhance(patch, mode) if mode else patch)
                    batch_geoms.append(geom)
        if not crops:
            continue
        texts.extend(engine.recognize(crops, fallback=fallback))
        geoms.extend(batch_geoms)
        items = [(t, g[0], g[1], g[2], g[3]) for (t, _), g in zip(texts, geoms)]
        candidates = parse_boxes(items)
        if any(sel_score(c, "") >= STOP_SCORE for c in candidates):
            break
    return texts, candidates


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
                        nanodet_nms_iou=cfg.nanodet_nms_iou,
                        rec_onnx=cfg.rec_onnx, rec_fallback=cfg.rec_fallback)

    diags, degraded, skipped = [], 0, 0
    started = time.time()
    # 다음 이미지 디코딩을 보조 스레드에서 선행한다. 디코딩은 코어를 0.6개만 쓰는데 그 구간에
    # 3.4개가 놀고, 메인 스레드는 시간의 절반 이상을 ORT Run 에서 보내며 GIL 을 놓는다.
    # 실측 -11.4%, 예측 차이 0장. 실패하면 조용히 메인 스레드 디코딩으로 되돌아간다.
    pre = ThreadPoolExecutor(max_workers=1) if cfg.prefetch and len(paths) > 1 else None
    fut = pre.submit(load_image, paths[0], cfg.draft_to) if pre is not None else None
    try:
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

            img = None
            if pre is not None:
                if fut is not None:
                    try:
                        img = fut.result()
                    except Exception:            # noqa: BLE001  선행 실패 → 메인에서 다시 읽는다
                        img = None
                fut = (pre.submit(load_image, paths[i + 1], cfg.draft_to)
                       if i + 1 < len(paths) else None)

            try:
                row = process_image(engine, path if img is None else img, cfg,
                                    top_k=top_k, max_k=max_k, image_id=path.stem)
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
    finally:
        if pre is not None:
            pre.shutdown(wait=False, cancel_futures=True)

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
