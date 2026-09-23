"""B4 검증 3단계: Kaggle val 신규(Group E, 126장)에서 IoU@0.3/0.5/0.7 + 포함률 + 대상 정확성.
stage1_target.py 와 완전히 같은 방법론(같은 GT 정답 선택 로직) — 검출기만 새 onnx로 교체.
사용: python stage4_kaggle_check.py <onnx> <score_thr>
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itda_ocr.engine import Engine
from itda_ocr.pipeline import Config, load_image

ROOT = Path(__file__).resolve().parents[1]
ONNX, THR = sys.argv[1], float(sys.argv[2])
D = ROOT / "data" / "Kaggle"
E = [l.strip() for l in open(ROOT / "results/expiry_region_clean_new_ids.txt") if l.strip()]
K = 9

def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    i = (ix1 - ix0) * (iy1 - iy0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0

def inclusion(pred, gt):
    ix0, iy0, ix1, iy1 = max(pred[0], gt[0]), max(pred[1], gt[1]), min(pred[2], gt[2]), min(pred[3], gt[3])
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    ga = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return ((ix1 - ix0) * (iy1 - iy0)) / ga if ga > 0 else 0.0

def yolo_all(path):
    out = {c: [] for c in range(4)}
    for line in Path(path).read_text().split("\n"):
        t = line.split()
        if len(t) != 5: continue
        c = int(t[0]); cx, cy, w, h = map(float, t[1:])
        out[c].append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out

cfg = Config(nanodet_onnx=ONNX, nanodet_score_thr=THR)
eng = Engine(det_side=cfg.det_side, threads=4, box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio,
             nanodet_onnx=cfg.nanodet_onnx, nanodet_score_thr=cfg.nanodet_score_thr,
             nanodet_expand=cfg.nanodet_expand, nanodet_nms_iou=cfg.nanodet_nms_iou)
from itda_ocr.engine import filter_boxes

DATE_RE = re.compile(r"(\d{2,4})\D?(\d{1,2})\D?(\d{1,2})")

def parse_ymd(text):
    t = re.sub(r"[^0-9]", "", text)
    for L in (8, 6):
        if len(t) >= L:
            s = t[:L]
            if L == 8: y, m, dd = int(s[:4]), int(s[4:6]), int(s[6:8])
            else: y, m, dd = 2000 + int(s[:2]), int(s[2:4]), int(s[4:6])
            if 1 <= m <= 12 and 1 <= dd <= 31 and 2015 <= y <= 2035: return (y, m, dd)
    return None

def pick_correct_date_box(img, boxes_norm, h, w):
    if len(boxes_norm) == 1: return 0
    parsed = []
    for b in boxes_norm:
        x0, y0, x1, y1 = int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)
        pad_x, pad_y = int((x1-x0)*0.15), int((y1-y0)*0.4)
        crop = img[max(0,y0-pad_y):y1+pad_y, max(0,x0-pad_x):x1+pad_x]
        text = eng.recognize([crop], use_cls=False)[0][0] if crop.size else ""
        parsed.append(parse_ymd(text))
    valid = [(i, p) for i, p in enumerate(parsed) if p]
    if len(valid) < len(boxes_norm): return len(boxes_norm) - 1  # OCR 실패 -> 폴백(마지막 박스)
    return max(valid, key=lambda x: x[1])[0]

records = {}
for iid in E:
    p = D / "images/val" / f"{iid}.jpg"
    img = load_image(p, 1280); h, w = img.shape[:2]
    g = yolo_all(D / "labels/val" / f"{iid}.txt")
    dates = g[0]
    if not dates: continue
    ci = pick_correct_date_box(img, dates, h, w)
    correct = dates[ci]
    decoys = [dates[i] for i in range(len(dates)) if i != ci] + g[2]
    det_img = load_image(p, 720)
    kept = filter_boxes(eng._nanodet.detect(det_img), det_img.shape, rerank=False)
    pool = []
    for _, _, q in kept[:K]:
        pool.append([float(q[:,0].min())/det_img.shape[1], float(q[:,1].min())/det_img.shape[0],
                     float(q[:,0].max())/det_img.shape[1], float(q[:,1].max())/det_img.shape[0]])
    iou_correct = max((iou(pd, correct) for pd in pool), default=0.0)
    incl_correct = max((inclusion(pd, correct) for pd in pool), default=0.0)
    incl_decoy = max((inclusion(pd, dz) for pd in pool for dz in decoys), default=0.0) if decoys else 0.0
    records[iid] = {"iou_correct": iou_correct, "incl_correct": incl_correct, "incl_decoy": incl_decoy}

n = len(records)
print(f"Group E {len(E)}장 중 gt_date 있는 이미지 {n}장 (onnx={ONNX}, thr={THR})")
for t in (0.3, 0.5, 0.7):
    print(f"  recall@{t}: {sum(r['iou_correct']>=t for r in records.values())/n*100:.1f}%")

TH = 0.7
imperfect = {i: r for i, r in records.items() if r["iou_correct"] < TH}
b2 = {i: r for i, r in imperfect.items() if r["incl_decoy"] >= TH and r["incl_correct"] < TH}
b1 = {i: r for i, r in imperfect.items() if r["incl_correct"] >= TH and i not in b2}
b3 = {i: r for i, r in imperfect.items() if i not in b1 and i not in b2}
print(f"\niou<0.7 {len(imperfect)}건 중: 마진(문제없음) {len(b1)} / 대상오인식 {len(b2)} {sorted(b2)} / 완전실패 {len(b3)} {sorted(b3)}")
json.dump({"records": records, "buckets": {"b1": sorted(b1), "b2": sorted(b2), "b3": sorted(b3)}, "onnx": ONNX, "thr": THR},
          open(ROOT / "results/stage4_kaggle126.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved results/stage4_kaggle126.json")
