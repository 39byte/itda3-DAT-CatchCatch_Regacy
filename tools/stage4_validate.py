"""검출기 검증 오케스트레이터 (B4/B5 공용).

evaluation 665 게이트 -> Kaggle val IoU 참고지표 -> Kaggle val pseudo-GT 성공판정.

⚠️ **채점은 `eval.run` 서브프로세스가 아니라 `process_image` 를 직접 부른다.**
   `eval.run` CLI 는 `--rec-onnx`/`--crop-tta` 를 넘기지 않으면 Config 기본값(번들
   PP-OCRv4, crop_tta=0.0)으로 돌아가고, `det_fallback` 은 아예 CLI 플래그가 없다.
   B5 최초 판정이 이 누락 때문에 PP-OCRv6 가 아닌 인식기로 측정돼 무효가 됐다.
   프로덕션과 같은 Config 를 한 곳에서 만들어 직접 넘기면 이런 누락이 원천적으로 없다.

사용:
    python tools/stage4_validate.py <weights/xxx.onnx>          # threshold 스캔부터
    python tools/stage4_validate.py <weights/xxx.onnx> 0.10     # threshold 고정(스캔 생략)
"""
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from itda_ocr.engine import Engine                      # noqa: E402
from itda_ocr.pipeline import Config, iter_images, process_image   # noqa: E402
from eval.score import classify, score_row              # noqa: E402

ONNX = Path(sys.argv[1]).resolve()
assert ONNX.exists(), f"onnx 없음: {ONNX}"
FIXED_THR = float(sys.argv[2]) if len(sys.argv) > 2 else None
PY = ROOT / ".venv/bin/python"
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
B1_ONNX = ROOT / "weights/date_detector_ema.onnx"

#: 프로덕션 파이프라인 설정 (predict.ipynb CFG 와 동일). 인식기는 PP-OCRv6 고정.
PROD = dict(nanodet_expand=0.10, crop_tta=0.20, det_fallback=True,
            rec_onnx=str(ROOT / "weights/ppocrv6_rec_small_date.onnx"), rec_fallback=False)

_ENGINES = {}


def get_engine(onnx, thr):
    key = (str(onnx), thr)
    if key not in _ENGINES:
        cfg = Config(nanodet_onnx=str(onnx), nanodet_score_thr=thr, **PROD)
        _ENGINES[key] = (Engine(det_side=cfg.det_side, threads=cfg.threads,
                                box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio,
                                nanodet_onnx=cfg.nanodet_onnx,
                                nanodet_score_thr=cfg.nanodet_score_thr,
                                nanodet_expand=cfg.nanodet_expand,
                                rec_onnx=cfg.rec_onnx, rec_fallback=cfg.rec_fallback), cfg)
    return _ENGINES[key]


def run_eval(images, gt_path, thr, out_dir, onnx=ONNX):
    """process_image 로 직접 채점. 반환: 평균 점수(0~50). 부수적으로 버킷·박스수 저장."""
    eng, cfg = get_engine(onnx, thr)
    gt = {r["image_id"]: r for r in csv.DictReader(open(gt_path, encoding="utf-8-sig"))}
    per, buckets, nboxes = {}, Counter(), []
    for p in iter_images(images):
        iid = p.stem
        if iid not in gt:
            continue
        row = process_image(eng, p, cfg, image_id=iid)
        diag = row.pop("_diag")
        per[iid] = {"score": score_row(row, gt[iid]),
                    "bucket": classify(row, gt[iid], diag["candidates"]),
                    "pred": row.get("final_date"), "gt": gt[iid]["final_date"],
                    "n_filtered": diag["n_filtered"]}
        buckets[per[iid]["bucket"]] += 1
        nboxes.append(diag["n_filtered"])
    n = len(per)
    score = sum(v["score"] for v in per.values()) / n
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image.json").write_text(
        json.dumps({"score": score, "n": n, "buckets": dict(buckets), "per_image": per},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    avg_box = sum(nboxes) / len(nboxes) if nboxes else 0
    print(f"    n={n}  점수 {score:.3f}/50  버킷 {dict(buckets)}  필터통과 박스 평균 {avg_box:.1f}개")
    return score


if FIXED_THR is not None:
    chosen = FIXED_THR
    print(f"=== 1단계 생략: threshold 를 {chosen} 로 고정 (사전 등록값) ===")
else:
    print("=== 1단계: val_split(85) threshold 스캔 (results/stage4_threshold_rule.md) ===")
    OUT = ROOT / "results/stage4_val85_scan"
    scores = {}
    for thr in GRID:
        print(f"  thr={thr}")
        scores[thr] = run_eval("/tmp/val_split_images", ROOT / "labels/expdate_valsplit/gt_dates.csv",
                               thr, OUT / f"thr{thr}")
    best = max(scores.values())
    tied = sorted(t for t, s in scores.items() if best - s <= 0.3)   # 85장 기준 ~1건
    chosen = tied[0]
    print(f"\n최고점 {best:.2f}, 동률 구간(0.3점 이내) {tied} -> 채택 threshold = {chosen}")

EVAL_IMG, EVAL_GT = ROOT / "data/ExpDate/evaluation/images", ROOT / "labels/expdate/gt_dates.csv"
print("\n=== 2단계 사전: B1 기준선을 같은 코드·같은 설정으로 동시 측정 ===")
b1_score = run_eval(EVAL_IMG, EVAL_GT, 0.05, ROOT / "results/stage4_eval665_b1_baseline", onnx=B1_ONNX)
print(f"  B1 기준선(현재 실측) = {b1_score!r}")

print(f"\n=== 2단계: evaluation 665장 (thr={chosen} 단 1회) ===")
eval_score = run_eval(EVAL_IMG, EVAL_GT, chosen, ROOT / "results/stage4_eval665")
print(f"\nB1 기준선 {b1_score:.3f} vs 후보({ONNX.name}) {eval_score:.3f} (thr={chosen})")
if eval_score < b1_score:
    print("\n### 판정: 퇴보 -> 규칙에 따라 무조건 기각. Kaggle 검증은 진행하지 않음. ###")
    sys.exit(1)
print("\n### 게이트 통과(퇴보 없음) -> Kaggle val 검증으로 진행합니다. ###")
(ROOT / "results/stage4_chosen_threshold.txt").write_text(str(chosen))

print("\n=== 3단계: Kaggle val 신규 126장 (IoU + 포함률 + 대상 정확성) — 참고 지표 ===")
r = subprocess.run([str(PY), str(Path(__file__).parent / "stage4_kaggle_check.py"),
                    str(ONNX), str(chosen)], cwd=ROOT)
if r.returncode != 0:
    raise SystemExit("stage4_kaggle_check.py 실패")

# 4단계: 성공 판정용. IoU recall 은 end-to-end 를 반영하지 못하는 것이 실측으로
# 확인돼(버킷3 12건 중 7건이 IoU 는 낮아도 정답) 판정은 pseudo-GT end-to-end 로 한다.
print("\n=== 4단계: Kaggle val pseudo-GT end-to-end (성공 판정 기준) ===")
r = subprocess.run([str(PY), str(Path(__file__).parent / "stage5_pseudogt_check.py"),
                    str(ONNX), str(chosen)], cwd=ROOT)
if r.returncode != 0:
    raise SystemExit("stage5_pseudogt_check.py 실패")
