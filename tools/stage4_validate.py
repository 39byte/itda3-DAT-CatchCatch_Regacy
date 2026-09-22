"""B4(date_detector_aug) 검증 오케스트레이터: val_split(85) 임계값 스캔(사전등록 규칙)
-> evaluation 665(1회, 규정값으로만) -> Kaggle val 신규 126(IoU+포함률+대상정확성).
사용: python stage4_validate.py <weights/date_detector_aug.onnx>
"""
import json, subprocess, sys
from pathlib import Path
ROOT = Path.home() / "Desktop" / "ITDA_CatchCatch"
ONNX = Path(sys.argv[1]).resolve()
assert ONNX.exists(), f"onnx 없음: {ONNX}"
PY = ROOT / ".venv/bin/python"
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
B1_ONNX = ROOT / "weights/date_detector_ema.onnx"
OUT = ROOT / "results/stage4_val85_scan"
OUT.mkdir(parents=True, exist_ok=True)


def run_eval(images, gt, thr, out_dir, onnx=ONNX):
    cmd = [str(PY), "-m", "eval.run", "--images", str(images), "--gt", str(gt),
           "--nanodet", str(onnx), "--nanodet-score-thr", str(thr), "--out", str(out_dir)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0 or not (out_dir / "report.json").exists():
        print(r.stderr[-3000:]); raise SystemExit(f"eval.run 실패 (thr={thr})")
    return json.loads((out_dir / "report.json").read_text())["score"]


print("=== 1단계: val_split(85) threshold 스캔 (사전 등록 규칙, results/stage4_threshold_rule.md) ===")
scores = {}
for thr in GRID:
    scores[thr] = run_eval("/tmp/val_split_images", ROOT / "labels/expdate_valsplit/gt_dates.csv", thr, OUT / f"thr{thr}")
    print(f"  thr={thr}: {scores[thr]:.2f}/50")

best = max(scores.values())
tied = sorted(t for t, s in scores.items() if best - s <= 0.3)  # 0.3점 이내 = 동률 구간(85장 기준 ~1건)
chosen = tied[0]  # 동률 구간에서 가장 작은 threshold
print(f"\n최고점 {best:.2f}, 동률 구간(0.3점 이내) {tied} -> 채택 threshold = {chosen}")

EVAL_IMG, EVAL_GT = ROOT / "data/ExpDate/evaluation/images", ROOT / "labels/expdate/gt_dates.csv"
print("\n=== 2단계 사전: B1 기준선을 같은 코드로 동시 측정 (하드코딩된 반올림값과 비교하는 오류 방지) ===")
b1_score = run_eval(EVAL_IMG, EVAL_GT, 0.05, ROOT / "results/stage4_eval665_b1_baseline", onnx=B1_ONNX)
print(f"  B1 기준선(현재 실측) = {b1_score!r}")

print(f"\n=== 2단계: evaluation 665장 (thr={chosen} 단 1회) ===")
eval_out = ROOT / "results/stage4_eval665"
eval_score = run_eval(EVAL_IMG, EVAL_GT, chosen, eval_out)
print(f"\nB1 기준선 {b1_score:.2f} vs B4 {eval_score:.2f} (chosen thr={chosen})")
if eval_score < b1_score:
    print("\n### 판정: 퇴보 -> 규칙에 따라 무조건 기각. Kaggle 검증은 진행하지 않음. ###")
    sys.exit(1)
print("\n### 판정: 퇴보 없음 -> 채택. Kaggle val 신규 126장 검증으로 이어서 진행합니다. ###")
(ROOT / "results/stage4_chosen_threshold.txt").write_text(str(chosen))
print(f"\nchosen_threshold={chosen} (results/stage4_chosen_threshold.txt 에 저장)")

print("\n=== 3단계: Kaggle val 신규 126장 (IoU + 포함률 + 대상 정확성) ===")
r = subprocess.run([str(PY), str(Path(__file__).parent / "stage4_kaggle_check.py"), str(ONNX), str(chosen)],
                    cwd=ROOT)
if r.returncode != 0:
    raise SystemExit("stage4_kaggle_check.py 실패")
