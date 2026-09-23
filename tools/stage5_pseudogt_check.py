"""B5 판정 3단계: Kaggle val pseudo-GT 기준 end-to-end 정답률.

Kaggle val 에는 날짜 텍스트 GT 가 없다. 그래서 오라클(GT 박스 + 고정 인식기 PP-OCRv6)이
읽은 값을 pseudo-GT 로 삼는다(results/stage6_oracle_kaggle126.json). IoU recall 은
end-to-end 성능을 반영하지 못한다는 것이 실측으로 확인됐으므로(버킷3 12건 중 7건이
IoU 는 낮아도 정답), 판정은 이 지표로 한다.

사용: python tools/stage5_pseudogt_check.py <onnx> <score_thr>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from itda_ocr.engine import Engine                      # noqa: E402
from itda_ocr.pipeline import Config, process_image     # noqa: E402

ONNX, THR = sys.argv[1], float(sys.argv[2])
D = ROOT / "data" / "Kaggle"
PSEUDO = json.load(open(ROOT / "results/stage6_oracle_kaggle126.json"))
TARGET8 = ["003399", "003451", "003453", "003358", "003527",   # 검출 실패 5
           "003401", "003418", "003477"]                        # 박스 잘림 3
B1_BASE = json.load(open(ROOT / "results/stage6_kaggle_pseudogt.json"))

cfg = Config(nanodet_onnx=ONNX, nanodet_score_thr=THR, nanodet_expand=0.10, crop_tta=0.20,
             det_fallback=True, rec_onnx=str(ROOT / "weights/ppocrv6_rec_small_date.onnx"),
             rec_fallback=False)
eng = Engine(det_side=cfg.det_side, threads=4, box_thresh=cfg.box_thresh,
             unclip_ratio=cfg.unclip_ratio, nanodet_onnx=cfg.nanodet_onnx,
             nanodet_score_thr=cfg.nanodet_score_thr, nanodet_expand=cfg.nanodet_expand,
             rec_onnx=cfg.rec_onnx, rec_fallback=cfg.rec_fallback)

res, ok = {}, 0
for iid, o in PSEUDO.items():
    pg = o.get("oracle_date")
    if not pg or pg == "NONE":
        continue
    row = process_image(eng, D / "images/val" / f"{iid}.jpg", cfg, image_id=iid)
    row.pop("_diag")
    pred = row.get("final_date", "NONE")
    hit = pred == pg
    ok += hit
    res[iid] = {"pred": pred, "pseudo_gt": pg, "ok": hit}

n = len(res)
base_ok = sum(1 for i in res if B1_BASE.get(i, {}).get("ok"))
print(f"=== Kaggle val pseudo-GT end-to-end (onnx={Path(ONNX).name}, thr={THR}) ===")
print(f"정답 {ok}/{n} ({ok / n * 100:.1f}%)   B1 기준선 {base_ok}/{n} ({base_ok / n * 100:.1f}%)"
      f"   차이 {ok - base_ok:+d}건")

rescued = [i for i in TARGET8 if i in res and res[i]["ok"] and not B1_BASE.get(i, {}).get("ok")]
broken = [i for i in res if res[i]["ok"] is False and B1_BASE.get(i, {}).get("ok")]
print(f"\n목표 8건 중 살아남: {len(rescued)}건 {rescued}")
print(f"기존 정답이 깨짐  : {len(broken)}건 {broken}")
print(f"\n판정: {'개선' if ok > base_ok else '동률' if ok == base_ok else '퇴보'}"
      f" (성공 기준 = B1 대비 정답 수 증가)")
json.dump({"n": n, "ok": ok, "b1_ok": base_ok, "rescued": rescued, "broken": broken,
           "per_image": res}, open(ROOT / "results/stage5_pseudogt_result.json", "w"),
          ensure_ascii=False, indent=1)
