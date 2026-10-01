"""출력 경로의 상위 폴더가 없어도 run() 이 raise 하지 않는다 (불변식 1)."""

import csv

from itda_ocr.pipeline import run


def test_run_creates_missing_output_dir(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "broken.jpg").write_bytes(b"")
    out = tmp_path / "no" / "such" / "dir" / "out.csv"

    res = run(tmp_path / "in", out, engine=object())   # 디코딩 실패라 엔진은 안 불린다

    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert res["n"] == 1
    assert rows == [{"image_id": "broken", "year": "NONE", "month": "NONE",
                     "day": "NONE", "final_date": "NONE"}]
