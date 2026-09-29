import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "kist_data/evaluation/images/test_00001.jpg"   # GT 2021-08-03

pytestmark = pytest.mark.skipif(not (ROOT / "weights/date_detector_ema.onnx").exists() or not SAMPLE.exists(),
                                reason="가중치나 샘플 이미지가 없음")


@pytest.fixture(scope="module")
def url():
    from serve.ocr_server import make_engine, make_handler
    srv = HTTPServer(("127.0.0.1", 0), make_handler(make_engine()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def post(url, data):
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST")) as r:
        return json.loads(r.read())


def test_reads_sample_and_never_raises(url):
    assert json.loads(urllib.request.urlopen(url + "/health").read()) == {"ok": True}
    assert post(url + "/ocr", SAMPLE.read_bytes())["final_date"] == "2021-08-03"
    bad = post(url + "/ocr", b"not an image")
    assert bad["final_date"] == "NONE" and bad["year"] == "NONE"


def test_pad_closeup_centers_crop():
    from serve.ocr_server import pad_closeup
    img = np.full((10, 20, 3), 7, np.uint8)
    out = pad_closeup(img)
    assert out.shape == (45, 60, 3)
    assert (out[17:27, 20:40] == 7).all()
