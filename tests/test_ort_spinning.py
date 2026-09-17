"""모든 ONNX 세션(NanoDet + RapidOCR det/cls/rec)에서 intra-op spinning 이 꺼졌는지 확인한다.

하나라도 켜져 있으면 검출·인식 교대 시 스레드 경합으로 전체가 약 2배 느려진다
(docs/SPEED_ANALYSIS.md §2). RapidOCR 은 옵션을 내부에서 만들기 때문에
업그레이드 시 감싸기가 조용히 무력화될 수 있어 이 체크를 남긴다.
"""

from pathlib import Path

from itda_ocr.engine import Engine

NANODET = Path(__file__).resolve().parents[1] / "weights" / "date_detector_ema.onnx"


def sessions(engine):
    yield "nanodet", engine._nanodet._sess
    for name, obj in (("det", engine._det), ("cls", engine._cls), ("rec", engine._rec)):
        ort_session = getattr(obj, "session", None) or obj.infer   # rec 는 session, det/cls 는 infer
        yield name, ort_session.session


def test_all_sessions_disable_spinning():
    rec = NANODET.with_name("ppocrv6_rec_small.onnx")
    engine = Engine(nanodet_onnx=str(NANODET), threads=4,
                    rec_onnx=str(rec) if rec.exists() else None)
    for name, sess in sessions(engine):
        try:
            value = sess.get_session_options().get_session_config_entry(
                "session.intra_op.allow_spinning")
        except RuntimeError:                    # 키 미설정 = ORT 기본값(spinning 켜짐)
            value = None
        assert value == "0", f"{name} 세션의 spinning 이 꺼지지 않았습니다 ({value!r})"


if __name__ == "__main__":
    test_all_sessions_disable_spinning()
    print("OK  4 sessions allow_spinning=0")
