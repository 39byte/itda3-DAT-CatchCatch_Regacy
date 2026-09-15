#!/usr/bin/env bash
# [ITDA 3rd] 소비기한 추출 - 가중치 확인 스크립트
# 채점 환경: 인터넷 차단 전 운영진이 1회 실행
set -e

echo "=== [ITDA] Checking Model Weights ==="

# 1. NanoDet-Plus-m 날짜 전용 검출기 가중치 (5.6MB)
#    경량 모델이므로 저장소 weights/date_detector_ema.onnx 에 직접 포함되어 있습니다.
if [ -f "weights/date_detector_ema.onnx" ]; then
    echo "[OK] weights/date_detector_ema.onnx found."
else
    echo "[WARNING] weights/date_detector_ema.onnx not found. Pipeline will fall back to RapidOCR default detector."
fi

# 2. RapidOCR 가중치 (det/cls/rec ONNX 약 16MB)
#    rapidocr-onnxruntime==1.4.4 wheel 내부에 포함되어 pip install 만으로 준비됩니다.
echo "[OK] RapidOCR default models are bundled within the python wheel package."

# 3. cv2 배포판 정규화 (GUI 빌드 -> headless)
#    rapidocr-onnxruntime 이 opencv-python 을 하드 의존으로 요구해서, requirements 에
#    무엇을 적든 opencv-python 과 opencv-python-headless 가 함께 설치된다. 둘은 같은
#    site-packages/cv2/cv2.abi3.so 를 쓰는데 pip 의 설치 순서는 레벨 내 역알파벳이라
#    항상 headless -> opencv-python 순이고, 결국 GUI 빌드가 파일을 덮어쓴다.
#    GUI 빌드의 cv2.abi3.so 는 번들 Qt5(libQt5Core/Gui/Test/Widgets)에 DT_NEEDED 로
#    걸려 있고 그 Qt5 가 시스템 libGL.so.1 을 찾는데, 이 라이브러리는 wheel 에
#    번들되지 않는다. GUI 라이브러리가 없는 서버 이미지에서는 import cv2 자체가
#    ImportError: libGL.so.1 로 죽는다. headless 를 마지막에 한 번 더 덮어써서 뒤집는다.
#    (인터넷이 필요하므로 오프라인 차단 전에 이 스크립트가 실행되어야 한다)
echo "--- Normalizing cv2 distribution to the headless build ---"
PY="$(command -v python || command -v python3 || echo "")"
if [ -z "$PY" ]; then
    echo "[WARNING] python not found on PATH; skipped cv2 normalization."
else
    "$PY" -m pip install --no-deps --force-reinstall --quiet \
        "opencv-python-headless==4.9.0.80" \
        || echo "[WARNING] headless reinstall failed (offline?). Continuing."
    "$PY" - <<'PYEOF' || echo "[WARNING] cv2 verification failed — is the project venv active?"
import cv2
gui = next((l.split(":", 1)[1].strip() for l in cv2.getBuildInformation().splitlines()
            if l.strip().startswith("GUI:")), "?")
if gui.upper() == "NONE":
    print(f"[OK] active cv2 {cv2.__version__} is the headless build (GUI={gui}).")
else:
    print(f"[WARNING] active cv2 {cv2.__version__} is a GUI build (GUI={gui}). "
          "It requires system libGL.so.1 and will fail to import on a headless host.")
PYEOF
fi

echo "=== Model weight verification complete. ==="
