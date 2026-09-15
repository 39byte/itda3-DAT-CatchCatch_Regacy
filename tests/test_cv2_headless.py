"""활성 cv2 가 headless 빌드인지 증명한다 — 채점 0점을 막는 유일한 체크.

opencv-python(GUI) 빌드가 이기면 채점 서버(GUI 라이브러리 없는 Ubuntu)에서
``import cv2`` 가 ``libGL.so.1`` 을 못 찾아 그대로 죽는다 → 정량 60점 0점.
왜 두 배포판이 함께 깔리고 왜 GUI 가 이기는지는 ``download_weights.sh`` §3 참고.
"""

import cv2


def gui_backend() -> str:
    """``cv2.getBuildInformation()`` 의 ``GUI:`` 값. headless 면 ``NONE``."""
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip()
    return "?"


def test_active_cv2_is_headless():
    gui = gui_backend()
    assert gui.upper() == "NONE", (
        f"GUI cv2 빌드가 활성 상태입니다 (GUI={gui}). 이 빌드는 시스템 libGL.so.1 을 "
        "요구해 headless 채점 서버에서 import cv2 가 실패합니다. "
        "`bash download_weights.sh` 를 (인터넷 연결 상태에서) 실행해 정규화하세요."
    )


if __name__ == "__main__":
    test_active_cv2_is_headless()
    print(f"OK  cv2 {cv2.__version__}  GUI={gui_backend()}")
