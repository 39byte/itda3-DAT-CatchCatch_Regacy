"""PP-OCRv6 small 인식기의 출력 헤드를 날짜 문자로 잘라낸 ONNX 를 만든다 (재학습 없음).

    python -m tools.prune_rec_head

마지막 층 ``linear_8`` 은 ``[120, 18710]`` 로 모델 파라미터의 42.6% 인데, 사전의 85% 가
한자이고 우리가 쓰는 건 ``DATE_CHARS`` 77자뿐이다. 원래는 18,710개를 다 계산한 뒤
``DateCTCLabelDecode`` 가 나머지를 −∞ 로 지웠다.

행렬곱의 각 출력은 자기 열의 가중치로만 계산되고 softmax 는 단조라서, 허용 문자 사이의
argmax — 곧 디코드 결과 — 는 원본에 −∞ 마스킹을 한 것과 같다. 인식 신뢰도 값만 77개로
재정규화되어 달라지는데, 파이프라인은 그 값을 판정에 쓰지 않는다.
실측: 인식 ORT −10.7%, 마스킹 7.3 ms/장 제거, 1,473장 예측 차이 0장.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from rapidocr_onnxruntime.ch_ppocr_rec.utils import CTCLabelDecode

from itda_ocr.engine import DATE_CHARS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "weights" / "ppocrv6_rec_small.onnx"
DST = ROOT / "weights" / "ppocrv6_rec_small_date.onnx"
HEAD_W, HEAD_B = "linear_8.w_0", "linear_8.b_0"


def prune(src: Path = SRC, dst: Path = DST) -> list[str]:
    model = onnx.load(str(src))
    meta = {p.key: p for p in model.metadata_props}
    # RapidOCR 디코더의 문자표 = ['blank'] + 사전 + [' ']. 모델 출력 차원과 1:1 로 대응한다.
    full = CTCLabelDecode(character=meta["character"].value.splitlines()).character
    allowed = set(DATE_CHARS)
    keep = [i for i, c in enumerate(full) if i == 0 or c in allowed]
    assert full[-1] == " " and keep[-1] == len(full) - 1, "끝의 공백 문자가 유지돼야 디코더가 다시 붙인다"

    inits = {t.name: t for t in model.graph.initializer}
    w = numpy_helper.to_array(inits[HEAD_W])
    b = numpy_helper.to_array(inits[HEAD_B])
    assert w.shape == (120, len(full)) and b.shape == (len(full),)
    inits[HEAD_W].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(w[:, keep]), HEAD_W))
    inits[HEAD_B].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(b[keep]), HEAD_B))

    # 선언된 출력·중간 형상의 클래스 차원도 맞춘다 (ORT 형상 추론 경고 방지)
    for vi in [*model.graph.value_info, *model.graph.output]:
        dims = vi.type.tensor_type.shape.dim
        if dims and dims[-1].dim_value == len(full):
            dims[-1].dim_value = len(keep)

    # 사전 메타데이터는 blank 와 디코더가 붙이는 끝 공백을 뺀 나머지
    new_dict = [full[i] for i in keep[1:-1]]
    meta["character"].value = "\n".join(new_dict)
    assert CTCLabelDecode(character=new_dict).character == [full[i] for i in keep]

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return [full[i] for i in keep]


if __name__ == "__main__":
    chars = prune()
    print(f"{DST.name}: {len(chars)} 클래스, {DST.stat().st_size / 1e6:.1f} MB "
          f"(원본 {SRC.stat().st_size / 1e6:.1f} MB)")
