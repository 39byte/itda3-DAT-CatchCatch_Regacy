# ITDA OCR CHALLENGE — 소비기한 추출 · [DAT] 캐치캐치

상품 뒷면 이미지에서 소비기한을 추출해 `submission.csv`(`image_id, year, month, day, final_date`)를
생성한다.

---

## 1. 실행

```bash
git clone https://github.com/39byte/itda3-DAT-CatchCatch.git
cd itda3-DAT-CatchCatch
pip install -r requirements.txt
bash download_weights.sh   # ⚠ pip install 직후 · 오프라인 차단 전에 실행 (아래 2번)

export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output /tmp/executed.ipynb
```

## 2. 가중치 — **오프라인 즉시 구동 (런타임 다운로드 0)**

인터넷이 차단된 채점 환경에서도 별도 다운로드 없이 즉시 실행됩니다.

1. **RapidOCR 기본 가중치** (det/cls/rec ONNX 약 16 MB): `rapidocr-onnxruntime==1.4.4` **wheel 내부에 포함**되어 있어 `pip install` 만으로 로컬에 완비됩니다. 이 중 방향 분류기(cls)를 사용합니다.
2. **NanoDet 날짜 전용 검출기** (5.6 MB): 저장소 `weights/date_detector_ema.onnx` 에 직접 포함되어 있습니다.
3. **PP-OCRv6 small 인식기** (12.1 MB): 저장소 `weights/ppocrv6_rec_small_date.onnx` 에 직접 포함되어 있습니다. 문자 사전은 ONNX 메타데이터에 내장되어 별도 파일이 필요 없습니다.
   원본 `weights/ppocrv6_rec_small.onnx`(21.2 MB)의 출력 헤드 18,710 클래스(85% 한자) 중 날짜 판독에 쓰는 77자만 남긴 판으로, `python -m tools.prune_rec_head` 가 재학습 없이 가중치를 잘라 만듭니다. 판독 결과는 원본과 같고(1,473장 예측 차이 0장) 인식 실행이 약 10% 빠릅니다.
4. **`download_weights.sh`**: 로컬 가중치 무결성(SHA256)을 검증하고, 아래 cv2 배포판 정규화를 수행한 뒤 정상 종료(exit 0)합니다. **`pip install -r requirements.txt` 직후, 인터넷 차단 전에 1회 실행해 주세요.**

### `download_weights.sh` 가 필요한 이유 — cv2 배포판 정규화

`rapidocr-onnxruntime` 이 `opencv-python` 을 하드 의존으로 요구해서 `opencv-python` 과
`opencv-python-headless` 가 **항상 함께** 설치됩니다. 둘은 같은
`site-packages/cv2/cv2.abi3.so` 를 쓰고, pip 의 설치 순서는 레벨 내 역알파벳이라
언제나 headless → opencv-python 순 — 즉 **GUI 빌드가 파일을 덮어씁니다.**

GUI 빌드의 `cv2.abi3.so` 는 번들 Qt5(`libQt5Core/Gui/Test/Widgets`)에 DT_NEEDED 로
걸려 있고, 그 Qt5 가 시스템 `libGL.so.1` 을 찾습니다. 이 라이브러리는 wheel 에
번들되지 않으므로, GUI 라이브러리가 없는 서버 이미지에서는
`ImportError: libGL.so.1: cannot open shared object file` 로 `import cv2` 자체가
실패합니다. `download_weights.sh` 는 마지막에 headless 를 한 번 더 덮어써서 이
순서를 뒤집고, 활성 배포판이 headless 인지 출력으로 확인시켜 줍니다.

버전을 맞추는 것만으로는 해결되지 않습니다 — 같은 4.9.0 이어도 GUI/headless 는 다른
바이너리입니다. 현재 활성 배포판은 아래로 확인할 수 있습니다.

```bash
python -m pytest tests/test_cv2_headless.py -q   # 또는
python -c "import cv2; print(cv2.getBuildInformation())" | grep 'GUI:'   # GUI: NONE 이어야 정상
```

- 노트북 실행 중 **네트워크 호출이 전혀 발생하지 않습니다.**
- 오프라인 실행 검증 완료: 네트워크를 완전히 차단한 상태에서 공식 채점 명령(`jupyter nbconvert`)으로 완주함을 확인했습니다.

## 3. 파이프라인 개요

```
[0] 정규화     EXIF 회전 보정 → JPEG draft() 축소 디코딩 (≥4MP 코호트 318→103 ms)
[1] 검출       RapidOCR DB 검출기 1회 (기본)  또는  NanoDet 날짜 전용 검출기 (Track B)
[2] 박스 필터  종횡비·높이·면적으로 후보를 좁혀 상위 몇 개만 인식으로 넘긴다   ← 핵심
[3] 인식       PP-OCRv6 small 로 크롭 배치 인식 (+ 방향 분류기로 180° 뒤집힘 처리)
               박스마다 여백 두 가지로 잘라 둘 다 읽는다 (크롭 TTA)          ← 핵심
[*] 폴백       검출 박스가 0개일 때만 범용 DB 검출로 재시도 (발동률 0.5%)
[*] 파이프라이닝 다음 이미지 디코딩을 보조 스레드로 선행 (유휴 코어 활용)
[4] 파싱       박스 병합 → 정규식 패밀리 → 부분 결과 허용
[5] 선별       하드 룰 캐스케이드로 소비기한 하나를 고른다                  ← 핵심
[6] 출력       스키마 검증 후 저장 (실패해도 raise하지 않고 안전값으로 복구)
```

**[1] 검출 — NanoDet 날짜 전용 검출기 (Track B).** `weights/date_detector_ema.onnx`
(NanoDet-Plus-m @480, 단일 클래스 `date`, 5.6 MB)가 있으면 `[1]` 이 범용 텍스트
검출 대신 이 모델을 쓴다. ExpDate evaluation 665장 기준:

| | 검출 recall (IoU 0.3) | recall@1 | end-to-end Score |
|---|---|---|---|
| RapidOCR 범용 텍스트 검출 | 82.9% | 26.5% | 36.85 / 50 |
| NanoDet 날짜 전용 검출 | **97.6%** | 50.2% | 39.71 / 50 |
| + 크롭 여백 보정 (`nanodet_expand=0.10`) | 97.6% | 50.2% | 41.41 / 50 |
| **+ 검출기 점수 순서 보존** | **97.6%** | **84.7%** | **43.47 / 50** |

**검출기를 바꾸면 그 뒤 두 단계도 같이 바뀌어야 한다.** 병합 직후 두 곳이 어긋나 있었다.

*(1) 순위.* `filter_boxes` 는 종횡비 사전확률로 박스를 **재정렬**한다 — 범용 텍스트
검출기에는 "날짜다움" 점수가 없어 기하로 추측할 수밖에 없기 때문이다. NanoDet 은
단일 클래스 `date` 검출기라 박스 점수가 곧 날짜다움인데, 그 위에 종횡비 재정렬을
덮으면 소비기한 박스 recall@1 이 **84.7% → 50.2%** 로 무너진다. 지금은
`Engine.detect_and_filter()` 가 NanoDet 경로에서 재정렬을 끄고 검출기 순서를 쓴다.

*(2) 크롭 여백.* RapidOCR DB 박스는 `unclip_ratio=1.6`
으로 이미 부풀려져 나오지만 NanoDet 회귀 박스는 글자에 딱 맞아서, 그대로 자르면
`2021.08.04` 가 `2021.08.0` 이 된다 — 이 한 가지로 665장 중 88장이 퇴행했다.
여백 비율은 스윕으로 정했고 0.06~0.15 가 평탄해 중앙값을 골랐다
(`itda_ocr/nanodet_det.py: DEFAULT_EXPAND`).

`predict.ipynb` 는 이 파일의 존재를 확인해 자동으로 NanoDet 경로를 켠다.

**[3] 인식 — 크롭 TTA (`Config.crop_tta=0.20`).** 여백을 **하나 고르는 것으로는 절단을 못 막는다.**
스윕이 0.06~0.15 에서 평탄했던 것도, 실측한 부족량을 그대로 반영한 비대칭·높이비례 마진이
+0.139(개선 18 / 퇴행 13, p=0.47)에 그친 것도 같은 이유다 — **"끝 글자가 크롭에 들어오는 임계 여백"이
이미지마다 다르다.** 그래서 값을 고르는 대신 박스마다 크롭을 두 개(원래 여백, +20%) 만들어 같은 배치로
읽고 **같은 기하 좌표**를 붙인다. 그러면 새 판정 로직 없이 기존 기계가 정리한다 —
`_filter_overlapping_boxes` 가 줄 병합에서 긴 판독을 남기고, `rank()` 동점 규칙(늦은 날짜 우선)이
절단 판독을 떨어뜨린다(끝이 잘리면 일이 작아지므로). 앞이 잘려 2자리 연도로 강등됐던 판독은 4자리로
복구되어 패턴 사전점수에서 이긴다.

통합 1,473장(KIST 평가 665 + expiry_region val 808) A/B:

| 부분집합 | 이전 | + 크롭 TTA | Δ | 완전일치 | 개선 / 퇴행 | 부호검정 p |
|---|---|---|---|---|---|---|
| KIST 665 | 46.789 | 47.835 | +1.045 | 92.3% → 94.7% | 23 / 5 | 9.1e-4 |
| expiry 808 | 44.338 | 44.994 | +0.656 | 86.1% → 87.8% | 18 / 10 | 0.19 |
| **통합 1,473** | 45.445 | **46.276** | **+0.832** | 88.9% → **90.9%** | 41 / 15 | **6.9e-4** |
| 고유 1,338 (바이트 중복 제거) | 45.142 | 46.039 | +0.897 | 88.2% → 90.3% | 39 / 13 | 4.1e-4 |
| expiry 비중복 674 (튜닝 미사용) | 43.464 | 44.273 | +0.809 | 84.0% → 85.9% | 17 / 8 | 0.11 |

회수한 41장의 원래 원인은 크롭 기하 27 · 선별 4 · 인식 4 · 순위 3 · 기타 3 이다. 검출 박스가 GT 박스보다
**가로로만** 모자라고(왼쪽 p50 14.1px / p90 29.4px, 오른쪽 7.7 / 19.6px) 세로는 이미 충분하다는 실측이
근거다. 최신 경량 인식기도 여백이 바뀌면 판독이 달라진다 — PP-OCRv6 기술보고서의 자체 지표
*crop margin robustness* 가 75.32% 다. TTA 는 그 불안정성을 없애려 하지 않고 **표본으로 쓴다.**

**[1] 검출 0개 폴백 (`Config.det_fallback=True`).** NanoDet 이 박스를 **하나도** 내지 못한 이미지는
GT 박스로 자르면 오라클이 정답을 깔끔히 읽는다 — 순수한 검출 재현율 문제다. 그럴 때만 `Engine` 이 이미
들고 있는 RapidOCR 범용 DB 검출기로 다시 찾는다. 통합 1,473장 중 **8장(0.54%)** 에서만 발동하므로 평균
지연 영향은 없고, 기존 예측을 건드릴 수 없는 구조라 퇴행이 불가능하다.
**A/B: 통합 +0.170 · 비중복 674장 +0.371 · 개선 5 / 퇴행 0** (회수한 5장 전부 0점 → 50점).

**[0] 디코딩 파이프라이닝 (`Config.prefetch=True`).** 디코딩은 코어를 0.61개만 쓰는데 그 구간에 3.4개가
논다(4코어 기준 실측). 메인 스레드는 시간의 절반 이상을 ORT `Run()` 에서 보내며 GIL 을 놓으므로,
`ThreadPoolExecutor(1)` 로 **다음 이미지 디코딩을 미리** 하면 유휴 코어가 채워진다.
**A/B(500장 3회 교대): 장당 −5.89%, 3회 모두 같은 방향, 예측 차이 0장.** 공식 실행의 CPU 사용률이
211~236% → 249~256% 로 오른 것으로도 확인된다. 선행이 실패하면 조용히 메인 스레드 디코딩으로 되돌아간다.

**PP-OCRv4 폴백은 이와 함께 끈다(`rec_fallback=False`).** 두 장치가 같은 문제("한 번 더 읽어야 하는
이미지")를 노리는데 TTA 가 더 넓게 잡는다. 폴백 단독 기여는 +0.133 이지만 TTA 위에서는 **+0.051** 로
줄고, 느린 v4 호출이 꼬리를 늘린다 — 1,473장 최대 지연 4,192 ms → **1,132 ms**, 1초 초과 32장 → 5장.
인식이 2배가 되는 비용은 폴백 제거로 상쇄되어 채점 환경 환산 장당 약 134~140 ms 다.

**[2]와 [5]가 이 설계의 요지입니다.**

- **[2] 박스 필터** — 이미지의 글자 밀도가 높아(1000px 기준 연결성분 중앙값 88개,
  텍스트 20~60줄) 검출된 텍스트를 전부 인식하면 어떤 모델을 써도 예산을 초과합니다.
  인식은 크롭당 비용이 붙으므로, 인식기에 넘기기 전에 값싼 기하 신호로 후보를 줄입니다.
  튜닝 목표는 캐스케이드 규칙 그대로 — 싼 단계는 재현율을 거의 1로 두고 정밀도는
  뒤 단계가 회수합니다.
- **[5] 선별** — 이미지의 71%에 날짜 모양 오답(품목보고번호·바코드·전화번호·특허번호·
  로트코드)이 있습니다. 이 과제의 본질은 인식이 아니라 **선별**입니다. 가장 효과가 큰
  단일 규칙은 "날짜 숫자가 더 긴 숫자열의 일부이면 기각"이며, 이는 `20130628332176`
  처럼 앞 8자리가 유효 날짜인 품목보고번호를 한 번에 제거합니다.

각 단계의 설계 근거가 된 문헌은 7절에 정리했습니다.

## 4. 저장소 구조

```
predict.ipynb        채점 대상 노트북 (CONFIG 셀은 템플릿 원문 그대로)
requirements.txt     == 로 고정
itda_ocr/            파이프라인 (import 시 부작용 없음)
  engine.py            엔진 래퍼, 스레드 고정, 박스 필터
  pipeline.py          이미지 로딩, 장당 처리, 배치 드라이버
  parse.py             정규식 패밀리, 박스 병합, 부분 추출
  select.py            선별 하드 룰 캐스케이드
eval/                자체 채점 하네스 (채점 대상 아님)
  score.py             공식 산식 + 오류 분류표
  expdate.py           ExpDate 데이터셋 → 정답 CSV 어댑터
  run.py               실행 → 채점 → 리포트
bench/timing.py      장당 비용 측정 (코어·스레드 고정)
tests/               단위 테스트
```

## 5. 개발용

```bash
pip install -r requirements-dev.txt             # 검증용(pytest). 채점 requirements 와 분리
python -m pytest tests/ -q                       # 단위 테스트
python -m bench.timing --images data --limit 60 # 장당 비용
python -m eval.run --images <dir> --gt <gt.csv>                              # 기준선 정확도
python -m eval.run --images <dir> --gt <gt.csv> --nanodet weights/date_detector_ema.onnx  # NanoDet 검출
python -m eval.failure_taxonomy --images <dir> --gt-dates <csv> --gt-boxes <json>  # 단계별 오답 분류
```

정답 CSV는 출처를 가리지 않습니다 — 스키마(`image_id,year,month,day,final_date`)만
같으면 ExpDate 어댑터 산출물이든 손으로 라벨링한 파일이든 그대로 채점됩니다.

## 6. 외부 데이터셋

성능 검증에 **ExpDate (Products-Real)** 를 사용합니다.

> Seker, A. C., & Ahn, S. C. (2022). A generalized framework for recognition of
> expiration dates on product packages using fully convolutional networks.
> *Expert Systems with Applications*, 203, 117310. KIST. CC BY 4.0.
> https://felizang.github.io/expdate/

데이터셋은 저장소에 포함하지 않습니다(`.gitignore`). 변환:

```bash
python -m eval.expdate --root <ExpDate 압축 해제 경로> --inspect   # 구조 먼저 확인
python -m eval.expdate --root <ExpDate 압축 해제 경로> --out labels/expdate
```

## 7. 참고 문헌

3절의 각 단계가 어디에 근거했는지만 간추렸습니다.

**검출 · 인식**

- Seker, A. C., & Ahn, S. C. (2022). A generalized framework for recognition of expiration dates on product packages using fully convolutional networks. *Expert Systems with Applications*, 203, 117310. — 가장 가까운 선행 연구. 날짜영역 검출 → DMY 분할 → 인식의 3단 캐스케이드. **[1]** 의 날짜 전용 검출기 구성과 ExpDate 개발셋의 출처.
- Peng, H., Bayón, J., Recas, J., & Guijarro, M. (2025). Efficient Expiration Date Recognition in Food Packages for Mobile Applications. *Algorithms*, 18(5), 286. — 경량 백본 직접 검출이 쓸 만한 정확도에 도달한다는 근거. **[1]** 의 NanoDet 선택. (단, FP16 권고는 CPU 채점 환경에 맞지 않아 따르지 않음)
- Zheng, J., Li, J., Ding, Z., Kong, L., & Chen, Q. (2023). Recognition of expiry data on food packages based on improved DBNet. *Connection Science*, 35(1), 2202363. — DB 계열 검출기가 식품 포장 도트매트릭스에 적합하다는 도메인 근거. **[1]** 의 RapidOCR 기본 경로.

**후보 축소**

- Viola, P., & Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. *CVPR 2001*. — 캐스케이드 튜닝 규칙(싼 단계는 재현율을 거의 1로 두고 정밀도는 뒤 단계가 회수). **[2]** 박스 필터의 튜닝 목표.

**파싱**

- Chang, A. X., & Manning, C. D. (2012). SUTime: A Library for Recognizing and Normalizing Time Expressions. *LREC 2012*. — 인식과 정규화의 분리, 불완전한 매치를 버리지 않고 부분 결과로 내보내는 설계. **[4]** 의 부분 추출이 필드별 부분점수로 직결됩니다.

**선별 (이 과제의 본질)**

- Huang, Z., et al. (2019). ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction. *ICDAR 2019*. — 검출/인식/KIE 3분할로 "문자 인식이 아니라 필드 선별 문제"라는 과제 경계를 확립. 1위 방법이 lexicon + 정규식이었습니다.
- Majumder, B. P., et al. (2020). Representation Learning for Information Extraction from Form-like Documents. *ACL 2020*, 6495–6504. — 후보 생성 → 후보 점수화의 2단계 프레임. **[4]→[5]** 구조와 동일.
- Gunel, B., et al. (2021). Data-Efficient Information Extraction from Form-Like Documents. *DI@KDD 2021*. — 같은 타입의 필드가 후보를 공유하는 문제(제조일자 vs 소비기한)의 정확한 서술과, 할당 단계에 비즈니스 로직 제약을 넣는 해법. **[5]** 하드 룰 캐스케이드.
- Palm, R. B., Winther, O., & Laws, F. (2017). CloudScan: A Configuration-Free Invoice Analysis System Using Recurrent Neural Networks. *ICDAR 2017*. — 손으로 만든 특징 위의 선형 모델이 대규모 데이터에서도 LSTM과 노이즈 수준 차이. **[5]** 랭커를 규칙으로 두는 근거.

**예산 · 안전**

- Zilberstein, S. (1996). Using Anytime Algorithms in Intelligent Systems. *AI Magazine*, 17(3), 73–83. — interruptible anytime algorithm. **[6]** 이 시작 즉시 유효 CSV를 쓰고 이후 개선하는 설계의 정식 명칭.
- Huang, G., et al. (2018). Multi-Scale Dense Networks for Resource Efficient Image Classification. *ICLR 2018*. — budgeted batch classification, 즉 "고정된 총 연산량을 쉬운 입력과 어려운 입력에 불균등하게 쓴다". 장수 N이 미지인 2400초 예산 배분의 정의.
