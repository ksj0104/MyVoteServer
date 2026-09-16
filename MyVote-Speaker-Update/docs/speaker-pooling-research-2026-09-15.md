# 짧은 단독 발화 묶음: 연구 근거와 고정 평가 절차

검토일: 2026-09-15. 이 문서는 모델 품질 결과가 나오기 전에 작성한 설계·평가 근거다. 실제 실행 결과는 별도 평가 artifact에 기록한다.

## 해결하려는 문제

현재 분석기는 신뢰도 조건을 통과한 단독 발화라도 연속 길이가 1초보다 짧으면 대부분 `insufficient_clean_speech`로 남긴다. 사용자가 제시한 영상에서는 짧은 응답과 겹침 때문에 이 경로의 손실이 확인되었다. 제안하는 방법은 **현재 10초 분석 창에서 같은 로컬 출력 head로 다시 예측한 짧은 단독 발화**를 모아 한 번의 화자 임베딩을 추출하는 것이다. 모델이 단독 발화라고 예측했다는 사실은 실제 한 사람만 들어 있다는 정답이 아니다.

## 확인한 1차 자료

### 1. 창 안의 단독 음성을 모아 임베딩을 추출하는 근거

Bredin의 *pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe* §2.1–2.2는 창마다 활성화된 로컬 화자별로 임베딩 하나를 추출한다. 해당 화자가 활성이고 다른 화자가 비활성인 샘플들을 연결하는 방법을 설명하며, 분할 모델 오류가 임베딩을 나쁘게 만들 수 있다고 명시한다. 로컬 화자 인덱스가 다음 창에서도 같은 사람이라는 보장은 없다. 이는 창 안에서 재검증한 음성을 모으는 방향의 근거이며, 본 구현의 정확한 임계값이나 온라인 신규 ID 확정 규칙을 입증하지 않는다. [논문, Interspeech 2023, §2.1–2.2](https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.pdf)

### 2. 현재 참고 구현의 정확한 마스크 위치

pyannote.audio 3.1.1의 `get_embeddings`는 각 `(chunk, local speaker)` 마스크를 사용한다. 겹침 제외 옵션은 단독 음성 프레임 수가 충분할 때 깨끗한 마스크를 선택하고, 부족하면 전체 화자 마스크로 돌아간다. 본 실험은 부족한 경우 겹침을 포함하는 fallback을 도입하지 않는다. [고정 버전의 화자 파이프라인](https://github.com/pyannote/pyannote-audio/blob/3.1.1/pyannote/audio/pipelines/speaker_diarization.py#L248-L299)

같은 버전의 ONNX WeSpeaker 어댑터는 **전체 창의 fbank와 평균 정규화를 먼저 계산한 후**, 마스크가 선택한 특징 프레임을 모아 ONNX에 넣는다. 따라서 현재 제안한 **PCM 조각 연결 → 기존 fbank/CMN → ONNX**는 그 구현과 수치적으로 같은 알고리즘이 아니다. 연결 경계의 25ms 특징 창과 전체 평균이 달라질 수 있다. 이 실험의 전처리 방법을 `raw_pcm_splice_v1`로 따로 기록하며 pyannote 구현 재현이라고 부르지 않는다. [고정 버전의 ONNX WeSpeaker 구현](https://github.com/pyannote/pyannote-audio/blob/3.1.1/pyannote/audio/pipelines/speaker_verification.py#L482-L565)

### 3. 짧은 발화의 신뢰도와 모델의 범위

Jung 등의 *Short utterance compensation in speaker verification via cosine-based teacher-student learning of speaker embeddings*는 2초 이하 음성에서 성능이 저하되는 문제를 다루고 이를 보완하도록 모델을 학습한다. 이 결과를 현재 WeSpeaker 모델의 200ms 발화 정확도나 묶음의 유효성으로 대입할 수는 없다. 짧은 음성이 실행 가능한 것과 정확한 화자 근거인 것은 다른 조건이다. [논문](https://arxiv.org/abs/1810.10884)

Plaquet·Bredin의 powerset 논문은 겹친 화자 쌍을 별도 클래스로 다루는 로컬 분할 방식을 설명한다. 해당 head 번호는 전역 인물 ID가 아니며, 현재 창의 분할 결과가 바뀌면 저장한 조각도 다시 검증해야 한다. [논문](https://arxiv.org/abs/2310.13025) · [저자 공식 저장소](https://github.com/FrenchKrab/IS2023-powerset-diarization)

WeSpeaker는 화자 임베딩 도구와 학습·평가 구성을 제공한다. 본 실험은 이미 준비된 동일 ResNet34 ONNX 가중치와 기존 80차원 Kaldi fbank·256차원 임베딩을 사용한다. 새 학습이나 모델 교체는 없다. [WeSpeaker 논문](https://arxiv.org/abs/2210.17016) · [현재 어댑터가 따르는 공식 ONNX 전처리](https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/bin/infer_onnx.py)

## 결과 확인 전에 고정한 규칙

- 조각 길이: **200ms 이상, 1초 미만**. 합친 실제 PCM 길이: **1–3초**, 최대 32개 구간.
- 기존 Preview의 신뢰도 `.8`, 겹침 확률 상한 `.1`, match `.53`, novelty `.38`, 확정 2회·누적 단독 음성 3초·후보 만료 20초를 유지한다.
- head 번호는 현재 창 안에서만 사용한다. 저장한 조각은 현재 segmentation으로 다시 검증한다.
- 하나의 묶음은 **임베딩 1회·tracker 근거 1회**다. 자막 적용을 위한 구간별 결과로 나누더라도 확정 근거를 늘리지 않는다.
- 원래 구간 목록을 보존한다. 조각 사이의 무음·다른 사람·겹침은 음성 길이나 화자 배정 범위에 포함하지 않는다.
- 겹친 창에서 같은 샘플을 다시 사용해 신규 화자를 확정하거나 prototype을 갱신하지 않는다.

200ms라는 숫자는 이 실험의 고정된 설계값이다. 위 논문의 짧은 무음 연결에 대한 주석 지침을 최소 발화 길이의 근거로 사용하지 않는다. 결과를 본 뒤 임계값을 골라 가장 좋은 값을 보고하지 않는다.

## 평가 입력과 증거

`scripts/evaluate_speaker_pooling.py prepare`는 ONNX 추론 없이 입력 SHA256, PCM SHA256, 고정 설정과 60개 분석 창을 저장한다. `run`은 이 계획의 입력 해시를 확인하고 실행 전후 소스 해시가 같은지 확인한다.

| 입력 | 구간 | 사용할 정답 |
|---|---|---|
| 사용자 YouTube `YlgFfqaJ-J0` | 명목상 40:00–42:00, 저장한 정확한 120초 WAV | 화자 정답 없음. 예측과 자막 부착 수만 보고 |
| AMI IS1009a | 원본 50–170초 | 기존 `only_words` RTTM |
| AMI ES2004a | 원본 720–840초 | 기존 `only_words` RTTM |

두 AMI 구간은 이전 개발·평가에서 이미 관찰했다. 이번에는 **고정 조건의 회귀 비교**이며 새로운 holdout이라고 부르지 않는다. 사전학습 모델이 해당 회의를 학습하지 않았다는 것도 입증하지 않았다. mixed-headset 녹음과 단어 기반 주석은 완전한 음성/비음성 정답이나 표준 DER 평가가 아니다.

각 입력에서 동일 일정의 segmentation을 한 번 실행해 두 모드에 공유한다. baseline과 pooling의 임베딩은 각 모드의 실제 PCM으로 각각 실행하고 새로운 tracker에서 시작한다. 실제 모델 호출 수를 기록한다. 참조 화자는 분할·묶음·임베딩·tracker 입력으로 사용하지 않는다.

점수는 정확한 support 구간들의 합집합으로 계산한다. 신규 ID 생성 시 한 번의 실제 tracker vote의 support에서 우세한 정답 화자를 사후 매핑하고, 정답 단독 발화 위에서 잘못 붙은 확정 ID 시간, 매핑할 수 없는 ID 시간, 미확정 시간, 중복 신규 ID와 최초 확정 시점을 보고한다. 자막용 fanout을 여러 신규 화자 확정으로 세지 않는다.

사용자 영상은 실제 Mac 재생에서 저장한 **90개 원문 자막의 원래 구간과 생성 이벤트**를 그대로 사용하고 현재 `SpeakerCaptionReducer`로 자동 부착 수를 비교한다. 자막 이벤트에는 저장된 서버 경과 시간을, 분석에는 고정 source-window 종료 시각을 사용한다. 이는 오프라인 혼합 재생 규칙이며 새로운 Mac 실측이나 화면 지연 측정이 아니다. 영상에 대한 화자 정확도는 주장하지 않는다.

## 채택 판단의 한계

확정 ID가 늘어나는 것만으로 개선이라고 판단하지 않는다. 잘못 묶인 사람, 겹침·빈 구간에 붙은 ID, 자막 부착 실패와 추가 임베딩 비용도 함께 본다. 회귀 결과가 좋아도 다른 공개 녹음과 실제 사용자 영상의 수동 화자 주석, Mac에서 ASR·번역과 동시에 실행한 지연 검증은 남는다.
