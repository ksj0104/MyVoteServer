# 화자 특징 근거와 발화 표시를 분리하는 실험

2026-09-15. 개발 소스의 선택 기능이며, 앞서 안내한 `speaker-evidence-ready-2026-09-15` ZIP에는 들어 있지 않다. 기존 Mac 업데이트를 다시 옮길 필요는 없다. 실제 Mac 실행·정확도·화면 지연은 별도 검증 대상이다.

## 변경 이유

기존 엔진은 특징 벡터를 추출할 수 있는 1초 이상 음성에만 화자 판정을 붙였다. 예를 들어 한 분석 창에서 같은 사람이 1.5초 말하고 잠시 쉰 뒤 0.2초 덧붙여도, 두 번째 발화는 신원을 알 수 없는 구간으로 남을 수 있었다.

공식 pyannote는 창 안의 로컬 화자마다 선택한 음성으로 임베딩을 얻은 뒤, 그 화자의 분할 활동에 클러스터 결과를 재구성한다. 이 설계의 **근거 음성 선택과 활동 재구성의 분리**를 참고했다. 전체 pyannote 파이프라인이나 학습 과정을 재현한 것은 아니다. [pyannote 3.1.1 get_embeddings/reconstruct](https://github.com/pyannote/pyannote-audio/blob/3.1.1/pyannote/audio/pipelines/speaker_diarization.py#L248-L391).

## 고정된 두 비교 방식

두 방식 모두 기존 프로필의 신원 판정 기준을 유지한다. 분석 창은 최대 10초, 로컬 head는 최대 3개다. head 번호는 현재 창 안의 출력 번호이며 다음 창의 사람 ID가 아니다.

1. 아직 처리하지 않은 시간 범위에서, 기존 신뢰도·겹침·클리핑 조건을 통과하는 연속 1초 이상 음성을 찾는다. 각 head에서 가장 긴 구간 하나를 고르고 동률이면 앞선 구간을 쓴다. 최대 3초까지만 임베딩한다. 음성 조각을 이어 붙이지 않는다.
2. 실제 사용한 음성의 시각과 길이만 `Observation`에 보존하고 tracker를 한 번 호출한다. 관측이 준비된 시각은 현재 창의 끝이다.
3. 원래 관측·판정을 불변 `HeadResolution`으로 보관한다. 표시할 발화는 별도 `DiarizationActivity`로 연결한다. 활동 길이를 관측 횟수나 화자 프로토타입 학습에 더하지 않는다.
4. `trusted`: 기존 기준을 통과한 동일 head의 모든 길이 발화에 결과를 표시한다.
5. `singleton`: 원래 powerset argmax가 단독 화자인 활동까지 표시 범위를 넓힌다. 신원 근거 선택은 4번과 동일하다. 두 화자가 함께 활성화된 argmax나 겹침 확률 0.1 초과 구간은 계속 차단한다.

입력 중복·침묵·다른 head·불확실 겹침으로 이름을 확장하지 않는다. 출력은 각 창이 새로 맡은 시간 범위만 한 번 덮는다. 확인 후보는 같은 후보의 독립 근거가 쌓인 뒤 확정할 수 있다. 창을 넘어서 로컬 head→사람 관계를 저장하지 않는다. 이번 비교에는 짧은 꼬리 보류와 과거 음성의 읽기 전용 신원 조회가 없으므로 기존 방식보다 확인이 늦어질 수도 있다.

## 원래 모델 출력을 보존하는 방법

`retain_frames=True`일 때만 실제 ONNX의 7개 정규화 확률, 원래 argmax head, 겹침 점수와 각 프레임의 원음 시각을 저장한다. 범위는 가드 안의 완전한 실제 PCM 셀이다. 패딩·가드 밖 출력을 발화 근거로 포함하지 않는다. 기존 모델 identity와 기본 span 출력은 같다. 저신뢰도 head가 지워진 이전 span 기록에서 head를 추측해 복원하지 않는다.

Powerset은 두 화자 조합을 별도 클래스로 표현한다. 최대 클래스 점수 0.8은 각 사람의 존재 확률이나 신원 정확도 80%를 뜻하지 않는다. 현재 0.8/0.1 기준은 제품 실험의 제한 조건이며 논문이 보증한 최적값이 아니다. [Powerset loss 논문, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/plaquet23_interspeech.html).

낮은 신뢰도 구간을 더 많이 표시하면 오류도 확산될 수 있다. 관련 연구는 낮은 confidence 영역과 오류의 관계를 분석했다. 따라서 표시되는 자막 수와 함께 잘못된 화자·비발화·실제 겹침에 붙은 이름의 시간을 확인한다. [Powerset calibration 연구, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/plaquet24_interspeech.html).

## 서버 연결과 기록

개발 소스에서는 `--experimental-speaker-activity trusted` 또는 `singleton`으로 선택한다. 기본값은 꺼짐이며 `--experimental-speaker-pooling`과 함께 지정할 수 없다. Mac bootstrap도 같은 옵션을 전달한다. 모델 추가·프로필 변경은 필요 없다.

- `speaker.identity_evidence`: 원래 tracker 판정의 실제 support, 근거 준비 시각과 resolution ID. 임베딩 값은 전송하지 않는다.
- `speaker.observed`: 표시 활동의 실제 시각, `assignment_origin=window_activity`, 위 근거를 가리키는 resolution ID.
- 자막 mapper에는 표시 활동만 전달한다. 근거 이벤트를 다시 넣어 중복 지정을 만들지 않는다. 새 화자 이벤트는 한 관측에서 한 번만 발행한다.

## 검증 기록

고정 비교 스크립트는 `scripts/evaluate_speaker_activity.py`다. 기존에 사용한 AMI 두 구간과 사용자 영상 40:00–42:00을 사용하므로 신규 홀드아웃이 아니다. 사용자 영상에는 화자 정답이 없고 기존 Mac 기록의 90개 자막에는 단어별 시각이 없다. 따라서 자막 전체 구간과 화자 연결을 비교하며, 단어별 정답이나 실제 새 Mac 지연으로 해석하지 않는다.

정답 화자 대응은 이름이 처음 만들어진 **원래 학습 support**에서만 사후 계산한다. 표시 범위를 넓힌 부분으로 정답 화자를 정하면 평가가 자기 결과에 편향되므로 사용하지 않는다. 각 모드의 독립 근거·표시 타임라인·실제 모델 호출·입력 및 소스 해시를 따로 기록한다. 최종 수치는 실행 결과 보고서에 정리한다.
