# 음성 주파수 분리 구현과 실행

작성일: 2026-09-15.

**후속 구현:** 아래 내용은 최초 파일 분리 단계의 기록이다. 현재는 [실시간 두 줄 자막 경로](live-overlap.md)까지 연결되어 있으며, 최신 ZIP·실행 명령은 [주파수·화자 실험판](frequency-speaker-preview.md)에 있다. 아래 579개 검사 수치는 당시 소스에 대한 결과다.

Python 전체 회귀 검사 **579개 통과, 생략 0개**. 실제 Mac의 새 분리 명령 실행은 아직 확인하지 않았다.

**학습 모델의 두 음원 추정에 시간·주파수 마스크를 적용하는 실험 경로를 구현했다.** 실제 WAV를 입력하면 원음과 두 분리 추정 WAV, 처리 시간·설정 보고서를 만든다. 현재 실행 중인 Mac 실시간 자막 서버에는 자동으로 활성화되지 않는다. 이 단계의 출력 A/B는 화자 ID가 아니며, 분리 품질과 출력 채널의 연속성을 검증한 뒤 자막 경로에 연결해야 한다.

## 적용한 알고리즘

1. 기존 로컬 ConvTasNet이 모노 혼합음에서 두 음원을 추정한다.
2. 두 출력에 **동일한 L1 이득**을 적용해 원 모델의 큰 출력 배율을 보정한다.
3. STFT로 짧은 시간마다 주파수 성분을 계산한다. 16 kHz 입력, Hann 창 512샘플(32 ms), 이동 128샘플(8 ms)이다.
4. 추정 음원의 제곱 크기로 전력을 구한다. 기본값은 앞뒤 한 프레임과 현재 프레임의 전력 평균이며, 주파수 방향으로는 평활하지 않는다.
5. 각 시간·주파수 칸에서 `M₀=P₀/(P₀+P₁)`, `M₁=1−M₀`를 계산하고 **원 혼합 스펙트럼**에 곱한다. 두 추정이 모두 약하면 반씩 배분한다.
6. 역 STFT로 같은 길이의 두 파형을 복원한다. 합이 원음과 일치하는지 검사하고, WAV 저장 시 원음과 두 출력에 같은 감쇠만 적용한다.

기본값은 단일 마스크 계산이다. 선택적으로 시간·주파수 평활 반경 0~2와 추가 재분석 0~2회를 지정할 수 있다. 추가 재분석은 학습이나 EM 최적화가 아니다. 기본 옵션은 최적화 완료된 값이 아닌 고정 비교 설정이다.

기존 `project_mixture_consistency`는 추정 위상을 보존하며 합 잔차를 분배한다. 새 방법은 원음 위상과 추정 전력 비율을 사용하므로 서로 다른 방법이다. 같은 주파수에서 두 음원이 상쇄되는 경우 마스크만으로 원래 위상을 복구할 수 없다. 모든 음성에서 기존 방법보다 좋아진다고 가정하지 않는다.

논문과 설계 근거: [시간·주파수 분리 연구](frequency-separation-research-2026-09-15.md). Wiener 비율 마스크, STFT, 혼합 일치 조건을 조합한 구현이며 새로운 학습 모델을 발명하거나 논문의 전체 학습 과정을 재현한 것은 아니다.

## 개발 PC에서 실행

프로젝트 폴더에서 다음과 같이 실행한다. 아래 입력의 시작점은 원 영상 약 41:06이며, 기존에 확보한 40:00~42:00 WAV의 66초부터 8초를 사용한다. 원 영상과의 AAC 샘플 단위 정렬은 확인하지 않았다.

```powershell
.venv\Scripts\python.exe -m myvote_engine.separate_cli `
  --input artifacts/speaker-youtube-YlgFfqaJ-J0-2026-09-15/clip-40m-42m.wav `
  --model artifacts/models/convtasnet-libri2mix-16k-e1ef95ab7a03/pytorch_model.bin `
  --start-s 66 --duration-s 8 `
  --output-dir artifacts/my-frequency-listening-test
```

출력 폴더는 새 경로여야 한다. `mixture.wav`, `source-A.wav`, `source-B.wav`, `result.json`을 만든다. 입력은 모노 16 kHz PCM16 WAV이며 한 번에 1~10초를 선택한다. 2초 실험보다 문맥을 넓혀 8초를 사용하지만 실제 음성 순도를 보장하지 않는다.

## Mac에서 실행

개발 PC의 `artifacts/frequency-preview-ready-2026-09-15/MyVote-Speaker-Update.zip`에 이번 소스를 담았다. 이전 동명 ZIP과 구분해 별도 폴더에 푼 뒤, 다음 두 파일을 Mac에 함께 복사한다.

- 기존 로컬 모델: `artifacts/models/convtasnet-libri2mix-16k-e1ef95ab7a03/pytorch_model.bin` — 20,394,640바이트.
- 시험 입력: `artifacts/speaker-youtube-YlgFfqaJ-J0-2026-09-15/clip-40m-42m.wav`.

모델 SHA256은 `8d97f012f7b2f22bb79cb0d0983a7ba27a52c1796ee3f63cbf25b4d28630adce`이며 어댑터가 크기와 해시를 검사한다. 다른 동명 모델을 대신 사용하지 않는다. 새 후처리에는 추가 학습 가중치가 없다. 기존 데모 가상환경의 NumPy·Torch·TorchAudio를 사용하며 LM Studio 모델 설정은 필요하지 않다.

세 파일을 다운로드 폴더에 두고, 새 소스 ZIP을 푼 폴더 이름이 `MyVote-Speaker-Update`일 때:

```bash
bash "$HOME/Downloads/MyVote-Speaker-Update/separate-audio.command" \
  "$HOME/Downloads/MyVote-Mac-Demo" \
  --input "$HOME/Downloads/clip-40m-42m.wav" \
  --model "$HOME/Downloads/pytorch_model.bin" \
  --start-s 66 --duration-s 8 \
  --output-dir "$HOME/Downloads/MyVote-Frequency-Result"
```

기존 데모 폴더 위치가 다르면 첫 번째 경로를 바꾼다. 새 실행은 검증된 소스를 기존 Python으로 직접 실행하며 설치된 엔진을 덮어쓰지 않는다. 서버와 동시에 실행하면 CPU를 공유하므로, Mac 성능 비교 시 Windows 데모 세션을 종료한 상태에서 측정한다. 실시간 서버 시작 명령 `start-update.command`와 이 오디오 분리 시험 명령은 별개다.

## 해석과 남은 통합

- `frequency_runtime_ms`: 이득 보정과 주파수 처리 시간. `model_runtime_ms`: 분리 모델 시간. `runtime_ms`: 청크 처리 전체 시간으로 첫 NumPy 가져오기와 모델 사전 로드는 제외한다.
- 원음을 모으는 시간·전사·번역·화면 표시 시간은 포함하지 않는다. STFT 이동 8 ms가 자막 지연 8 ms라는 뜻이 아니다.
- 기본 모델은 CPU에서 실행한다. Mac M4 Max·MPS 실측이나 가속 지원을 주장하지 않는다.
- 모델 카드의 16 kHz 설명과 체크포인트의 8 kHz 메타데이터, 카드 내 CC-BY-SA 3.0/4.0 표기가 일치하지 않는 기존 제한이 있다. 이 어댑터는 카드의 16 kHz 경로를 따른다.
- 합 잔차가 작거나 출력끼리 다르게 들리는 것만으로 두 화자가 올바르게 분리됐다고 판단하지 않는다. 소음·웃음·배경 음악도 출력에 배분된다.
- 실제 서버 통합에는 겹침 감지 후 별도 작업 예약, 청크 사이 A/B 순서 연결, 각 음원의 품질 확인과 전사, 복수 화자를 표시할 자막 계약이 필요하다. 현재 화자 미확정 문제의 해결판으로 활성화한 기능은 아니다.

## 실제 비교 결과

고정된 같은 모델 추정으로 세 방법을 비교했다. AMI 회의에서 서로 다른 화자의 단독 발화로 주석된 4초 구간을 추출해, 정답 파형을 보존한 혼합 3개를 만들었다. 혼합 헤드셋 녹음이므로 주석에 없는 웃음·잡음이나 다른 사람의 작은 음성은 남을 수 있다. 이는 자연스러운 동시 발화의 정확도나 독립 평가 집합을 뜻하지 않는다.

| 처리 | 혼합 원음 대비 평균 SI-SDR 개선량 |
|---|---:|
| 분리 모델 + 동일 L1 이득 보정 | +6.907 dB |
| 기존 합 잔차 보정 | +2.818 dB |
| 새 시간·주파수 마스크 | +5.103 dB |

SI-SDR은 정답 파형에 비해 다른 음성·변형이 얼마나 남는지 보는 신호 점수이며 높을수록 좋다. 새 방법은 기존 합 잔차 보정보다 세 사례 모두 높았지만, 모델과 이득 보정만 쓴 경우보다 세 사례 모두 낮았다. 한 사례는 새 방법도 −3.212 dB로 혼합 원음보다 나빠졌다. 따라서 새 기법을 일반적인 품질 개선이나 실시간 기본값으로 승격하지 않았다.

유튜브 약 **41:06~41:14**의 실제 8초 입력도 처리했다. 첫 비교 실행에서 Windows CPU 1스레드 기준 모델 약 **2.282초**, 새 DSP **45.8 ms**였다. 별도로 실제 CLI 전체 연결에서도 처리 약 **2.356초**, 이 중 주파수 처리 **42.1 ms**를 기록했다. 8초 문맥을 모으는 시간과 Mac 실측은 포함하지 않는다. 이 영상에는 정답 분리 파형이 없어 분리 품질 점수를 붙이지 않았다.

최종 수치 입력 검증 보완 후 동일 비교를 재실행해 **모든 청취 WAV와 품질 점수가 이전과 같음**을 확인했다. 재실행은 단위 검사와 동시에 수행했으므로 그 처리 시간은 성능 비교에 사용하지 않는다. 초기 비교 4회 + CLI 1회 + 최종 비교 4회, 실제 분리 모델 호출 총 9회를 수행했다.

개발 폴더의 결과:

- `artifacts/frequency-separation-verified-2026-09-15/`: 최종 코드로 만든 4개 입력·3개 방법의 비교와 WAV 34개.
- `artifacts/frequency-cli-2026-09-15/`: 실제 실행 명령의 원음·출력 A/B와 결과 JSON.
- `artifacts/frequency-separation-2026-09-15/`: 입력 검증 보완 전 최초 비교와 독립적인 WAV 점수 검증 기록.

소스: `src/myvote_engine/frequency_separation.py`(DSP), `overlap_separation.py`(모델과 DSP 연결), `separate_cli.py`(WAV 실행).
