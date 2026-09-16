# SepFormer 분리 모델 비교와 Mac 실행

작성일: 2026-09-15.

최종 Python 검사 **604개 통과, 생략 0개**. 실제 로컬 SepFormer 가중치 실행 검사를 포함했다.

**WHAMR 16 kHz SepFormer를 실행 가능한 별도 분리기로 추가했다.** 잡음과 잔향을 포함한 영어 음성으로 학습된 두 음원 분리 모델이다. 한국어 방송이나 실시간 자막 성능은 별도로 검증해야 한다. [공식 모델 카드](https://huggingface.co/speechbrain/sepformer-whamr16k/blob/21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5/README.md).

## 구현

`src/myvote_engine/sepformer.py`는 공개 모델의 고정 구조를 Torch로 실행한다. 런타임에 SpeechBrain이나 HyperPyYAML을 설치하지 않아도 된다. 파일 3개의 길이·SHA256과 모든 텐서의 이름·형상·자료형·유한값을 확인하고, 검증한 바이트를 `weights_only=True`로 읽는다.

원음은 모노 16 kHz, 호출당 1~10초다. 두 출력은 시간축과 길이를 유지하며, 모델이 계산하지 않은 마지막 0~7개 샘플은 메타데이터로 표시한다. 출력 A/B는 사람의 이름이나 확정 화자 ID가 아니다.

이 모델의 원래 출력에는 별도의 L1 보정이나 채널별 음량 정규화를 하지 않는다. 잡음이 있는 원음과 두 음원 출력의 합이 다를 수 있다. 무조건 합을 원음으로 맞추면 제거한 성분을 되돌릴 수 있어, 주파수 마스크는 `--compare-frequency`로 별도 비교한다. [공식 분리 호출 코드](https://github.com/speechbrain/speechbrain/blob/89ead74d163463d30c62329a09cfdb4c54f5abc1/speechbrain/inference/separation.py).

현재 어댑터는 **CPU 전용**이다. 실행 스레드는 전용 CLI 프로세스에서 정하며, 어댑터 자체가 다른 모델의 전역 스레드 설정을 바꾸지 않는다. 실시간 gateway에서 자동으로 켜지지 않는다.

## 준비할 파일

모델 폴더는 개발 PC의 다음 위치에 준비돼 있다.

```text
C:\Users\seong\PycharmProjects\MyVote\artifacts\models\sepformer-whamr16k-21a5b500c6f5
```

| 파일 | 바이트 | SHA256 |
|---|---:|---|
| encoder.ckpt | 17,272 | `31d23d395a408b887b8f6ac01e477f00bcba27f95785426359fb52d52f1dc6ed` |
| masknet.ckpt | 113,112,646 | `e5fb8c690668e5d1bbbc9a8256974577093a09cd08e845e9f30024fdd33472ce` |
| decoder.ckpt | 17,272 | `2d959cb46de5b15f008bf5476cf7c19bc680b310c7e10883e3eddfdbde533cb8` |

가중치 합계는 **113,147,190바이트**다. 공개 저장소 `speechbrain/sepformer-whamr16k`의 revision `21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5`이며 모델 카드는 Apache-2.0을 명시한다. 프로그램에서 새 파일을 자동 다운로드하지 않는다.

## Windows 개발 PC에서 실행

```powershell
.venv\Scripts\python.exe -m myvote_engine.sepformer_cli `
  --input artifacts/speaker-youtube-YlgFfqaJ-J0-2026-09-15/clip-40m-42m.wav `
  --model-dir artifacts/models/sepformer-whamr16k-21a5b500c6f5 `
  --start-s 66 --duration-s 8 --threads 1 `
  --compare-frequency --output-dir artifacts/my-sepformer-listening-test
```

출력 폴더는 새 경로여야 한다. `mixture.wav`, `source-A.wav`, `source-B.wav`, `result.json`을 만든다. `--compare-frequency`를 지정하면 주파수 마스크를 적용한 두 WAV도 만든다. 모든 청취 파일에 동일한 추가 감쇠만 사용한다. 두 출력이 서로 다르다는 것만으로 정확한 음원 분리라고 판단하지 않는다.

## Mac에서 실행

개발 PC의 `artifacts/sepformer-preview-ready-2026-09-15/MyVote-Speaker-Update.zip`을 푼 폴더, 위 모델 폴더, 시험 `clip-40m-42m.wav`를 Mac Downloads에 복사한다. ZIP은 소스만 포함하며 모델 가중치는 별도다. 기존 `MyVote-Mac-Demo` 가상환경의 Torch·NumPy를 재사용한다. LM Studio 설정 변경은 필요하지 않다.

```bash
bash "$HOME/Downloads/MyVote-Speaker-Update/separate-sepformer.command" \
  "$HOME/Downloads/MyVote-Mac-Demo" \
  --input "$HOME/Downloads/clip-40m-42m.wav" \
  --model-dir "$HOME/Downloads/sepformer-whamr16k-21a5b500c6f5" \
  --start-s 66 --duration-s 8 --threads 1 --compare-frequency \
  --output-dir "$HOME/Downloads/MyVote-SepFormer-Result"
```

실제 폴더 위치가 다르면 경로를 바꾼다. 성능 측정은 다른 데모 세션을 중지한 상태에서 수행한다. 이 명령은 오디오 분리 시험이며 실제 Mac에서 아직 실행을 확인하지 않았다. Mac GPU 가속이나 실시간 처리 속도를 보장하지 않는다.

라이선스 원문과 변경 고지는 `src/myvote_engine/licenses/`에 포함했다.

## 실제 비교와 채택 판단

최종 어댑터에 실제 2.0004375초 음성을 입력해 SpeechBrain 1.1.1의 `separate_batch`와 비교했다. 두 출력 64,014개 샘플의 최대 차이와 상대 RMS 차이는 모두 **0**이었다. 마지막 7개 미처리 샘플도 같았다. 이는 공식 계산의 재현 근거이며 화자 분리 정확도를 뜻하지 않는다.

기존과 정확히 같은 AMI 통제 혼합 3개에 대한 평균 SI-SDR 개선량은 다음과 같다. 높을수록 정답 파형에 대한 오차가 적다는 뜻이다.

| 처리 | 입력 혼합 대비 평균 개선량 |
|---|---:|
| 기존 ConvTasNet + 공통 이득 보정 | +6.907 dB |
| SepFormer 원출력 | +2.882 dB |
| SepFormer + 주파수 마스크 | +5.385 dB |

원출력은 세 사례 중 두 개에서 기존 모델보다 낮았고, 마스크를 추가한 경우도 두 개에서 기존 모델보다 낮았다. 마스크는 SepFormer 원출력보다 세 사례 모두 점수가 높았다. 이 비교에서는 마스크가 잡음을 재유입해 음질을 악화시켰다는 가설이 입증되지 않았다. 잔차에는 잡음뿐 아니라 음성·모델 변형·출력 배율 차이가 포함될 수 있다.

이 통제 자료의 정답은 단독 화자로 주석된 혼합 헤드셋 녹음에서 가져왔으므로 주석 밖 소리나 잡음까지 포함할 수 있다. 점수는 잡음을 제거하는 모델에 불리하게 작용할 수도 있다. 자료가 3개이며 자연스러운 겹침 음성이나 독립 평가 집합의 정확도를 증명하지 않는다.

사용자 영상의 약 **41:06~41:14**, 8초 구간도 처리했다. Ryzen 5 5600X·약 16GB Windows PC에서 Torch **1스레드** 모델 계산 **29.013초**가 걸렸다. 나머지 4초 입력 3개는 각각 13.891·13.843·14.039초였다. 모델 로딩·입력 수집·전사·번역·표시는 제외한 값이며 Mac 성능이 아니다. 영상에는 정답 분리 음원이 없어 품질 점수를 계산하지 않았다.

이번 결과로 SepFormer를 실시간 기본 모델로 채택하지 않았다. 비교용 어댑터와 오디오 출력은 남긴다. 분리 품질과 처리량을 개선하고 각 출력의 전사·화자 확인·동시 자막까지 연결하는 작업은 아직 남아 있다.

개발 폴더의 근거:

- `artifacts/sepformer-reference-verified-2026-09-15/`: 공식 구현과 최종 어댑터 수치 일치.
- `artifacts/sepformer-evaluation-2026-09-15/RESULTS.md`: 모든 사례의 점수·청취 파일·속도.
- `artifacts/sepformer-evaluation-2026-09-15/comparison/`: 실제 모델 호출 4회, 미양자화 배열 15개, 공통 이득 WAV 26개 및 독립 재검증.
- `artifacts/sepformer-cli-2026-09-15/`: 실제 실행 명령으로 만든 사용자 영상 원음·분리 결과.
