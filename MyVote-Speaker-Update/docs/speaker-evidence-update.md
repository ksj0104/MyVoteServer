# 화자 미확정 원인 검증을 위한 서버 업데이트

엔진 식별자: `speaker-evidence-2026-09-15`.

이번 변경은 실제 자막을 만든 단어와 원래 시각을 선택적으로 기록한다. 이 기록으로 어느 단어에 화자 근거가 겹치는지 검증할 수 있다. 새 모델·지연 추가·자막 경계 변경은 없다. 일반 Windows 앱의 전사·번역·오버레이 흐름은 그대로 사용한다. 화자 정확도 해결판은 아니다.

## Mac 실행

개발 PC에서 `artifacts/speaker-evidence-ready-2026-09-15/MyVote-Speaker-Update.zip`을 복사한다. 모델·가상환경·LM Studio 설정은 기존 것을 사용한다. ZIP에는 모델·인증서·키가 없다.

이전 서버를 Ctrl+C로 종료하고 ZIP을 새 폴더에 푼 뒤 실행한다.

```bash
bash "$HOME/Downloads/MyVote-Speaker-Update/start-update.command" \
  "$HOME/Downloads/MyVote-Mac-Demo"
```

시작 시 `검증한 업데이트: "speaker-evidence-2026-09-15"`가 표시되고 `gateway.listening`의 `engine_revision`도 같아야 한다. 기존 주파수 분리 옵션을 시험하려면 이전과 같이 `--experimental-overlap-model`과 모델 경로를 추가한다. 단어 시각 기록은 Windows의 평가 클라이언트가 요청했을 때만 켜진다.

## Windows에서 실제 영상 기록

프로젝트 폴더의 PowerShell에서 다음을 실행한다. 저장된 사용자 영상 40:00–42:00의 정확한 120초 WAV를 실시간 속도로 전송한다. PC에서 영상을 다시 재생할 필요는 없다. 실행 중인 앱 세션은 먼저 종료한다.

```powershell
.venv\Scripts\python.exe -m myvote_engine.live `
  --input artifacts/speaker-youtube-YlgFfqaJ-J0-2026-09-15/clip-40m-42m.wav `
  --target 192.168.219.103:50051 `
  --ca-cert artifacts/first-demo-ready-2026-09-15/windows-client/ca.crt `
  --client-cert artifacts/first-demo-ready-2026-09-15/windows-client/client.crt `
  --client-key artifacts/first-demo-ready-2026-09-15/windows-client/client.key `
  --source-language ko --target-language ko `
  --record-word-timings `
  --output-dir artifacts/mac-word-timings-user-video
```

출력 폴더는 사용하지 않은 새 경로로 정한다. 한국어→한국어는 원래 비교와 같은 연결 시험 조건이며 언어 간 번역 품질을 평가하지 않는다. `events.jsonl`의 `caption.source.data.word_trace`에 원래 단어가 보존되고, `summary.json`의 `negotiated_capabilities`에 `caption_words_v1`이 있어야 한다. 서버가 미지원이면 오디오 전송 전에 중단한다. [기록 형식·크기 제한](committed-word-trace.md).

## 이번 조사 결과

- 저장된 90개 자막 중 서로 다른 확정 ID 두 개가 겹치는 자막은 없었다. 낮은 신뢰도의 음성이 76개 자막에 포함되어 있었다. 문장을 더 나누는 것만으로 해결된다는 근거는 없다.
- 고정 2초 뒤의 문맥으로 판단하는 실험은 화자 표시 자막을 4개에서 2개로 줄였고 첫 표시도 늦어져 채택하지 않았다. 실행 서버의 화자 지연·임계값은 바꾸지 않았다.
- 현재 엔진은 임베딩에 사용한 깨끗한 음성 구간에만 식별 결과를 붙인다. 다음 연구는 화자 특징을 학습하는 구간과 실제 발화를 표시하는 구간을 분리하는 것이다. 같은 사람이라는 추론이 틀린 구간으로 전파될 수 있어 정답 주석과 함께 검증해야 한다.

Mac의 현재 서버에는 오디오 없이 인증된 상태 확인을 한 번 수행했다. 정상 종료했지만 엔진 식별자와 새 capability가 없어 이 업데이트가 실행 중임을 확인하지 못했다. 새 코드의 실제 Mac 단어 기록·화자 정확도·M4 Max 지연은 아직 미검증이다.

최종 Python 회귀 검사는 **712개 통과, 생략·실패·오류 0개**, 실제 프로세스 종료 코드 0이었다. 검사 전후 소스 해시가 같다. 선택 기능 협상, 실제 로컬 mTLS, 단어 시각·바이트 한도, 미지원 서버의 캡처 전 중단과 기존 화자·오버레이 계약을 포함한다. 실제 Mac 추론이나 보이는 Windows 화면 검증과는 구분한다.
