# MyVote 서버·Windows 업데이트 안내

버전 식별자: **`history-latency-2026-09-16`**.

이번 버전은 Windows의 **저장 기록 열기·화자 편집·내보내기**, 서버의 **전사 지연 단계 기록**, 완료된 번역이 정리 단계의 timeout 때문에 다시 실패로 표시되는 문제의 수정을 포함한다. 기존 동일 언어 보정·문맥 재수정·겹침 분리 기능을 포함한다. 화자 후보 보존 시간은 기존 20초이며, 새 화자 모델은 이번 배포에 추가하지 않는다.

## 1. Mac에 옮길 파일

**`MyVote-Speaker-Update.zip` 한 개**를 옮긴다. 이번 파일의 Windows 위치:

```text
C:\Users\seong\PycharmProjects\MyVote\artifacts\history-latency-ready-2026-09-16\MyVote-Speaker-Update.zip
```

파일명이 같은 예전 배포와 구분해 위 폴더의 ZIP을 사용한다. 기존 Mac의 `MyVote-Mac-Demo/.venv`, `models`, `connection` 인증서 폴더와 겹침 분리 모델 `pytorch_model.bin`은 그대로 사용한다. 새 모델 다운로드나 Python 패키지 재설치는 필요 없다.

## 2. Mac에서 업데이트

1. Windows MyVote에서 **중지** 후 **MyVote 종료**를 누른다. 트레이 메뉴의 종료도 가능하다.
2. Mac 서버 터미널에서 **Control+C**를 눌러 기존 서버를 종료한다.
3. 새 ZIP을 Mac의 `~/Downloads`에 복사한다. 기존 `MyVote-Speaker-Update` 폴더는 Finder에서 다른 이름으로 보관하고 새 ZIP을 푼다.
4. 새 폴더 이름이 정확히 `MyVote-Speaker-Update`인지 확인한다. 폴더 내부의 파일 일부만 기존 폴더에 섞어 넣지 말고 압축을 푼 전체 폴더를 사용한다.
5. LM Studio에서 기존 채팅 모델과 로컬 API 서버 `127.0.0.1:1234`를 실행한다.
6. 터미널에서 실행한다. 다음은 기존 환경과 모델이 다운로드 폴더에 있는 경우다.

```bash
bash "$HOME/Downloads/MyVote-Speaker-Update/start-update.command" \
  "$HOME/Downloads/MyVote-Mac-Demo" \
  --experimental-overlap-model "$HOME/Downloads/pytorch_model.bin" \
  --overlap-threads 1
```

겹침 모델 없이 시험할 때는 두 선택 옵션을 빼고 실행한다.

```bash
bash "$HOME/Downloads/MyVote-Speaker-Update/start-update.command" \
  "$HOME/Downloads/MyVote-Mac-Demo"
```

아래 식별자와 `gateway.listening` 로그를 확인한 다음 Windows에서 연결한다.

```text
검증한 업데이트: "history-latency-2026-09-16"
```

`gateway.listening`의 `engine_revision`도 같은 값이어야 한다. 기존 Mac 주소 기준 연결은 `192.168.219.103:50051`이다. 실행한 터미널은 서버를 쓰는 동안 유지한다.

## 3. Windows 파일과 확인 방법

새 앱:

```text
C:\Users\seong\PycharmProjects\MyVote\artifacts\windows-history-latency-2026-09-16\app\MyVote.Desktop.exe
```

같은 배포 폴더의 `MyVote-Setup-0.1.0-win-x64.exe`는 설치 파일이다. 바탕화면에 준비한 **MyVote History Preview**는 새 앱을 기존 기록 폴더와 연결해 실행한다.

- **저장된 자막 → 기록 열기:** Mac 연결 없이 과거 자막을 확인한다. **이름 변경** 또는 **선택 구간에 지정** 후 앱을 종료하고 다시 열어 수정 내용이 유지되는지 확인한다. **이중 언어 자막 저장**으로 SRT/VTT/JSON을 내보낸다.
- **새로고침:** 최근 저장한 기록을 다시 찾는다. 기본적으로 최근 100개 기록을 표시한다.
- **자막 시작 → 자막 창 열기:** 기존처럼 Windows 재생 음성의 자막을 확인한다. 새 실시간 세션을 시작해야 오버레이를 사용할 수 있다. **백그라운드로 숨기기**는 기본 창을 트레이로 숨긴다.
- 동일 언어는 원문·보정, 다른 언어는 원문·번역으로 표시한다. 후속 문맥에 따른 수정은 원문 번호를 바꾸지 않고 결과 번호를 올린다.

압축 해제 폴더 기준 `docs/saved-session-history.md`에 기록 사용법이 있다. 새 지연 데이터는 원인 분석용이며 `events.jsonl`의 `caption.source.data.source_timing`에서 확인한다. 필드 설명은 같은 폴더의 `docs/asr-stage-timing.md`에 있다.

바로가기는 기존 `artifacts/windows-live-overlap-handoff-2026-09-15/sessions`를 기록 폴더로 사용한다. 실행 파일을 직접 열거나 설치기로 설치하면 기본 기록 폴더가 달라 기존 기록이 목록에 없을 수 있으므로 이번 확인에는 바로가기를 사용한다. 기본 앱 창의 X는 종료가 아닌 트레이 숨기기다.

## 4. LM Studio 모델을 바꿀 때

같은 채팅 API를 쓰는 모델은 코드 교체 없이 바꿀 수 있다. 새 모델을 로드하고 아래 목록의 정확한 `id`를 확인한 뒤 서버 실행 명령에 `--llm-model "정확한-ID"`를 추가해 재시작한다.

```bash
curl http://127.0.0.1:1234/v1/models
```

전사·화자·겹침 분리 모델은 별도 구성 요소다. 채팅 모델 파일을 바꾸는 것으로 함께 교체되지 않는다. 새 모델의 속도와 재수정 JSON 응답 품질은 별도 확인이 필요하다.

## 5. 확인 범위

Python 회귀 검사 **820개**가 실패·오류·skip 없이 통과했다. 실제 Python 백엔드와 Windows 화면 로직을 연결한 검사 **215개**도 통과했다. 시험용 자막 2,104개를 불러오고 화자 수정·내보내기·재시작 후 복원·보류 분리 적용을 확인했다.

설치 패키지에 들어가는 실행 엔진도 **22개 검사**를 통과했다. 별도 Python 설치 없이 실행되는 엔진을 두 번 실행해 기록 열기·수정·내보내기·재시작 후 복원을 확인했다. 실제 설치기 설치/제거와 화면 조작은 이 검사에 포함하지 않았다.

새 Mac·LM Studio의 번역/보정 품질과 실제 추가 지연, 화자 미확정 문제 전체, 다중 모니터 오버레이 사용성을 모두 해결·검증한 버전이라는 뜻은 아니다. 후보 120초 보존 실험은 부착 결과가 악화돼 채택하지 않았다.
