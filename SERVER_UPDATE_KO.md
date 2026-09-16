# MyVote 의미 구간 번역 업데이트

버전: **`semantic-translation-2026-09-16`**

공개 배포 안내다. Mac `192.0.2.10`·Windows `192.0.2.20`은 문서 전용 예시 주소이며 실제 주소가 아니다. `%USERPROFILE%`·`$HOME` 경로는 각 사용자 환경에 맞춘다. 운영 `server.toml`은 Git에서 제외하며 [server.example.toml](server.example.toml) 복사 후 실제 LAN 주소·인증서·모델을 별도로 준비한다. 이 문서의 초기 업데이트 정책과 현재 프로젝트 보완 정책은 [README](README.md)를 구분해 확인한다. [Git 공개 범위](docs/git-publishing.md)

## 달라진 동작

전사·번역·화자 추론은 Mac 서버에서 실행한다. Windows는 PC 재생 음성을 보내고, 원문 미리보기와 확정 자막을 표시한다.

1. 전사 모델이 원문을 갱신하면 기본 앱과 오버레이의 **전사 중** 영역에 먼저 표시한다.
2. 서버는 전사에서 확정된 단어를 화자별로 모아 오케스트레이터 모델에 보낸다. 아직 바뀔 수 있는 전사 끝부분은 미리보기로만 표시한다. 음성에서 화자가 확인되지 않았으면 발화별 임시 버퍼를 사용한다.
3. 오케스트레이터는 의미상 자연스럽게 옮길 수 있는 앞부분을 선택한다. 아직 문맥이 부족하면 기다리라는 응답을 보낸다. 선택된 실제 원문만 **TranslateGemma 12B MLX 4bit**에 보내 번역한다.
4. 선택한 구간만 확정 자막으로 전달하고, 남은 원문은 다음 판단에 사용한다. 요청 중 추가된 원문도 보존한다.
5. 원문과 목표 언어가 같으면 오케스트레이터가 선택한 구간의 명확한 전사 오류·띄어쓰기·구두점만 보정한다. TranslateGemma 번역 경로를 사용하지 않는다. 기존 후속 문맥 재수정도 오케스트레이터로 유지한다.

미리보기는 확정 자막 목록과 SRT/VTT에 중복 저장하지 않는다. 실제 원문 시간과 단어 기록은 확정 구간에 연결한다. JSON 형식·구간 ID가 틀리거나 모델 요청이 실패하면 해당 원문은 보존하고 번역 실패로 기록한다.

### 기본 대기 정책

| 항목 | 기본값과 의미 |
|---|---|
| 판단 요청 간격 | 화자 버퍼별 최소 0.8초. 오케스트레이터는 한 번에 한 요청, 번역은 최대 4개 독립 작업을 처리한다. |
| 의미 구간 수집 | 첫 확정 단어를 받은 뒤 최대 4초에 마감을 요청한다. 발화 종료도 마감 경계가 된다. |
| 모델 요청 제한 | 기본 총 3.5초: 구간 판단 최대 1초 + 번역 최대 2.5초. 클라이언트의 더 짧은 번역 예산과 남은 전체 체류 제한을 따른다. |
| 원문 전체 체류 제한 | 첫 단어의 서버 전사 확정부터 최대 8초. 초과하면 원문을 보존하고 해당 번역 시도를 종료한다. |
| 대기 원문 | 최대 64개 단어 단위·4,000자. 초과 입력은 실패 원문으로 남긴다. |
| 앞 문맥 | 화자 버퍼의 완료 원문 2개와 최근 대화 2개를 구간 판단에 참고한다. TranslateGemma에는 선택된 원문만 전달한다. |

4초·8초는 서버 처리 정책이다. 음성 발생부터 화면 표시까지의 실측 지연이나 보장값은 아니다. 전사 안정화, 모델 처리, 네트워크와 화면 표시 시간이 따로 있다. 모델이 자주 기다리면 번역 요청 수가 줄지 않을 수도 있어 실제 영상으로 품질과 지연을 확인해야 한다.

## Mac에 옮길 파일

**다음 폴더의 `MyVote-Speaker-Update.zip` 한 개**를 옮긴다. 같은 이름의 이전 배포와 구분한다.

```text
%USERPROFILE%\PycharmProjects\MyVote\artifacts\semantic-translation-ready-2026-09-16\MyVote-Speaker-Update.zip
```

기존 Mac의 **`MyVote-Mac-Demo` 폴더는 계속 필요하다.** 그 안의 `.venv`, `models`, `connection`을 재사용하므로 다시 복사하거나 설치할 필요는 없다. 전사·화자 모델은 그대로 사용한다. 기존 `pytorch_model.bin`은 선택한 겹침 분리 기능용이다.

TranslateGemma와 별도로 오케스트레이터 모델이 하나 필요하다. 초기 후보는 **`mlx-community/Qwen3-4B-Instruct-2507-4bit`**다. 두 모델을 LM Studio에 로드한다. 압축 해제 폴더의 `docs/translation-orchestrator-architecture.md`에 구조와 모델 선택 근거가 있다.

## Mac에서 실행

1. Windows MyVote에서 **중지 → MyVote 종료**를 누른다. 창의 X는 트레이로 숨기기다.
2. Mac의 기존 서버 터미널에서 **Control+C**를 누른다.
3. 기존 `~/Downloads/MyVote-Speaker-Update` 폴더를 다른 이름으로 보관한다. 새 ZIP을 `~/Downloads`에서 풀어 새 전체 폴더를 사용한다.
4. LM Studio에서 TranslateGemma와 오케스트레이터를 로드하고 로컬 API 서버를 켠다. 기존 주소는 `127.0.0.1:1234`다. 최신 MLX 런타임에서 TranslateGemma의 동시 처리 수를 4로 설정하는 구성을 먼저 권장한다. 모델 복제 4개 구성도 아래에서 설명한다.
5. `curl http://127.0.0.1:1234/v1/models`에서 두 모델의 실제 ID를 확인한다. 아래 `<...>`를 그 값으로 바꾸고 실행한다.

```bash
bash "$HOME/Downloads/MyVote-Speaker-Update/start-update.command" \
  "$HOME/Downloads/MyVote-Mac-Demo" \
  --translation-profile translategemma \
  --llm-model "<TranslateGemma의 실제 ID>" \
  --orchestrator-model "<Qwen의 실제 ID>" \
  --translation-workers 4
```

기존 겹침 분리 모델도 함께 쓸 때는 다음 명령을 사용한다.

```bash
bash "$HOME/Downloads/MyVote-Speaker-Update/start-update.command" \
  "$HOME/Downloads/MyVote-Mac-Demo" \
  --translation-profile translategemma \
  --llm-model "<TranslateGemma의 실제 ID>" \
  --orchestrator-model "<Qwen의 실제 ID>" \
  --translation-workers 4 \
  --experimental-overlap-model "$HOME/Downloads/pytorch_model.bin" \
  --overlap-threads 1
```

`검증한 업데이트`와 `gateway.listening` 로그의 `engine_revision`이 **`semantic-translation-2026-09-16`**인지 확인한다. 서버 터미널은 실행해 둔다. 연결 주소의 문서 예시는 `192.0.2.10:50051`이며 실제 Mac LAN 주소로 바꾼다.

## Windows도 새 버전으로 실행

원문 미리보기 수신 기능이 추가되어 **이번에는 Windows와 Mac을 모두 업데이트**해야 한다.

```text
%USERPROFILE%\PycharmProjects\MyVote\artifacts\windows-semantic-translation-2026-09-16\app\MyVote.Desktop.exe
```

설치 파일은 같은 폴더의 `MyVote-Setup-0.1.0-win-x64.exe`다. 앱 폴더로 옮겨 실행할 때는 EXE 하나가 아니라 `app` 폴더 전체가 필요하다.

새 앱과 새 서버가 연결할 때 `semantic_translation_v1` 기능을 협상한다. 예전 클라이언트는 기존 번역 경로로 연결되므로 이번 동작을 확인하려면 새 앱을 실행해야 한다. 기본 앱과 오버레이의 기존 읽기 유지 기능도 포함한다.

## 영상으로 확인

1. 새 앱에서 기존 연결·인증서 설정을 확인하고 **자막 시작 → 자막 창 열기**를 누른다.
2. 영상을 재생한다. 원문이 먼저 **전사 중 / 문맥 확인 중** 영역에 나타나는지 본다.
3. 모델이 선택한 구간의 번역 또는 동일 언어 보정이 확정 자막으로 나타나는지 본다. 처리된 원문은 미리보기에서 빠져야 한다.
4. 영상을 일시정지하거나 **중지**해 마지막 확정 원문도 처리되는지 확인한다.
5. 기존 테스트 영상 `https://www.youtube.com/watch?v=YlgFfqaJ-J0`의 약 40분대도 확인한다. 이번 변경은 구간 판단·표시 흐름 개선이며 화자 구분 정확도 향상을 입증한 변경은 아니다.

세션 `events.jsonl`에서 다음 항목을 확인할 수 있다.

- `session.started.data.semantic_translation_status`: `enabled`
- `transcript.preview`: 실시간 원문 미리보기와 revision
- `caption.source`: 모델이 선택한 실제 원문, `semantic_request_id`, `semantic_force_flush`
- `translation.completed`: 선택 구간의 결과
- `translation.failed`의 `semantic_*`: 모델 JSON 오류·시간 초과 등
- `semantic_buffer_wait_ms`, `semantic_model_ms`, `semantic_total_ms`: 전사 확정 이후 수집 대기와 모델 처리 시간
- `semantic_selection_ms`, `semantic_translation_ms`: 오케스트레이터 선택과 결과 생성의 각 처리 시간(해당 단계의 큐 대기 포함)
- `server_ingress_age_ms`, 기존 `source_timing`: 서버에 들어온 오디오 기준 지연을 별도로 확인하는 값

`translation_budget_scope=semantic_stable_source_clock`은 서버의 전사 확정 시각 기준이다. 이전 버전의 오디오 수신 기준 2.5초 예산과 직접 비교하지 않는다.

## LM Studio 모델을 바꿀 때

모델 ID를 아래 API 목록에서 확인한다.

```bash
curl http://127.0.0.1:1234/v1/models
```

번역 모델은 `--llm-model`, 오케스트레이터는 `--orchestrator-model`로 지정하고 서버를 재시작한다. TranslateGemma에는 전용 `--translation-profile translategemma`가 필요하다. 모델 ID만 바꿔 범용 채팅 프롬프트를 TranslateGemma에 보내는 방식은 사용하지 않는다. 오케스트레이터는 **기다림/구간 선택 JSON**을 정확하게 반환해야 한다. 모델별 지시 준수·언어 품질·실제 처리 시간은 다르므로 동일 영상으로 다시 확인한다. ASR·화자·겹침 분리 모델은 별도 구성 요소다.

### TranslateGemma 인스턴스 4개를 각각 로드했을 때

인스턴스별로 서로 다른 ID를 지정한다. 위 실행 명령에 다음 옵션을 추가하면 각 인스턴스가 작업 하나씩 맡는다. 실제 ID로 바꿔 입력한다.

```bash
  --translation-model-id "<TranslateGemma 인스턴스 1 ID>" \
  --translation-model-id "<TranslateGemma 인스턴스 2 ID>" \
  --translation-model-id "<TranslateGemma 인스턴스 3 ID>" \
  --translation-model-id "<TranslateGemma 인스턴스 4 ID>"
```

모델을 한 번 로드하고 동시 작업만 4개로 쓸 때는 이 반복 옵션이 필요 없다. 두 방식 모두 같은 Mac GPU를 공유하므로 실제 지연을 비교해 선택한다.

## 검증 범위

로컬 자동 검사는 원문 누적, 연속 구간 선택, 요청 중 원문 추가, 실패 시 원문 보존, 시간 제한, 전송 계약과 화면 상태를 확인한다. 시험용 모델 응답을 사용한 검사는 실제 LM Studio 번역 품질을 평가한 것이 아니다. 이 ZIP을 Mac에 반영한 뒤 실제 모델과 영상에서의 품질·지연을 확인해야 한다.
