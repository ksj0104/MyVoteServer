# 별도 ASR 텍스트 API 운영 안내

기본 주소는 `http://127.0.0.1:8000`이다. 오디오가 아닌 외부 ASR 텍스트를 받으며 기존 Windows/gRPC 50051 앱을 자동 연결하지 않는다. [구조·범위](streaming-architecture.md), [WebSocket 계약](streaming-protocol.md)을 함께 읽는다.

현재는 임시 서버에서 검증을 마친 상태이며 **8000번 상시 서비스는 실행하지 않았다**. 운영할 관리 키를 사용자가 설정한 뒤 아래 명령으로 시작해야 한다. 기존 50051 음성 서버와 LM Studio 1234번은 그대로 유지했다.

## 설치와 시작

프로젝트 폴더에서 기존 `.env`가 없을 때 예제를 복사하고, 관리용 무작위 비밀을 생성해 `.env`의 `STREAMING_API_KEY`에 저장한다. 24자 미만이거나 비어 있으면 서버 시작을 거부한다. 생성한 키·세션 토큰을 저장소나 로그에 넣지 않는다.

```bash
cp -n .env.example .env
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
python3 -m venv .venv-streaming
.venv-streaming/bin/python -m pip install -r requirements.txt
.venv-streaming/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

키 생성 명령의 결과를 직접 `.env`에 넣은 다음 서버를 시작한다. `Settings`가 `.env`를 읽으며 기존 환경변수가 있으면 그 값을 우선한다. `.venv-streaming`은 기존 `MyVote-Mac-Demo/.venv`와 별도다. 이 명령들은 50051 음성 서버나 LM Studio를 재시작하지 않는다.

`requirements.txt`는 `requirements-lock.txt`를 constraints로 참조한다. 둘을 함께 보관해야 확인된 버전으로 설치되며, 플랫폼별 선택 의존성은 해당 플랫폼에서 지원될 때만 설치된다.

`TRANSLATION_BASE_URL=http://127.0.0.1:1234/v1`, `TRANSLATION_MODEL=google/gemma-4-26b-a4b`가 기본이다. 해당 모델은 LM Studio에 이미 로드되어 있어야 한다. 새 API의 범용 채팅 백엔드는 `/v1/chat/completions`를 사용하며, 기존 TranslateGemma 전용 프롬프트·어댑터를 바꾸지 않는다.

## REST 계약

| 메서드·경로 | 인증 | 결과 |
| --- | --- | --- |
| `POST /sessions` | 관리 키 Bearer | 201, 세션 ID·세션 토큰·WebSocket 경로 |
| `GET /sessions/{id}` | 해당 세션 토큰 Bearer | 현재 `session_state` 스냅샷 |
| `DELETE /sessions/{id}` | 해당 세션 토큰 Bearer | 204, 세션 정리·작업 취소 |
| `POST /sessions/{id}/glossary` | 해당 세션 토큰 Bearer | 용어집 갱신; 향후 번역 작업에 적용 |
| `GET /health` | 없음 | 프로세스 liveness; 모델 추론·품질 검증 아님 |
| `GET /metrics` | 관리 키 Bearer | Prometheus 텍스트 형식 운영 지표 |

Bearer 헤더는 `Authorization: Bearer <credential>`다. 관리 키는 세션 REST/WebSocket의 대체 토큰이 아니다. 세션 토큰 오류와 없는 세션은 세션 REST에서 404로 처리할 수 있어 해당 세션의 존재를 인증 없이 확인할 수 없다. `/docs`와 `/openapi.json`은 실행한 FastAPI의 스키마 참고용이다.

세션 생성 요청 예시:

```json
{"source_language":"en","target_language":"ko","translation":{"temperature":0.1,"max_context_segments":3},"streaming":{"max_latency_ms":1500,"stability_threshold":0.75}}
```

빈 객체 `{}`도 기본 영어→한국어 설정으로 유효하다. 예시 응답의 토큰은 실제 자격증명이 아닌 자리표시자다.

```json
{"session_id":"<session_id>","session_token":"<secret_session_capability>","websocket_path":"/ws/translate/<session_id>","protocol_version":"1"}
```

원문 언어는 `en`·`ko` 계열만 지원한다. 다른 원문 언어로 세션을 만들면 HTTP 422 `UNSUPPORTED_SOURCE_LANGUAGE`가 되며 목표 언어 지원은 백엔드에 달려 있다. `translation.model`은 서버에서 허용한 모델 ID만 가능하다. `temperature`는 0~0.2, 문맥 구간 수는 0~10, `streaming.stability_threshold`는 0~1이다. 배포 설정의 제한은 클라이언트 입력보다 우선한다. 입력 스키마 오류는 422, 큰 HTTP 본문은 413, 세션 상한 도달은 429다.

용어집 요청은 `{"entries":{"release":"배포"}}` 형태다. 최대 1000개, 키 1~128자, 값 1~256자, 공백만 있는 항목은 거부하고 키·값 전체 합은 64000자 이하여야 한다. 응답은 `entries` 개수와 `applies_to="future_translation_jobs"`를 알려준다.

`GET /health`의 `status="ok"`는 서버가 응답한다는 뜻이다. `backend_quality_verified=false`, `storage="in_memory"`, `gateway="asr_text"`를 함께 제공하며 모델이 로드되었거나 번역할 수 있다는 보장은 아니다.

## 클라이언트와 합성 ASR

두 도구는 프로젝트 폴더의 `.env` 또는 환경변수에서 키를 읽으며 기본적으로 텍스트 대신 안전한 이벤트 요약만 출력한다. 실제 번역문을 터미널에 표시하려면 `--show-text`를 명시한다. 아래 테스트는 실제 모델 부하를 만들 수 있으므로 기존 음성 세션과 동시에 실행할 때 주의한다.

```bash
.venv-streaming/bin/python scripts/client.py --text "Please close the window." --show-text
.venv-streaming/bin/python scripts/client.py
.venv-streaming/bin/python scripts/simulate_asr.py --interval 150 --url http://127.0.0.1:8000 --show-text
.venv-streaming/bin/python scripts/simulate_asr.py --interval 150 --disconnect-after 2
```

대화형 클라이언트는 입력 한 줄을 새로운 final ASR 발화로 전송한다. `:reset`은 문맥과 결과를 초기화하고 `:quit` 또는 EOF로 종료한다. 시뮬레이터는 같은 `utterance_id`의 누적 partial을 150ms 간격으로 보내고 마지막에 final을 보낸다. 오디오를 캡처하거나 ASR 모델을 실행하지 않는다.

도구는 마지막 final 입력 번호 이상의 스냅샷, 해당 발화 구간, `final=true`를 모두 확인한 뒤 성공 종료한다. 기본 제한은 90초이며 `--timeout`으로 바꿀 수 있다. 미완성 원문·모델 오류·시간 초과는 비정상 종료한다. 이것은 번역 품질 평가나 기존 Windows 앱 E2E 테스트가 아니다.

`SOURCE_RECONCILIATION_PENDING`은 기존 확정 원문과 충돌한 partial이므로 대기 안내만 표시한다. ASR 제공자는 같은 `utterance_id`의 명시적 final 수정으로 확인해야 한다. 이 안내를 받았다고 과거 입력을 재전송하지 않는다.

네트워크 단절은 기본 2회까지 재연결·재인증하고 새 스냅샷을 읽는다. 이미 보낸 ASR은 자동 재전송하지 않는다. `--reconnects 0`으로 복구 시도를 끌 수 있다. `--disconnect-after`는 시뮬레이터가 의도적으로 연결을 끊는 시험 옵션이다. 송신 자체가 중단되어 수락 여부가 모호하면 오류로 종료하며 상태를 확인해야 한다.

도구가 생성한 세션은 정상/오류 종료 시 기본 DELETE한다. `--keep-session`이면 메모리 만료까지 남기지만 도구는 토큰을 파일이나 콘솔에 저장하지 않으므로 잃어버린 토큰을 복원할 수 없다. 기존 세션은 `--session-id`와 `STREAMING_SESSION_TOKEN`으로 접속하며 도구가 자동 삭제하지 않는다. `--api-key`·`--session-token`도 지원하지만 프로세스 인자 노출을 줄이려면 환경변수를 권장한다.

## 배포 한도와 관측

| 기본값 | 의미 |
| --- | --- |
| `MAX_SESSIONS=32` | 동시 메모리 세션 상한 |
| `SESSION_INACTIVITY_SECONDS=1800` | 비활동 세션 만료; 영구 기록 보관 기간 아님 |
| `TRANSLATION_WORKERS=4` | 프로세스 내부 모델 작업 슬롯; Uvicorn worker 수 아님 |
| `MAX_PENDING_JOBS=128`, `MAX_PENDING_PER_SESSION=16` | 전체/세션별 대기 작업 상한 |
| `TRANSLATION_TIMEOUT=30` | 모델 요청 제한, 초 |
| `HOLD_PARTIAL_TAIL=true` | 구분자가 없는 마지막 partial 토큰을 보류; final 또는 다음 구분자로 확인 |
| `MAX_SEGMENT_TOKENS=24`, `ALLOW_OVERSIZE_COMPLETE=true` | 24토큰은 선호 크기; 첫 명확한 완성 문장 전체는 원문/상태 제한 안에서 초과 허용 |
| `MAX_SEGMENT_LATENCY_MS=1500` | 구간 판단의 목표 대기 한도; 번역 완료 SLA 아님 |
| `MAX_BUFFER_AGE_MS=20000` | 미완성 구간 판단 대기 한도; 초과 원문은 상태에 보존하며 기존 gRPC 정책과 별개 구현 |
| `MAX_STATE_BYTES=65536`, `MAX_SNAPSHOT_BYTES=524288` | 원문·번역 상태의 합산 UTF-8 예산 / REST·WebSocket 출력 스냅샷 크기 상한 |
| `RAW_TRANSCRIPT_LOGGING=false` | 원문을 기본 로그에 남기지 않음 |

현재 상태는 프로세스 메모리에만 있으므로 Uvicorn은 **1 worker**로 실행한다. 재시작·삭제·만료 시 세션은 사라진다. 여러 프로세스로 늘리려면 공유 저장소와 세션 소유권·이벤트 복구 설계가 선행되어야 한다. 운영 지표를 수집하더라도 이전 gRPC 서버의 수치를 새 API의 성능 실측으로 해석하지 않는다.

| 지연 지표 (`myvote_` 접두부) | 측정 범위 |
| --- | --- |
| `segmentation_to_queue_ms` | 구간 생성부터 작업 큐 접수까지 |
| `client_queue_wait_ms` | 출력 이벤트가 클라이언트 전송 큐에서 기다린 시간 |
| `client_write_ms` | WebSocket 프레임 쓰기 호출에 걸린 시간만 측정 |
| `model_result_to_client_ms` | `translation_update`가 출력 큐에 들어간 뒤 쓰기 완료까지; 모델 전체 추론 시간 아님 |

위 지표는 실제 오디오 발생부터 Windows 화면에 표시될 때까지의 지연이 아니다. 특히 프레임 쓰기 완료는 클라이언트 렌더링 완료를 뜻하지 않는다.

## 새 API의 실제 모델 진단

실제 ASGI 앱·임시 localhost Uvicorn·HTTPX/WebSocket·기존 Gemma 4 모델을 연결해 **고정 ASR final 입력 5건 모두 성공**을 확인했다. `I want to go to the hospital today.`는 `오늘 병원에 가고 싶어요.`로 번역됐으며 각 세션은 구간 1개로 완료했다.

| 실행 | 관측한 완료 시간 |
| --- | --- |
| 단독 세션 1개 | 1216.6ms |
| 동시 세션 4개 | 1301.7 / 1363.1 / 1341.6 / 1371.9ms |

700ms draft 대기도 포함한 소규모 고정 입력 진단이다. 오디오·실제 ASR·영상·기존 Windows 앱·지속 부하·p95를 측정한 결과가 아니며, 처리량이나 지연을 보장하지 않는다. 진단용 API는 종료했다.

자동 회귀 검사는 **새 서비스 94/94개**, **기존 서버 148/148개**를 통과했다. 새 환경의 `pip check`도 의존성 문제 없이 통과했고 설치 버전은 `requirements-lock.txt`와 일치한다. 재실행 명령은 다음과 같다.

```bash
.venv-streaming/bin/python -B -m unittest discover -s realtime_tests
MyVote-Mac-Demo/.venv/bin/python -B -m unittest discover -s tests
.venv-streaming/bin/python -m pip check
```

모의 백엔드로 CLI 실제 프로세스의 인증·재연결·final 대기·미완성 오류 종료·세션 삭제와 기본 출력의 자격증명/원문/번역문 제외도 확인했다. 코드 검사는 실제 영상 품질 검증을 대신하지 않는다.

## 컨테이너

```bash
docker compose build
docker compose up
```

Compose는 `.env`의 관리 키를 요구하고 API만 시작한다. 호스트 포트는 기본 `127.0.0.1:8000`에만 공개한다. 이미지에는 `app/`·런타임 의존성 정의와 lock 파일만 포함하며 `.env`·ASR·모델·인증서·기존 데모 환경은 넣지 않는다. 비루트·읽기 전용 파일시스템으로 실행하고 `/tmp`만 임시 쓰기 공간을 제공한다. 기본 종료 유예는 모델 요청 제한 30초보다 긴 45초다. 모델 제한을 늘리면 컨테이너 종료 유예도 함께 검토한다.

Docker Desktop의 모델 기본 주소는 `http://host.docker.internal:1234/v1`이다. 별도 위치는 `.env`의 `DOCKER_TRANSLATION_BASE_URL`로 지정한다. 호스트 모델 API가 컨테이너에서 접근 가능해야 하며 Compose가 LM Studio의 바인딩을 바꾸지는 않는다. Linux나 별도 Docker 호스트에서는 실제로 도달 가능한 주소·네트워크 구성을 별도로 확인한다.

현재 작업한 Mac에는 Docker 명령이 없어 컨테이너 이미지 빌드·실행은 검증하지 않았다. Python에서의 API/클라이언트 검증과 Docker 런타임 검증은 별개다.

원격 접근이 필요하면 의도적으로 수신/포트 매핑을 변경하고 HTTPS/WSS 프록시와 인증을 함께 구성한다. TLS 검증을 끄거나 세션 토큰을 URL에 넣어 문제를 우회하지 않는다. WebSocket 헤더·인증 프레임·텍스트 본문을 프록시/APM 로그로 수집하지 않는다.

공식 API 참고: [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/), [HTTPX AsyncClient](https://www.python-httpx.org/async/), [websockets asyncio 클라이언트](https://websockets.readthedocs.io/en/14.2/reference/asyncio/client.html), [LM Studio OpenAI 호환 API](https://lmstudio.ai/docs/developer/openai-compat). 컨테이너 구성은 [FastAPI 공식 Docker 안내](https://fastapi.tiangolo.com/deployment/docker/)처럼 의존성과 앱을 구분해 복사하지만, 이것만으로 운영 환경 인증이나 실측 성능을 보장하지 않는다.
