# ASR 텍스트 스트리밍 프로토콜 v1

대상은 새 `app/` 서비스의 `/ws/translate/{session_id}`다. 기존 50051 gRPC 오디오 프로토콜과 호환되지 않는다. 바이너리 오디오가 아닌 **UTF-8 JSON 텍스트 프레임**만 받는다.

## 연결과 인증

먼저 [관리 API](streaming-api.md)로 세션을 만들고 반환된 `session_token`을 보관한다. Python 등 사용자 헤더를 지원하는 클라이언트는 WebSocket 핸드셰이크에 `Authorization: Bearer <session_token>`을 넣는다. 관리 키는 세션 토큰을 대신하지 않는다.

브라우저 WebSocket은 다음 첫 프레임으로 인증할 수 있다. 연결 후 **5초 이내**, 다른 입력보다 먼저 보내며 인증 프레임은 1024바이트 이하여야 한다.

```json
{"type":"authenticate","token":"<session_token>"}
```

토큰을 쿼리 문자열에 넣지 않는다. 브라우저에 관리 키를 배포하지 말고 신뢰하는 서버가 생성한 세션 권한만 전달한다. 인증 성공 시 첫 이벤트는 현재 상태의 `session_state`다. 인증 실패는 WebSocket 종료 코드 1008, 크기 초과는 1009, 느린 소비자 등의 연결 종료는 1013으로 나타날 수 있다.

## ASR 입력은 발화별 누적 텍스트

```json
{"type":"asr_partial","sequence":0,"utterance_id":"utterance-1","text":"The shipment weighs"}
```

```json
{"type":"asr_partial","sequence":1,"utterance_id":"utterance-1","text":"The shipment weighs about twenty"}
```

```json
{"type":"asr_final","sequence":2,"utterance_id":"utterance-1","text":"The shipment weighs about twenty kilograms.","is_final":true}
```

`text`는 새 단어만 붙이는 delta가 아니라 **해당 발화의 최신 전체 전사**다. partial은 같은 `utterance_id`로 이전 가설을 대체한다. ASR 수정으로 기존 단어가 바뀌거나 짧아질 수 있으므로 서버와 클라이언트 모두 단순 이어 붙이기를 하면 안 된다.

| 필드 | 규칙 |
| --- | --- |
| `type` | `asr_partial` 또는 `asr_final` |
| `sequence` | 세션 전체에서 엄격히 증가하는 0 이상의 정수; 중복·역순 거부 |
| `utterance_id` | 명시 권장; 같은 발화 수정은 같은 ID, 새 발화는 새 ID |
| `text` | 누적 원문; 기본 배포 상한 8000자, 스키마 자체의 상한은 32000자 |
| `is_final` | 기본 `false`; `asr_final`은 항상 final로 처리 |
| `language`, `target_language` | 선택 사항; 세션 언어쌍과 일치해야 함 |
| `speaker_id` | 선택 사항; 같은 발화 안에서는 변경 불가 |
| `session_id` | 선택 사항; 있으면 연결한 세션과 일치해야 함 |
| `timestamp`, `pause_ms` | 선택적 시각/정적 메타데이터; 완료 의미나 강제 번역 권한을 대신하지 않음 |
| `tokens` | 선택적 토큰 배열; `text`, `start`, `end`, `confidence`; 시각은 음수가 아니고 end ≥ start |

원문 언어는 `en`·`ko` 계열만 지원하며, 다른 원문 언어로 세션을 생성하면 HTTP 422 `UNSUPPORTED_SOURCE_LANGUAGE`로 거부한다. 목표 언어는 백엔드 능력에 따라 검증해야 한다. 기본 `HOLD_PARTIAL_TAIL=true`는 구분자 없이 끝난 마지막 partial 토큰을 보류한다. 반복 가설에 `hosp`가 나타났더라도 다음 구분자나 final 전까지 완성 단어로 단정하지 않는다.

`utterance_id`를 생략하면 서버가 현재 발화에 배정하고 final 뒤 다음 입력을 새 발화로 취급한다. **final 이후 전사 수정에는 원래 ID를 명시하고 다시 `asr_final`을 보낸다.** 명시 ID 없이 수정하면 새 발화로 처리될 수 있다. 같은 final 발화를 partial로 되돌리는 입력은 `FINAL_REQUIRES_FINAL`로 거부한다.

한 프레임의 기본 상한은 UTF-8 65536바이트다. 세션·원문·이벤트 속도·작업 큐에도 별도 상한이 있으며, 스키마에 맞는 입력도 서버 배포 한도를 넘으면 거부된다. 입력에 정의되지 않은 필드는 허용하지 않는다.

출력 스냅샷의 기본 상한은 REST/WebSocket 모두 524288바이트(512KiB)이며, 원문·번역 상태의 합산 UTF-8 예산은 65536바이트다. 초과는 명시적인 오류로 처리하며 final 이벤트를 조용히 버려서 정상처럼 보이게 하지 않는다. 예제 클라이언트의 WebSocket 수신 상한도 512KiB다.

## 출력은 스냅샷

`session_state`, `translation_update`, `translation_final`은 다음 필드를 가진 **전체 현재 상태**다. 클라이언트는 `full_text`를 계속 덧붙이지 말고 최신 스냅샷으로 교체한다.

| 필드 | 의미 |
| --- | --- |
| `type`, `session_id`, `protocol_version` | 이벤트 종류, 해당 세션, 버전 `"1"` |
| `generation_id` | reset 이후 이전 작업 결과와 구분하는 세대 |
| `revision` | 번역 상태 revision |
| `last_sequence` | 서버가 처리한 마지막 ASR sequence; 다음 입력은 이보다 커야 함 |
| `source_language`, `target_language` | 세션 언어쌍 |
| `committed_source`, `unstable_source` | 확정/미확정 원문 상태 |
| `committed`, `draft`, `full_text` | 확정 번역, 수정 가능한 번역, 현재 표시용 전체 번역 |
| `segments` | 원문·번역의 식별 가능한 구간 목록 |
| `final` | 현재 final 원문들이 해소되고 번역 구간도 모두 확정됐는지 여부 |

각 구간에는 `segment_id`, `utterance_id`, `source`, `translation`, `committed`, `source_revision`, `speaker_id`, `status`가 포함된다. 결과는 이 식별자에 연결하며, 다른 작업의 완료 순서를 화면 순서로 사용하지 않는다. ASR 수정으로 원문 revision이 바뀌면 과거 번역을 현재 원문에 붙여서는 안 된다.

이벤트 이름이 `translation_final`이어도 한 구간만 확정됐을 수 있다. 전체 완료를 기다릴 때는 **`final=true`**, 기대 입력 이상의 `last_sequence`, 필요한 `segments[].utterance_id`를 함께 확인한다. 미완성 final 원문은 `INCOMPLETE_SOURCE` 오류로 원문만 남길 수 있고 전체 `final`은 false일 수 있다. `asr_final`은 ASR의 확정 신호이지 미완성 내용을 억지로 번역하거나 draft 대기 없이 즉시 commit하라는 요청이 아니다. 완결성을 통과한 번역도 기본 700ms draft 대기 정책을 따르며, 동일한 기존 번역이 있으면 final이라는 이유만으로 다시 모델을 호출하지 않는다.

## 제어 메시지

```json
{"type":"ping"}
```

응답은 `{"type":"pong","last_sequence":2}` 형태다. ping은 liveness 용도이며 번역 성공이나 재전송 ACK가 아니다.

```json
{"type":"reset_context"}
```

원문·출력·진행 작업을 초기화하고 `generation_id`를 올린다. `sequence` 카운터는 유지하므로 0부터 재시작하지 않는다. 취소되기 전 모델 작업의 결과가 새 세대에 반영되어서는 안 된다.

```json
{"type":"session_config","source_language":"en","target_language":"ko","translation":{"temperature":0.1},"streaming":{"max_latency_ms":1500,"stability_threshold":0.75}}
```

생성 시와 같은 `SessionConfig` 구조다. 언어쌍을 바꾸려면 reset 후 비어 있는 상태에서 재설정한다. 모델은 서버가 허용한 모델만 사용할 수 있으며 클라이언트 설정으로 서버 자원 한도를 우회할 수 없다. 이 메시지는 임의의 설정 파일 경로나 모델 다운로드를 실행하지 않는다.

```json
{"type":"glossary_update","entries":{"release":"배포","checkpoint":"체크포인트"}}
```

전달한 용어는 기존 용어집에 병합하며 같은 키는 새 값으로 갱신한다. 향후 번역 작업에 적용하고 이미 확정된 번역을 자동으로 소급 재번역하지 않는다. 상세 크기 제한은 API 안내를 따른다.

## 오류와 재연결

오류 이벤트는 `type="error"`, `code`, 안전한 `message`, `recoverable`을 포함하며 경우에 따라 `last_sequence`도 제공한다. 예시:

```json
{"type":"error","code":"STALE_SEQUENCE","message":"Sequence must increase","recoverable":true,"last_sequence":2}
```

중복 또는 역순 sequence는 처리하지 않는다. `recoverable=true`는 수정된 입력을 같은 세션에서 계속 보낼 수 있다는 뜻이지 실패한 입력이 처리됐거나 자동 재전송된다는 뜻이 아니다. 모델 오류·시간 초과·미완성 원문 오류를 성공으로 간주하지 않는다.

| 오류 코드 | 의미 |
| --- | --- |
| `INCOMPLETE_SOURCE` | 미완성 원문 보존, 강제 조각 번역 안 함 |
| `TRANSLATION_TIMEOUT` | 모델 요청 시간 초과 |
| `BACKEND_UNAVAILABLE`, `BACKEND_RATE_LIMITED` | 백엔드 연결/HTTP 오류 또는 백엔드 요청 제한 |
| `BACKEND_INVALID_RESPONSE`, `BACKEND_INCOMPLETE_RESPONSE` | 잘못되거나 정상 완료되지 않은 모델 응답 |
| `BACKEND_RESPONSE_LIMIT`, `SNAPSHOT_LIMIT` | 모델 응답 또는 출력 스냅샷 크기 제한 |
| `SOURCE_LIMIT`, `SESSION_LIMIT` | 원문 또는 세션 상태 자원 제한 |
| `UNSUPPORTED_SOURCE_LANGUAGE` | 지원하지 않는 원문 언어 |
| `TRANSLATION_ERROR` | 위 분류에 포함되지 않는 번역 오류 |

한 세션은 한 번에 WebSocket 하나만 연결한다. 연결이 끊겨도 세션은 메모리에 남아 비활동 만료까지 다시 인증할 수 있다. `GET /sessions/{id}` 또는 새 WebSocket의 첫 `session_state`로 현재 상태를 확인한 뒤 `last_sequence + 1` 이상으로 다음 입력을 보낸다. **이벤트 replay, 입력 deduplication 복구, exactly-once 전송 보장은 없다.** 송신 도중 단절되면 수락 여부가 모호할 수 있으므로 원본 입력을 무조건 재전송하지 않는다.

프로세스 재시작·세션 삭제·만료 후에는 기존 토큰으로 복원할 수 없다. 새 세션을 만들어야 하며 이전 메모리 문맥이나 결과가 이어진다고 가정하지 않는다.
