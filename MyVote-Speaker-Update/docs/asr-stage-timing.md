# 전사에서 자막 원문 생성까지의 지연 기록

`history-latency-2026-09-16` 배포에 포함한다. 이전 `context-refinement-2026-09-15` ZIP에는 없는 변경이다. 이 문서의 `../artifacts/` 검증 링크는 개발 PC의 프로젝트에서만 열리며 검증 폴더 자체는 Mac 업데이트 ZIP에 포함하지 않는다.

## 왜 추가했는가

종료된 실제 Mac 사용 기록에서 1,009개 원문 중 46개가 번역 제출 전에 2,500ms 예산을 소진했다. 기존 `asr_ms`는 전사 작업 시작 이후의 누적 시간이며, 가장 가까운 `transcript.updated`만으로 특정 자막을 확정한 전사 창을 식별할 수 없었다. [원래 기록 분석](../artifacts/live-latency-audit-2026-09-16/README.md).

새 기록은 원문·번역 정책이나 마감을 바꾸지 않는다. 기존 `transcript.updated`/`asr.failed`에 `asr_timing`, `caption.source`에 `source_timing`을 추가한다. 새 이벤트 종류나 추가 모델 호출은 없다. 기존 클라이언트는 추가 필드를 무시하고 원문·화자·시각을 그대로 처리한다. 전체 메타데이터는 `events.jsonl`에 남고 화면에는 표시하지 않는다.

## 어떤 창이 이 자막을 확정했는가

`source_timing.commit_window.window_id`는 **그 자막의 마지막 단어를 실제로 안정화한 창**이다. `emitting_window_id`는 **자막 묶음을 내보내게 한 창**이다. 창 ID는 한 세션의 주 전사 작업에서 증가하며, 원음 시각은 별도 `window_start_ns`/`window_end_ns`에 있다.

예를 들어 두 번째 전사 창에서 단어가 합의됐지만 짧은 구절이어서 세 번째 창까지 기다렸다면, `commit_window.window_id=2`, `emitting_window_id=3`이 된다. 뒤의 창 처리 시간을 그 단어의 전사 비용으로 잘못 연결하지 않는다. 한 창에서 여러 자막이 나오거나 마지막 모델 호출이 실패해 이전 합의 단어만 내보낼 때도 같은 규칙을 유지한다.

`commit_reason`은 실제 안정화 이유 `agreement` 또는 `flush`를 보존한다. 끝점에서 최신 미확정 접미부를 내보내는 `flush`를 두 번의 모델 합의로 표시하지 않는다.

## 필드 해석

모든 `*_ms`는 이름에 명시된 두 지점 사이의 시간이다. `clock_scope=server_process_monotonic`은 같은 엔진 프로세스의 단조 시계를 뜻한다. Windows 캡처 시계와 Mac 시계를 빼지 않는다.

| 필드 | 의미 |
|---|---|
| `commit_window.queue_ms` | 해당 PCM 창이 실제 대기열에 들어간 뒤 주 전사 작업이 시작되기까지 |
| `commit_window.pre_dispatch_ms` | 주 전사 작업 시작부터 native 작업 제출까지 |
| `commit_window.dispatch_wait_ms` | 제출부터 동기 전사 함수 진입까지. executor와, 사용하는 경우 공유 ASR 자원 대기를 함께 포함 |
| `commit_window.transcriber_call_ms` | 동기 전사 어댑터 호출 시간. 전처리·모델·출력 변환·fence 비용을 포함하며 순수 신경망 시간은 아님 |
| `commit_window.resume_wait_ms` | 동기 함수 종료 후 async 호출자가 다시 실행되기까지 |
| `post_asr_to_commit_ms` | async 복귀부터 해당 단어의 안정화 확인까지 |
| `latest_word_commit_to_source_ms` | 마지막 단어 안정화부터 원문 이벤트 데이터 생성까지 |
| `first_word_commit_to_source_ms` | 자막 첫 단어 안정화부터 원문 이벤트 데이터 생성까지. 앞 행과 겹치므로 더하지 않음 |
| `ingress_to_commit_window_queue_ms` | 자막 마지막 원음 프레임의 실제 서버 수신부터 확정 창이 대기열에 들어가기까지 |
| `ingress_to_latest_word_commit_ms` | 같은 실제 수신부터 마지막 단어 안정화까지 |
| `ingress_to_source_created_ms` | 같은 실제 수신부터 원문 이벤트 데이터 생성까지 |
| `source_playhead_lag_ms` | 자막 끝과 출력 창 끝의 **원음 시간축상 간격**. 처리 시간 측정값이 아님 |

`ingress_to_commit_window_queue_ms`에는 창 준비, 앞선 전사/합의 반복, VAD·입력 처리 등 여러 원인이 섞일 수 있다. 이를 순수 ‘음성 감지 대기’ 또는 ‘안정화 모델 시간’으로 이름 붙이지 않는다. 네트워크 수신 시각 이력이 없거나 만료·불연속·미래 시각으로 유효한 기준점을 만들 수 없으면 관련 값은 `null`이며 새 기준 시각을 만들어 대신 쓰지 않는다.

일반적인 새 전사 호출에서 다음 항목의 합은 `ingress_to_source_created_ms`와 같다. 반올림 오차는 허용한다.

```text
ingress_to_commit_window_queue_ms
+ queue_ms + pre_dispatch_ms + dispatch_wait_ms
+ transcriber_call_ms + resume_wait_ms
+ post_asr_to_commit_ms + latest_word_commit_to_source_ms
```

`commit_window.processing_ms`, `ingress_to_latest_word_commit_ms`, `first_word_commit_to_source_ms`는 겹치는 누적/구간 값이므로 이 합에 추가하지 않는다.

## 캐시·실패·대기열 교체

- 동일 PCM의 마지막 flush는 추가 모델 호출을 하지 않는다. `cached_from_window_id`와 `cached_hypothesis`에 마지막 실제 디코드 ID와 시간을 보존한다. 새 호출이 없는 창의 `transcriber_call_ms`/`post_asr_to_commit_ms`는 `null`이다. 0ms 모델 추론으로 집계하지 않는다.
- 모델 예외에도 함수 종료와 async 복귀 시각을 기록한다. 최종 호출 실패 뒤 이전 합의 단어를 살릴 때는 이전의 확정 창과 실패한 출력 창을 구분한다.
- 같은 세그먼트의 대기 PCM을 최신 창으로 교체하면 새 창이 들어온 시각으로 기록한다. 이전 창의 대기 시간을 아직 도착하지 않았던 음성에 붙이지 않는다. 과부하 제거와 종료 시 보조 대기열 기록도 제거한다.
- 단어별 확정 기록은 아직 자막으로 소비되지 않은 안정 단어에만 남긴다. 출력 묶음과 시간 기록을 함께 소비해 중간 sink 예외 뒤 다른 자막에 오래된 시간을 붙이지 않는다.

## 검증 범위와 남은 측정

[개발 소스 검사 기록](../artifacts/asr-stage-timing-2026-09-16/integration-final/summary.json)은 관련 **248개 검사, 실패·오류·skip 0개, native exit 0**이다. 합의 후 구절 대기, 캐시 끝점, 변경 없는 가설, 실패한 끝점, 여러 자막, 출력 중 예외, 대기열 교체·제거, 원래 ingress 마감, 실제 localhost mTLS JSON 전달과 기존 원격 자막 projection을 확인한다. 전사와 번역 응답은 제어된 시험용이다.

이 변경 자체가 처리 속도를 높이거나 이미 누락된 번역을 복구하지는 않는다. `source_created` 이후 원문 이벤트 전달, 화자/겹침 등록, 번역 제출·공유 LLM 자원 대기·모델 응답·Windows 표시까지는 별도 구간이다. 기존 `provider_elapsed_ms`에는 여러 대기가 포함된다. 현재 추가한 주 전사 시간으로 겹침 분리의 두 전사 트랙이나 문맥 재검토 작업의 지연까지 측정했다고 주장하지 않는다. 실제 새 Mac 실행과 동시 부하 측정이 남아 있다.
