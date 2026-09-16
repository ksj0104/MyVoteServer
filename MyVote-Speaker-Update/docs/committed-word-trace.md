# 선택적 확정 단어 시각 기록

`caption_words_v1`은 앞으로 녹음할 평가 자료에 **실제로 자막을 구성한 ASR 단어와 원래 시각**을 남기는 선택 기능이다. 단어 정렬·재전사·화자 추론·자막 분할을 추가로 실행하지 않는다. 기존 Windows 요청은 `caption_groups_v1`만 유지하므로 기본 동작에는 기록이 붙지 않는다.

## 프로토콜

인증된 클라이언트가 `OpenSession.capabilities`에 `caption_words_v1`을 요청하고 서버가 `session.started.capabilities`에 같은 값을 반환한 경우에만 활성화한다. protobuf 변경은 없다. `GatewayClient.open(capabilities=("caption_words_v1",))`처럼 명시적으로 요청할 수 있다. 이 기능만 요청해도 겹침 음성 분리가 활성화되는 것은 아니다.

원래 `caption.source` 이벤트의 `data`에 다음 `word_trace`만 추가한다. 이벤트 종류나 개수, 자막 ID·source revision·텍스트·구간·번역 요청은 바꾸지 않는다.

```json
{
  "schema": "myvote.committed_words",
  "schema_version": 1,
  "origin": "stable_word_assembler",
  "source_revision": 1,
  "stabilizer_revision": 2,
  "commit_reason": "agreement",
  "window_start_ns": 0,
  "window_end_ns": 1024000000,
  "emit_start_ns": 0,
  "words": [
    {"index": 0, "text": "Hello", "start_ns": 100000000, "end_ns": 300000000},
    {"index": 1, "text": " world.", "start_ns": 300000000, "end_ns": 450000000}
  ],
  "boundary_eligible": true,
  "boundary_ineligible_reasons": []
}
```

세션·자막 ID는 바깥 이벤트에, track·capture epoch와 자막 전체 시각은 원래 `caption.source.data`에 있다. `stabilizer_revision`은 단어 확정기 내부 revision이며 자막 source revision과 다르다. `window_*`와 `emit_start_ns`는 이 단어 묶음이 발행된 분석 창과 발행 경계다. 각각의 단어는 기존 `TimedWord`의 문자열과 정수 nanosecond를 그대로 복사한다. 문자열을 순서대로 연결하고 양끝 공백만 제거한 결과가 원문 자막과 같아야 한다.

`boundary_eligible`은 시각 구조가 후속 경계 평가에 적합한지 나타낼 뿐, ASR 또는 화자 정답을 보장하지 않는다. 원래 단어 시각의 다음 문제는 잘라내거나 보정하지 않고 그대로 기록한다.

- `zero_duration_word`: 길이 0인 단어.
- `overlapping_word_times`: 앞선 단어와 시간이 겹침.
- `word_outside_caption`: 발행 자막 구간 밖의 시각.
- `word_outside_commit_window`: 발행 시점 분석 창 밖의 시각.

이 경우 `boundary_eligible=false`가 된다. 단어 순서가 거꾸로이거나 source 텍스트가 일치하지 않는 등 trace 자체가 유효하지 않으면 전체 trace를 생략한다.

## 제한과 생략

- 자막당 최대 128개 단어.
- trace를 포함한 전체 `caption.source.data`를 UTF-8 JSON으로 직렬화한 크기: 서버 이벤트 제한과 64KiB 중 작은 값 이하.
- 세션 전체 추가 trace 바이트: 최대 1MiB. JSON 필드명과 구분 문자까지 포함한 원래 payload 대비 증가량을 센다.
- 제한을 넘으면 trace를 조각내거나 자막을 취소하지 않고 **trace 전체만 생략**한다. 원래 자막과 번역 경로는 계속 실행한다.

활성화한 세션의 최종 `counts`에는 `word_trace_emitted`, `word_trace_bytes`, `word_trace_omitted_invalid`, `word_trace_omitted_word_limit`, `word_trace_omitted_event_limit`, `word_trace_omitted_session_limit`이 들어간다. 모델·원본 PCM·임베딩은 기록하지 않는다. 단어 문자열은 이미 같은 이벤트에 실리는 전사 텍스트의 구성 요소다.

`commit_reason`은 해당 자막 발행을 일으킨 TranscriptUpdate의 사유다. `flush`는 종료 시 최신 단어를 내보낸 것이며 반복 추론의 일치와 다르다. 한 묶음 안의 각 단어가 개별적으로 같은 횟수의 일치 검사를 받았다는 뜻은 아니다.

`live.py --record-word-timings`는 선택 capability를 요청하고 이벤트 journal에 원래 데이터를 저장한다. 서버가 이 기능을 협상하지 않으면 오디오 캡처·전송을 시작하기 전에 중단하며 `word_trace.unavailable`을 기록한다. 기본 데스크톱과 WPF에는 trace 전달이나 표시를 추가하지 않았다. `CaptionStore` snapshot/SRT/VTT는 계속 자막 단위 자료이며 원본 trace 보존 위치는 wire event journal이다.

## 검증 범위

검사는 실제 로컬 mTLS 네 경우, 원래 단어 시각·문자열, flush 사유, JSON 바이트/세션 한도, 전체 trace 생략, 동일 ASR 호출·자막 snapshot·capture 기준 번역 deadline을 포함한다. 실제 live journal 기록과 미지원 서버에서 캡처 전 중단하는 경우도 검사한다. mTLS 검사에는 주입한 ASR·번역 fixture를 사용했으며 실제 음성 모델 품질을 측정하지 않았다. Mac 서버 배포·실제 영상의 새 단어 시각 수집은 별도 단계다.
