# 첫 데모 실행 안내

공개용 초기 데모 안내다. Mac `192.0.2.10`·Windows `192.0.2.20`은 문서 전용 예시이며 실제 접속 주소가 아니다. 운영 설정 `server.toml`은 Git에서 제외한다. 새 환경은 [server.example.toml](server.example.toml)을 복사하고 실제 LAN 주소·인증서·모델을 별도로 준비해야 한다. 최신 정책은 [README](README.md), 공개·비공개 파일 구분은 [Git 배포 안내](docs/git-publishing.md)를 확인한다.

2026-09-15. Windows에서 재생하는 영상의 소리를 같은 LAN의 Mac으로 보내
원문·번역·화자 정보를 Windows 자막 창에서 확인하는 개발용 데모다.
실제 Mac의 모델 추론·전체 지연·GUI 조작은 아직 검증 전이다.

## 준비물

- Windows: 새 데모 실행 파일과 개인 설정 바로가기. 별도 Python·AI 모델 설치는 필요 없다.
- Mac: Apple Silicon, macOS 14 이상, native arm64 Python 3.12 또는 3.13, LM Studio.
  Python은 [공식 Mac 다운로드](https://www.python.org/downloads/macos/)에서 준비할 수 있다.
- 같은 LAN 연결. 서버 주소의 문서 예시는 `192.0.2.10`이다. 실제 Mac LAN 주소를 사용하며 공유기 관리 주소와 혼동하지 않는다.
- Mac으로 옮길 `MyVote-Mac-Demo.zip`. 이 개인용 ZIP에는 서버 인증서와 서버 개인 키가 들어 있다.
  해당 Mac에만 옮긴다. Windows용 개인 키와 CA 발급 키는 ZIP에 넣지 않는다.

## 1. Mac의 LM Studio 설정

1. LM Studio에서 `Qwen3.5-4B`를 검색한다.
2. `lmstudio-community/Qwen3.5-4B-GGUF`의 **Q4_K_M**을 받는다. 가중치 약 2.71GB다.
3. 모델을 로드하고 **Enable Thinking을 끈다**. 서버에서 사용하는 모델 설정에도 적용돼야 한다.
4. 문맥 길이는 초기 실험값 **4096 tokens**로 둔다.
5. **Developer → Start Server**를 실행한다. 주소는 `http://127.0.0.1:1234`다.

LM Studio는 Mac 내부에서만 호출한다. Windows는 별도의 MyVote 서버 50051번에 연결하므로,
이 구성에서 LM Studio의 LAN 공개 옵션은 필요하지 않다. 현재 데모는 LM Studio의 인증을 요구하지 않는
loopback API를 사용한다. 인증을 켠 서버의 토큰 주입은 아직 지원하지 않는다.
[모델과 Thinking 설정](https://lmstudio.ai/models/qwen/qwen3.5-4b),
[모델 파일](https://huggingface.co/lmstudio-community/Qwen3.5-4B-GGUF/tree/main),
[LM Studio 서버 안내](https://lmstudio.ai/docs/developer/core/server).

## 2. Mac 음성 서버 준비·실행

`MyVote-Mac-Demo.zip`을 Mac의 다운로드 폴더에 옮겨 압축을 푼다.
`MyVote-Mac-Demo` 폴더 안에 `start-demo.command`, `scripts`, `models`, `connection`이 있어야 한다.

Mac 터미널에서 처음 한 번:

```bash
cd "$HOME/Downloads/MyVote-Mac-Demo"
bash start-demo.command --install --download-asr --experimental-speakers
```

이 명령은 다음을 수행한다.

- 이 폴더 안에 `.venv`를 만들고 음성 엔진의 Python 의존성을 설치한다.
- `mlx-community/whisper-large-v3-turbo`를 `models/whisper-turbo`에 받는다. 약 1.61GB이며 인터넷이 필요하다.
- 동봉된 Silero VAD·pyannote segmentation·WeSpeaker ONNX를 사용한다.
- Mac의 주소·파일과 LM Studio 모델 목록을 확인하고, 같은 서버 프로세스에서 준비용 추론 후 서버를 시작한다.

`--install`과 `--download-asr`를 주지 않은 실행은 패키지·모델을 자동으로 받지 않는다.
기존 Whisper 모델을 덮어쓰지 않는다. 이후 실행은 아래 명령만 사용한다.

```bash
cd "$HOME/Downloads/MyVote-Mac-Demo"
bash start-demo.command --experimental-speakers
```

**`gateway.listening`과 `port: 50051`이 표시되면 Windows에서 시작한다.**
모델 준비 중에는 아직 연결할 수 없다. 이 터미널과 LM Studio를 켜 두며 Mac을 잠자기 상태로 두지 않는다.
서버 종료는 이 터미널에서 `Ctrl+C`다.

준비용 추론은 1초 무음과 짧은 진단 문장을 사용한다. 사용자 자막으로 전송하지 않으며,
준비 성공을 실제 영상의 전사·번역 품질 검증으로 보지 않는다.

모델 ID가 여러 개여서 자동 선택되지 않으면 표시된 정확한 ID를 넣는다.

```bash
bash start-demo.command --experimental-speakers --llm-model 'LM_STUDIO가_표시한_정확한_ID'
```

Mac IP가 달라졌다면 실행 인자뿐 아니라 Windows 대상 주소와 인증서 SAN도 맞춰야 한다.
별도로 받은 개인 묶음의 인증서 SAN이 실제 서버 주소에 맞는지 확인한다. 문서 예시 `192.0.2.10`은 발급 대상이 아니며, 주소가 바뀐 상태에서 검증을 끄고 연결하지 않는다.
Mac이 Python의 수신 연결 허용을 물으면 이 데모 서버에 대해 허용한다.
필요한 경우 시스템 설정 → 네트워크 → 방화벽 → 옵션에서 해당 Python 앱의 수신 연결을 확인한다.
[Apple 방화벽 안내](https://support.apple.com/guide/mac-help/block-connections-to-your-mac-with-a-firewall-mh34041/mac).

## 3. Windows에서 확인

개인 데모 폴더의 **Start-Windows-Demo** 바로가기를 연다. 주소와 인증서 경로가 미리 입력된다.
기본 시연 언어는 원문 자동 감지 → 한국어이며 앱에서 변경할 수 있다. 사용자 언어 선택을 확정한 설정은 아니다.

1. Mac에서 `gateway.listening`이 표시됐는지 확인한다.
2. Windows 앱에서 **시작**을 누른다.
3. 처음에는 음악이 적고 한 사람이 분명히 말하는 짧은 영상을 재생한다.
4. 원문과 번역이 들어오는지, 자막 창의 크기·위치를 바꿀 수 있는지 확인한다.
5. 두 사람이 번갈아 말하는 부분에서 자동 화자 ID와 미확정 표시를 확인한다.
6. 화자 이름을 바꾸고, 잘못 배정된 구간을 수동으로 지정한다.
7. **중지**한 뒤 SRT·VTT를 내보내 마지막 자막과 수동 화자 이름이 남는지 확인한다.

앱의 기본 시작 메뉴 항목은 기존 개인 설정을 사용할 수 있다. 이번 연결 정보를 쓰려면 생성된
데모 바로가기를 사용한다. 개인 설정 파일을 옮기면 인증서 절대 경로와 바로가기도 다시 만들어야 한다.

## 화자 실험 설정의 의미

`--experimental-speakers`는 포함된 고정 실험 프로필을 명시적으로 적용한다.
서버의 기본 임계값은 바꾸지 않는다. 이전 공개 회의의 새 120초 구간에서는 네 화자 중 세 명을 확정했고,
단독 발화 시간의 49.60%는 미확정이었다. 영상·강의·방송에서 같은 정확도가 나온다는 뜻은 아니다.
짧은 발화나 겹친 목소리는 늦게 확정되거나 미확정으로 남을 수 있다.

번역은 화자 확정을 기다리지 않고 먼저 표시한다. 자동 ID는 사람의 실명을 알아내는 기능이 아니며,
이름과 구간은 사용자가 수정할 수 있다.

## 이번 데모의 범위

- 실제 Mac과 연결된 뒤 확인해야 하는 **개발용 데모**다. M4 Max 지연·전체 화자 품질을 검증한 릴리스가 아니다.
- 한 Windows 클라이언트·한 세션이며 현재 세션 시간 제한은 1시간이다. 2시간 이상 방송 지원은 후속 작업이다.
- 지원 서버는 연결 유지 메시지를 광고한다. 새 Windows 클라이언트는 오디오가 없을 때 이 메시지를 보내며, 가짜 무음 PCM을 만들지 않는다.
- 자동 재연결·장치 변경 복구·앱 재시작 후 기록 복원은 미완성이다. 연결이 끊기면 이미 받은 결과를 보존하고 새 세션으로 다시 시작한다.
- Windows GUI 자동 점검 도구가 native pipe 오류로 연결되지 않아 실제 화면 조작은 아직 확인하지 못했다.
- 이번 패키지의 설치·통신 검사와 실제 모델·GUI 확인 결과는 서로 구분해 기록한다.

문제가 생기면 Mac 터미널의 마지막 오류 분류와 Windows 세션 결과를 확인한다.
`server.key`, `client.key`, `ca.key` 파일 내용은 진단에 필요하지 않다.
