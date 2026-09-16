# 겹침 음성의 시간·주파수 분리 설계

검토일: 2026-09-15. 이 문서는 논문 근거와 이번에 구현한 실험 경로를 구분한다. 아래 매개변수는 MyVote의 초기 설정이며 한국어 방송이나 M4 Max에서 최적임을 검증한 값이 아니다.

후속 진행: [실시간 두 줄 자막](live-overlap.md) 연결과 [SepFormer 실제 비교](sepformer.md)를 완료했다. 아래 후속 후보·계획은 최초 연구 시점의 기록이며, 최신 배포 안내는 [주파수·화자 실험판](frequency-speaker-preview.md)에 있다.

## 적용 방향

**학습 분리기가 추정한 각 목소리의 시간·주파수 분포로 혼합음을 재배분하는 Wiener-like 전력 비율 마스크를 추가했다.** [frequency_separation.py](../src/myvote_engine/frequency_separation.py)가 이 계산을 수행하고, [overlap_separation.py](../src/myvote_engine/overlap_separation.py)가 실제 ConvTasNet 출력에 적용한다. 기존 `separation_dsp.py`의 혼합 일치 투영과 별도 비교할 수 있다. 실시간 gateway에는 자동으로 켜지지 않는다. 새로운 화자 ID는 주파수나 출력 채널 번호가 아닌, 별도로 검증한 음성 특징과 시간 연속성으로 결정해야 한다.

고정된 저음·고음 대역을 각각 한 사람에게 할당하지 않는다. 같은 순간 두 목소리가 같은 주파수에 있을 수 있고, 한 사람의 배음과 자음도 넓은 대역에 걸친다. 모노 혼합 `x=s₀+s₁` 하나만으로 두 원음을 유일하게 복원할 수 없으므로 학습된 음성 구조가 필요하다. 시간·주파수 마스크는 이 추정을 적용하는 수단이다. 이 설계상의 한계는 아래 위상·도메인 불일치 문제와 함께 평가한다.

## 논문에서 가져오는 기법

| 1차 자료 | 설계에 사용하는 부분 | 적용 한계 |
|---|---|---|
| Liutkus & Badeau, ICASSP 2015, [Generalized Wiener filtering with fractional power spectrograms](https://dihana.cps.unizar.es/proceedings/ICASSP/2015/pdfs/0000266.pdf) | 추정 스펙트럼의 크기에 지수 α를 적용한 비율 마스크. 이번 구현은 제곱 전력인 α=2로 고정한다. | 통계적 가정 아래의 필터다. 부정확한 음원 추정을 정답으로 바꾸지 않는다. |
| Isik et al., Interspeech 2016, [Single-Channel Multi-Speaker Separation using Deep Clustering](https://www.jonathanleroux.org/pdf/Isik2016Interspeech09.pdf) | 음성 분리에서 시간·주파수별 화자 소속 추정 및 제곱 크기 기반 Wiener-like 마스크 | 논문의 학습 방식·평가 집합을 이번 후처리가 재현하는 것은 아니다. |
| Wisdom et al., ICASSP 2019, [Differentiable Consistency Constraints](https://arxiv.org/html/1811.08521v1) | 식 (9)의 가중 혼합 일치 투영, STFT 역변환과의 관계 | 현재 구현에 이미 있는 부분이다. 합이 원음과 같다는 것은 각 출력의 화자 순도를 뜻하지 않는다. |
| Luo & Mesgarani, TASLP 2019, [Conv-TasNet](https://arxiv.org/abs/1809.07454) | 학습한 시간 영역 필터와 마스크를 통한 초기 음원 추정 | 실수 크기 마스크는 위상 복원에 한계가 있다. 기존 학습 분리기 출력보다 항상 좋아지지 않는다. |
| Chen et al., ICASSP 2020, [Continuous speech separation: dataset and analysis](https://arxiv.org/abs/2001.11482) | 완전 겹침뿐 아니라 교대 발화·부분 겹침을 포함하는 연속 평가 | 신호 점수만으로 실제 자막 인식 개선을 결론 내리지 않는다. |
| Niu et al., 2021, [Separation Guided Speaker Diarization in Realistic Mismatched Conditions](https://arxiv.org/abs/2107.02357) | 분리와 화자 군집화의 결합, 분리 실패 시 기존 경로 유지 | 논문은 모의 음성에서 잘 되는 Conv-TasNet도 실제 대화의 말투 차이에 불안정할 수 있음을 보고한다. |

공개 참고 구현은 저자들의 [Norbert softmask](https://github.com/sigsep/norbert/blob/master/norbert/__init__.py)다. 이번 모노 경로는 다채널 공간 공분산이나 beamforming을 사용하지 않는다. Norbert의 의존성을 추가할 필요 없이 필요한 수식을 NumPy로 구현할 수 있다.

## 계산 절차

입력은 동일한 샘플 위치의 혼합 파형 `x[N]`과 학습 분리기의 추정 `z[2,N]`이다. DSP 함수는 1~160,000 샘플의 유한 실수 배열, 모노 16 kHz를 받는다. 실제 모델을 부르는 wrapper와 CLI는 1~10초의 정규화 PCM을 받는다. Wrapper는 원래 Asteroid 추론 경로의 공통 L1 이득 보정을 두 모델 출력에 동일하게 적용한다. 이 배율은 비율 마스크에서는 상쇄되지만 입력 추정 대비 보정량 진단에는 반영된다.

1. **STFT:** 512 샘플 periodic Hann, hop 128 샘플. 16 kHz에서 창 길이 32 ms, hop 8 ms, 주파수 빈 간격 31.25 Hz다. 이는 현재 DSP와 같은 설정이며 새 논문의 최적값을 주장하는 것이 아니다. `X=STFT(x)`, `Zₖ=STFT(zₖ)`를 계산한다.
2. **수치 범위:** 두 추정 스펙트럼 전체의 공통 최대 크기로만 정규화한다. 채널별 정규화를 하면 두 추정의 상대 음량을 잃으므로 금지한다. 이 공통 배율은 아래 비율에서 상쇄된다.
3. **시간·주파수별 가중치:** 공통 정규화한 스펙트럼에서 `Vₖ(t,f)=|Zₖ(t,f)|²`를 계산한다. 기본값은 앞·현재·뒤 프레임의 전력을 평균하는 시간 반경 1, 주파수 평균 반경 0이다. 경계에서는 실제로 존재하는 프레임 수로 나눈다.
4. **마스크:** 평활한 전력의 합이 `epsilon=1e-12`보다 큰 빈에서 `M₀=V₀/(V₀+V₁)`, `M₁=1-M₀`. 그 이하인 빈은 `M₀=M₁=0.5`로 둔다. 단순히 분모에 epsilon만 더하면 합이 1보다 작아질 수 있으므로 합 보존을 별도로 보장한다.
5. **혼합 위상으로 복원:** `Wₖ=MₖX`. 마스크 합이 1이므로 `W₀+W₁=X`다. 현재의 window-square 정규화 overlap-add와 정확한 길이 crop으로 파형을 복원한다. 개별 출력의 자르기나 peak 정규화 없이, 필요할 때 두 출력에 같은 감쇠를 적용한다.

이 Wiener-like 단계는 크기 추정과 원 혼합의 위상을 사용한다. 반면 기존 투영은 `Pₖ=Zₖ+Mₖ(X-Z₀-Z₁)`로 학습 추정의 복소 성분을 보존한다. 완전히 상쇄된 혼합 빈에서 실수 마스크만으로 두 원음의 반대 위상을 되살릴 수 없다. 따라서 Wiener 경로가 기존 투영을 무조건 대체하면 안 된다.

기본값 `refinement_iterations=0`은 위 계산을 한 번만 한다. 추가 반복 1 또는 2를 지정하면 복원 파형을 다시 STFT로 분석해 마스크를 갱신하고, 같은 원 혼합 스펙트럼에 적용한다. **이는 제한된 재분석·마스크 반복이며 expectation-maximization(EM), 새로운 학습, 새로운 이론을 구현한 것이 아니다.** 반복이 기존 오분리를 강화할 수도 있으므로 기본값을 한 번으로 유지한다. α 조절이나 투영 결과와의 β 혼합은 이번 API에 구현하지 않았다.

### 평활화와 지연

- 시간·주파수 평활 반경은 각각 정수 0~2를 허용한다. 기본 시간 반경 1은 프레임 중심 기준 ±8 ms를 평균한다. 미래 프레임도 사용하는 대칭 평균이며 causal EMA가 아니다.
- 기본 주파수 반경 0은 인접 빈을 평균하지 않는다. 선택적 반경 1·2는 각각 ±31.25 Hz·±62.5 Hz까지 평균한다. 작은 배음 간격을 지우는 부작용을 평가해야 한다.
- centered STFT 자체도 프레임 중심 기준 16 ms의 미래 샘플을 사용한다. 현재 ConvTasNet은 전체 청크의 global normalization을 사용하므로 이 후처리의 hop 8 ms나 프레임 평균 반경을 전체 시스템 지연으로 보고하면 안 된다.
- 청크 경계마다 출력 0·1의 사람 순서가 바뀔 수 있다. 이번 함수는 청크 사이에 채널 순서나 화자 ID를 연결하지 않는다. 평활과 추가 재분석도 한 청크 내부에서만 수행한다.

## 구현 API와 진단

`refine_frequency_masks(mixture, estimates, *, time_smoothing_radius=1, frequency_smoothing_radius=0, refinement_iterations=0, peak_limit=None)`는 읽기 전용 `sources[2,N]`, `common_gain`, `diagnostics`를 반환한다. 모델 추론이나 네트워크 연결은 이 순수 DSP 함수 안에서 수행하지 않는다. Wrapper `FrequencySpeechSeparation`은 실제 모델 추론·공통 L1 보정·이 DSP를 연결하고, 중복 호출을 막는 작업 잠금과 원 모델의 미처리 꼬리 샘플 메타데이터를 유지한다.

진단에는 고정 전력 지수 2·평활 반경·추가 반복 수, 공통 이득, 합 잔차 RMS, 각 입력 추정에서 바뀐 양의 RMS를 기록한다. `mask_ambiguity_power_fraction`은 두 최종 마스크가 모두 0.4~0.6인 빈에 놓인 혼합 STFT 전력의 비율이다. 혼합이 무음이면 `None`이다. 이 모호성 구간은 MyVote의 설명용 기준이며, 교정된 화자 신뢰도 문턱이 아니다. Wrapper는 모델·DSP·전체 동기 호출 시간을 각각 기록한다. 엔트로피나 별도의 약한 빈 비율은 이번 진단에 포함하지 않았다.

마스크 모호성, 합 일치, 출력 상관, 임베딩 코사인 차이는 모두 **진단**이다. 정답 화자를 모르는 상황에서 분리 정확도나 신뢰 확률로 사용하면 안 된다. 특히 잡음·음악도 두 출력에 분배되므로 출력 2개가 화자 2명을 뜻하지 않는다.

[separate_cli.py](../src/myvote_engine/separate_cli.py)는 기존 로컬 모노 16 kHz PCM16 WAV에서 1~10초를 선택해 실제 알고리즘을 실행한다. `python -m myvote_engine.separate_cli --help`로 옵션을 확인할 수 있다. 새 출력 폴더에 `mixture.wav`, `source-A.wav`, `source-B.wav`, `result.json`을 만든다. 세 청취 WAV 모두 같은 추가 이득을 사용하며 기존 폴더는 덮어쓰지 않는다. 이 명령은 화자 ID를 부여하거나 실시간 gateway 설정을 바꾸지 않는다.

## 검증과 M4 Max 적용

수학 검증은 정확한 샘플 수, 공통 이득에 대한 불변성, 채널 교환 시 결과 교환, 무음·동일 추정·사라진 출력·반대 위상·매우 작은 수치·비유한 입력, 수정하지 않은 입력, 재합성 잔차를 포함한다. 단순 사인파 두 개의 성공은 사람 목소리 분리의 성공으로 계산하지 않는다.

음성 검증은 동일한 모델 추정을 한 번 얻고 기존 투영·기본 Wiener 경로·선택적 추가 반복을 비교한다. 정답 음원이 있는 혼합에서는 두 채널 순열을 고려한 SI-SDR improvement와 각 화자의 단어 오류를 함께 보는 것이 다음 검증 기준이다. 사용자가 지정한 40~42분 방송은 청취용 동등 음량 WAV로 비교하되 정답 분리 음원이 없다는 한계를 유지한다. 신호 검증 결과만으로 인식·화자 정확도를 선언하지 않는다. 교대 발화, 웃음, 음악, 3명 이상 겹침도 후속 실패 조건으로 포함한다.

M4 Max 64 GB에서는 이 NumPy STFT 후처리를 서버의 별도 분리 작업에 두는 구성이 적합하다. 이는 구조상 판단이며 해당 하드웨어에서 잰 속도가 아니다. 창을 기다리는 시간, 분리 모델 시간, DSP 시간, 자막 도착 시간을 따로 측정해야 한다. 기존 첫 자막 경로를 분리 작업 완료까지 기다리게 하지 않는다.

후속 모델 후보:

- **TF-GridNet:** 복소 스펙트럼을 직접 추정하고 전체 주파수·개별 대역의 시간 구조를 함께 학습한다. 이 방향은 이번 수동 Wiener 후처리보다 풍부한 위상·문맥 모델링 후보다. 공개 가중치의 실제 샘플레이트·연산자·라이선스를 확인한 뒤 Mac에서 평가한다. 논문의 성능을 MyVote 한국어 방송 성능으로 옮겨 쓰지 않는다. [논문·공식 코드 연결](https://arxiv.org/abs/2211.12433).
- **SpeechBrain SepFormer WHAMR 16k:** 공식 모델 카드는 모노 16 kHz, 잡음·잔향이 포함된 WHAMR 학습, Apache-2.0을 명시한다. 현재의 짧은 ConvTasNet 실험과 비교할 후보이며, 이 문서 작성 중 가중치를 내려받거나 설치하지 않았다. Mac MPS 가속·실시간 속도는 별도 확인해야 한다. [공식 모델 카드](https://huggingface.co/speechbrain/sepformer-whamr16k/blob/main/README.md).

사용자가 제공한 [방송 영상](https://www.youtube.com/watch?v=YlgFfqaJ-J0)의 40분대를 재현 대상으로 사용했다. 이번 설계 문서 자체는 해당 영상의 겹침 화자가 해결되었다는 증거가 아니다.
