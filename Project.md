# NVIDIA 로봇 AI 모델 비교 분석: GR00T N1.6, Cosmos Policy, DreamDojo

## 개요

NVIDIA는 로봇 AI 분야에서 세 가지 핵심적인 AI 모델을 발표하며 로봇공학의 새로운 패러다임을 제시하고 있습니다. 각 모델은 서로 다른 접근 방식과 목적을 가지고 있으며, 함께 사용될 때 강력한 시너지 효과를 발휘합니다.

- **GR00T N1.6**: 휴머노이드 로봇을 위한 오픈소스 Vision-Language-Action(VLA) 파운데이션 모델로, 멀티모달 입력을 통해 직접적인 로봇 행동을 생성
- **Cosmos Policy**: 대규모 사전 훈련된 비디오 모델을 로봇 제어 정책으로 변환하는 혁신적인 접근법으로, 비디오 모델의 물리적 이해를 활용
- **DreamDojo**: 44,000시간의 인간 1인칭 시점 영상으로 훈련된 세계 파운데이션 모델로, 로봇 환경의 물리적 시뮬레이션과 예측을 담당

## 1. GR00T N1.6

### 정의 및 특징

GR00T N1.6은 NVIDIA Isaac 플랫폼의 일부로 개발된 휴머노이드 로봇용 오픈소스 파운데이션 모델입니다. 이는 범용적인 로봇 스킬을 위한 Vision-Language-Action(VLA) 모델로, 다양한 로봇 플랫폼 간 교차 적용이 가능한 크로스 임베디먼트(cross-embodiment) 특성을 가지고 있습니다.

### 아키텍처

**핵심 아키텍처 구성요소:**

- **Base VLM**: 내부 NVIDIA Cosmos-2B VLM 변형 사용
  - 유연한 해상도 지원으로 패딩 없이 원본 종횡비로 이미지 인코딩
  - 일반적인 비전-언어 작업과 다음 행동 예측 같은 구현된 추론 작업으로 훈련
- **DiT 구조**: N1.5의 16개 레이어 대비 2배 큰 32개 레이어의 DiT(Diffusion Transformer) 사용
- **어댑터 제거**: N1.5의 VLM 후 4레이어 트랜스포머 어댑터를 제거하고, 대신 사전훈련 중 VLM의 상위 4개 레이어를 언프리즈
- **이중 시스템 아키텍처**: 듀얼 시스템 아키텍처로 고수준 추론과 저수준 제어를 분리

### 주요 개선사항 (N1.5 대비)

**아키텍처 개선:**

- 더 큰 DiT 모델 (32층 vs 16층)
- VLM 통합 방식 개선으로 더 나은 멀티모달 융합
- 상대적 행동 예측 방식으로 전환하여 더 부드럽고 정확한 동작 생성

**데이터 및 모델링 개선:**

- 절대 조인트 각도나 End-Effector 위치 대신 대부분의 임베디먼트에서 상대적 행동 청크 예측
- 더 빠른 수렴 속도로 더 부드러운 행동 생성
- 강화된 상태 정규화 및 추가적인 데이터 증강 기법 적용

### 학습 데이터

N1.5 데이터 혼합에 추가로 수천 시간의 텔레오퍼레이션 데이터를 포함:

- **Bimanual YAM Arms**: 양팔 조작 작업
- **AGIBot Genie1**: 다양한 조작 시나리오
- **Simulated Galaxea R1 Pro**: BEHAVIOR 스위트에서의 시뮬레이션 데이터
- **Unitree G1**: 전신 로코매니퓰레이션(locomotion + manipulation) 데이터

### 적용 분야 및 성능

**지원 로봇 플랫폼:**

- Bimanual YAM (양팔 로봇)
- Agibot Genie-1
- Unitree G1 (휴머노이드)

**성능 특징:**

- 전역 배치 크기 16,384로 300K 스텝 사전훈련
- 작업별 후훈련: 10K-30K 스텝, 전역 배치 크기 1K 이하
- 장거리 추론, 손재주, 멀티태스킹 능력 요구하는 복잡한 실제 로봇 실험 수행

## 2. Cosmos Policy

### 정의 및 특징

Cosmos Policy는 대규모 사전훈련된 비디오 모델(Cosmos-Predict2)을 단일 단계 후훈련을 통해 효과적인 로봇 정책으로 변환하는 혁신적인 접근법입니다. 아키텍처 수정 없이 로봇 시연 데이터만을 사용하여 비디오 모델을 로봇 제어 정책으로 직접 적응시킵니다.

### 아키텍처

**핵심 설계 원리:**

- **Latent Action Encoding**: 로봇 행동을 비디오 모델의 latent diffusion 프로세스 내에서 latent frame으로 인코딩
- **통합된 생성 프로세스**: 단일 diffusion 모델이 시각적 관찰, 행동 생성, 미래 상태 예측을 동시에 수행
- **Pre-trained Foundation 활용**: Cosmos-Predict2의 사전훈련된 물리적 이해와 핵심 학습 알고리즘을 그대로 활용

**모델 구성:**

- Cosmos-Predict2-2B를 기반으로 한 2B 파라미터 바이매뉴얼 로봇 조작 정책 모델
- 복잡한 다단계 파이프라인 없이 로봇 시연 데이터에 직접 미세조정
- 행동, 미래 상태, 값 추정을 모델의 latent sequence에 주입

### 학습 방법론

**핵심 훈련 특징:**

- **단일 단계 후훈련**: 복잡한 다단계 파이프라인 없이 직접 로봇 시연 데이터에 미세조정
- **End-to-End 학습**: 보고(see), 상상하고(imagine), 결정(decide)하는 기능을 단일 모델에 통합
- **Test-time Planning**: 성공 가능성이 높은 행동 궤적의 테스트 타임 플래닝 지원

**훈련 요구사항:**

- 소규모 ALOHA 로봇 데이터 미세조정(<200개 시연): 8개 80GB GPU (H100) × 48시간
- 대규모 실험: 32개 80GB GPU 필요

### 벤치마크 성능

**LIBERO 시뮬레이션 벤치마크:**

- **평균 성공률: 98.5%** (최고 성능)
- Spatial: 98.1%, Object: 100.0%, Goal: 98.2%, Long: 97.6%
- 기존 최고 성능 모델들(CogVLA 97.4%, OpenVLA-OFT 97.1%) 대비 우수한 성능

**RoboCasa 시뮬레이션 벤치마크:**

- **평균 성공률: 67.1%** (24개 주방 조작 작업)
- **데이터 효율성**: 50개 시연으로 300개 시연 사용 모델들과 동등 이상 성능
- 기존 최고 성능: FLARE 66.4%, GR00T-N1.5 + HAMLET 66.4% 대비 개선

### 적용 분야

- **바이매뉴얼 조작**: 양팔을 사용한 복잡한 조작 작업
- **장기간 작업 계획**: 미래 상태 예측을 통한 전략적 행동 계획
- **실제 로봇 제어**: ALOHA 로봇에서의 실제 조작 작업
- **시뮬레이션 환경**: 다양한 로봇 시뮬레이션 벤치마크에서의 정책 평가

## 3. DreamDojo

### 정의 및 특징

DreamDojo는 대규모 인간 1인칭 시점 영상으로 훈련된 생성형 세계 파운데이션 모델입니다. 로봇 모터 제어를 입력받아 미래의 픽셀 시퀀스를 생성하는 인터랙티브 세계 모델로, 물리 엔진이나 메시, 수작업 코딩 없이 순전히 데이터 기반으로 물리 법칙을 학습합니다.

### 아키텍처

**World Foundation Model 구조:**

- **Latent Action Pre-training**: 대규모 인간 데이터셋에서 latent action으로 사전훈련하여 포괄적인 물리적 지식 습득
- **Target Embodiment Post-training**: 타겟 로봇 플랫폼에서 연속적인 로봇 행동으로 후훈련
- **Autoregressive Generation**: 장기간 자기회귀적 생성으로 1분 이상의 안정적인 롤아웃 지원
- **Real-time Inference**: 증류 파이프라인을 통해 실시간 10 FPS 생성 달성

**기술적 특징:**

- 엔진, 메시, 수작업 코딩 없는 순수 데이터 기반 물리 시뮬레이션
- VR 텔레오퍼레이션을 위한 실시간 상호작용 지원
- 다양한 환경과 객체에 대한 강력한 일반화 능력

### 학습 데이터

**DreamDojo-HV 데이터셋:**

- **규모**: 44,711시간의 다양한 인간 1인칭 시점 영상
- **다양성**: 기존 최대 데이터셋 대비 15배 더 긴 기간, 96배 더 많은 스킬, 2,000배 더 많은 장면
- **포괄성**: 일상생활의 다양한 상호작용과 물리적 조작 행동 포함

**데이터 우위성:**

- 기존 세계 모델 훈련용 최대 데이터셋 대비 압도적 규모
- 인간의 자연스러운 행동 패턴과 물리적 상호작용의 다양성 확보
- 실제 환경에서의 복잡한 물리 법칙과 인과관계 학습

### 주요 기능

**1. 실시간 시뮬레이션**

- 10 FPS 실시간 생성으로 1분 이상 안정적인 연속 롤아웃
- Teacher-Student 증류를 통한 속도 최적화

**2. 라이브 텔레오퍼레이션**

- 실시간 VR 텔레오퍼레이션 지원
- 온라인 롤아웃 생성으로 즉각적인 피드백

**3. 정책 평가**

- 실제 로봇 배포 없이 정책 체크포인트의 신뢰할 수 있는 평가
- 시뮬레이션 성공률과 실제 결과 간의 강한 상관관계

**4. 모델 기반 플래닝**

- 테스트 타임 개선을 위한 모델 기반 플래닝 지원
- 도전적인 작업에서 더 높은 성공률 달성

### 적용 분야

**지원 로봇 플랫폼:**

- GR-1 (휴머노이드)
- Unitree G1 (휴머노이드)
- AgiBot (산업용 로봇)
- YAM (바이매뉴얼 로봇 암)

**핵심 응용 영역:**

- **정책 벤치마킹**: 실제 배포 전 안전한 정책 평가
- **시뮬레이션 기반 훈련**: 고충실도 시뮬레이션 환경에서의 로봇 훈련
- **연구 개발**: 새로운 로봇 행동 및 알고리즘의 빠른 프로토타이핑

## 4. 세 모델의 핵심 차이점 비교

| 비교 항목             | GR00T N1.6                                       | Cosmos Policy                                     | DreamDojo                                        |
| --------------------- | ------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------ |
| **모델 타입**   | Vision-Language-Action (VLA)                     | Video-to-Policy                                   | World Foundation Model                           |
| **주요 목적**   | 직접적인 로봇 행동 생성                          | 비디오 모델의 로봇 제어 적응                      | 물리적 세계 시뮬레이션                           |
| **입력 데이터** | 카메라 이미지 + 자연어 명령 + 로봇 상태          | 시각적 관찰 + 로봇 시연                           | 로봇 모터 제어 명령                              |
| **출력**        | 로봇 행동 (액션 청크)                            | 로봇 행동 + 미래 상태 + 가치                      | 미래 시각적 상태 시퀀스                          |
| **학습 방법**   | 멀티모달 파운데이션 모델 훈련                    | 비디오 모델 후훈련                                | 대규모 인간 영상 사전훈련                        |
| **데이터 규모** | 수천 시간 로봇 시연 데이터                       | 50-300개 시연 (작업별)                            | 44,000시간 인간 영상                             |
| **강점**        | 범용성과 언어 이해, 크로스 플랫폼 호환, 오픈소스 | 데이터 효율성, SOTA 벤치마크 성능, 미래 예측 능력 | 대규모 물리 지식, 실시간 시뮬레이션, 환경 일반화 |
| **한계**        | 대량 훈련 데이터 필요, 언어 일반화 한계          | 특정 작업에 특화, 아키텍처 제약                   | 직접적 제어 불가, 고해상도 메모리 요구           |
| **실시간 성능** | 로봇 제어 주기에 맞춘 추론                       | 플래닝 포함 실시간 제어                           | 10 FPS 실시간 생성                               |
| **적용 분야**   | 휴머노이드 로봇 범용 제어                        | 정밀 조작 작업                                    | 시뮬레이션 및 평가                               |

## 5. VLA 모델과의 차이점

### VLA 모델 정의

Vision-Language-Action (VLA) 모델은 시각적 관찰, 자연어 지시, 그리고 로봇 행동을 통합한 멀티모달 파운데이션 모델입니다. VLA는 다음 세 가지 핵심 기능을 결합합니다:

- **Vision**: 카메라 이미지나 비디오를 통한 환경 인식
- **Language**: 자연어 명령 이해 및 상황적 추론
- **Action**: 물리적 로봇 행동 생성

### GR00T N1.6와 VLA의 관계

**GR00T N1.6는 전형적인 VLA 모델입니다:**

- Cosmos-2B VLM을 기반으로 한 비전-언어 이해
- 자연어 명령을 로봇 행동으로 직접 변환
- 멀티모달 입력(시각 + 언어 + 로봇 상태)을 단일 모델에서 처리
- 크로스 임베디먼트 일반화를 통한 다양한 로봇 플랫폼 지원

### Cosmos Policy와 VLA의 차이점

**Cosmos Policy의 독특한 특징:**

- **비디오 중심 접근**: VLA의 언어 이해보다는 비디오 모델의 시공간적 이해에 중점
- **Latent Action Encoding**: 행동을 언어가 아닌 latent frame으로 인코딩
- **미래 예측 통합**: 단순 행동 생성을 넘어 미래 상태와 가치 함수까지 예측
- **데이터 효율성**: 전통적인 VLA 대비 적은 시연 데이터로 높은 성능 달성

**기존 VLA와의 비교:**

```
전통적인 VLA: [이미지 + 텍스트] → [행동]
Cosmos Policy: [비디오 시퀀스] → [행동 + 미래상태 + 가치]
```

### DreamDojo (World Model)와 VLA의 근본적 차이

**World Model vs VLA 패러다임:**

**VLA 접근법:**

- **직접 매핑**: 관찰에서 행동으로의 직접적인 End-to-End 학습
- **반응적**: 현재 상황에 대한 즉각적인 행동 결정
- **언어 중심**: 자연어 명령을 중심으로 한 의사결정

**World Model 접근법 (DreamDojo):**

- **간접 매핑**: 세계 모델을 통한 미래 예측 후 최적 행동 계획
- **예측적**: 장기간에 걸친 결과 시뮬레이션 기반 의사결정
- **물리 중심**: 물리 법칙과 인과관계 이해를 중심으로 한 추론

### 각 접근법의 장단점

| 접근법                            | 장점                                                   | 단점                                                           |
| --------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| **VLA (GR00T N1.6)**        | 단순한 E2E 학습, 언어 명령 직접 이해, 빠른 실시간 추론 | 장기 계획 한계, 물리 법칙 이해 부족, 새로운 상황 일반화 어려움 |
| **Video-Policy (Cosmos)**   | 시공간적 이해 우수, 데이터 효율적, 미래 예측 가능      | 언어 이해 제한적, 비디오 도메인 의존성, 실시간 제약            |
| **World Model (DreamDojo)** | 강력한 물리 이해, 장기 계획 가능, 안전한 시뮬레이션    | 직접 제어 불가, 높은 계산 비용, 복잡한 파이프라인              |

## 6. 시너지 효과 및 통합 활용 방안

### 세 모델의 상호 보완적 역할

**계층적 로봇 AI 시스템:**

```
DreamDojo (World Model) → 환경 이해 및 장기 계획
    ↓
Cosmos Policy (Video Policy) → 중기 전략 및 미래 예측
    ↓
GR00T N1.6 (VLA) → 실시간 행동 실행 및 언어 이해
```

### GR00T N1.6 + DreamDojo: 정책 평가 및 시뮬레이션

**통합 워크플로우:**

1. **Safe Policy Development**
   
   ```
   GR00T N1.6 정책 개발 → DreamDojo 시뮬레이션 → 안전성 검증 → 실제 배포
   ```
2. **데이터 증강 파이프라인**
   
   - DreamDojo로 합성 경험 데이터 생성
   - GR00T N1.6의 훈련 데이터 확장
   - 드문 시나리오나 위험한 상황의 안전한 학습
3. **성능 벤치마킹**
   
   - 실제 로봇 실험 전 정책 성능 예측
   - 시뮬레이션 성공률과 실제 성과의 강한 상관관계 활용
   - 비용 효율적인 정책 비교 및 선택

**실제 구현 예시:**

```python
# 의사 코드
policy = GR00T_N1_6.load_checkpoint()
simulator = DreamDojo.load_world_model()

# 시뮬레이션 기반 정책 평가
sim_results = simulator.evaluate_policy(policy, test_scenarios)
if sim_results.success_rate > threshold:
    real_deployment(policy)
```

### Cosmos Policy + DreamDojo: 세계 모델 기반 플래닝

**Model-based Planning 파이프라인:**

1. **계층적 플래닝**
   
   - DreamDojo: 거시적 환경 변화 예측 (1분+ 시간 범위)
   - Cosmos Policy: 중간 수준 행동 계획 (수 초 범위)
   - 실시간 조정 및 오류 복구
2. **불확실성 기반 의사결정**
   
   ```
   DreamDojo 예측 → 여러 시나리오 생성 → Cosmos Policy 평가 → 최적 행동 선택
   ```
3. **적응적 재계획**
   
   - 실시간 관찰과 DreamDojo 예측 비교
   - 편차 발생 시 Cosmos Policy를 통한 동적 재계획
   - 예상치 못한 상황에서의 강건한 대응

**활용 사례:**

```
장기 조작 작업 (예: 요리):
1. DreamDojo: 전체 요리 과정 시뮬레이션
2. Cosmos Policy: 각 단계별 최적 행동 계획
3. 실시간 피드백: 계획 수정 및 적응
```

### 전체 파이프라인에서의 역할 분담

**통합 로봇 AI 아키텍처:**

```
[자연어 명령] → GR00T N1.6 (언어 이해)
                      ↓
[작업 분해] → DreamDojo (환경 예측 및 제약 조건)
                      ↓
[전략 계획] → Cosmos Policy (세부 행동 계획)
                      ↓
[실행 및 모니터링] → GR00T N1.6 (실시간 제어)
                      ↓
[피드백 루프] → DreamDojo (결과 예측 및 수정)
```

**각 모델의 특화 역할:**

- **GR00T N1.6**:
  
  - 자연어 명령 해석
  - 실시간 로봇 제어
  - 크로스 플랫폼 호환성
- **Cosmos Policy**:
  
  - 중기 행동 계획
  - 시각적 시나리오 이해
  - 데이터 효율적 적응
- **DreamDojo**:
  
  - 장기 환경 시뮬레이션
  - 물리 법칙 기반 예측
  - 안전성 검증

### 실제 활용 사례

**1. 스마트 제조업 시나리오**

```
작업: 복잡한 조립 라인에서의 품질 관리

DreamDojo → 제품 결함 시나리오 시뮬레이션
Cosmos Policy → 최적 검사 경로 계획
GR00T N1.6 → 실시간 검사 및 분류 실행
```

**2. 가정용 서비스 로봇**

```
작업: "저녁 식사 준비해줘"

GR00T N1.6 → 명령 이해 및 작업 분해
DreamDojo → 주방 환경 및 재료 상태 예측
Cosmos Policy → 조리 단계별 최적 행동 시퀀스
GR00T N1.6 → 실제 조리 동작 실행
```

**3. 의료 보조 로봇**

```
작업: 수술실에서의 기구 전달

DreamDojo → 수술 진행 상황 예측
Cosmos Policy → 안전한 기구 전달 경로
GR00T N1.6 → 정밀한 기구 조작 및 전달
```

**성능 최적화 전략:**

1. **동적 모델 선택**
   
   - 작업 복잡도에 따른 적응적 모델 사용
   - 실시간 제약 vs 정확도 트레이드오프 관리
2. **지속적 학습 루프**
   
   - 실제 경험을 통한 세 모델의 동시 개선
   - 시뮬레이션-실제 도메인 갭 최소화
3. **리소스 효율적 배치**
   
   - 클라우드-엣지 하이브리드 컴퓨팅
   - 모델별 최적 하드웨어 할당

## 7. 결론

### 각 모델의 위치와 역할

NVIDIA의 세 가지 로봇 AI 모델은 각각 독특한 강점을 가지며, 로봇공학의 서로 다른 측면을 해결합니다:

**GR00T N1.6**는 **범용적인 로봇 인터페이스** 역할을 합니다. VLA 모델로서 자연어 명령을 직접 로봇 행동으로 변환하는 능력은 인간-로봇 상호작용의 핵심입니다. 오픈소스 특성과 크로스 플랫폼 호환성으로 로봇공학 생태계의 표준이 될 가능성이 높습니다.

**Cosmos Policy**는 **정밀한 조작의 전문가** 역할을 담당합니다. 비디오 모델의 시공간적 이해를 활용하여 기존 VLA 모델보다 뛰어난 성능을 보이며, 특히 데이터 효율성 면에서 실용적 배포의 새로운 가능성을 제시합니다.

**DreamDojo**는 **물리적 세계의 시뮬레이터** 역할을 합니다. 44,000시간의 인간 경험에서 학습한 물리 법칙 이해는 안전하고 신뢰할 수 있는 로봇 개발의 핵심 인프라가 됩니다.

### 로봇 AI 발전에 대한 시사점

**1. 통합적 접근의 중요성**
세 모델은 단독으로도 뛰어난 성능을 보이지만, 통합될 때 진정한 시너지를 발휘합니다. 이는 로봇 AI가 단일 모델의 한계를 넘어 다중 모델 시스템으로 발전해야 함을 시사합니다.

**2. 데이터 중심 개발 패러다임**
특히 DreamDojo의 44,000시간 데이터와 Cosmos Policy의 데이터 효율성은 로봇 AI 발전에서 데이터의 질과 양이 모두 중요함을 보여줍니다. 합성 데이터 생성과 실제 데이터 수집의 균형이 핵심입니다.

**3. 실용적 배포의 현실화**
GR00T N1.6의 오픈소스 공개와 Cosmos Policy의 높은 성능은 로봇 AI가 연구실을 넘어 실제 산업 환경으로 확산될 수 있는 토대를 마련했습니다.

**4. 안전성과 신뢰성**
DreamDojo를 통한 시뮬레이션 기반 검증은 로봇 AI의 안전한 배포를 위한 필수 요소임을 보여줍니다. 실제 배포 전 가상 환경에서의 충분한 검증이 표준이 되어야 합니다.

### 향후 전망

**단기 전망 (1-2년)**

- GR00T N1.6 기반의 다양한 휴머노이드 로봇 상용화
- Cosmos Policy의 산업용 조작 로봇 적용 확대
- DreamDojo를 활용한 로봇 개발 워크플로우 표준화

**중기 전망 (3-5년)**

- 세 모델의 완전 통합된 로봇 AI 플랫폼 등장
- 실시간 학습과 적응이 가능한 차세대 로봇 시스템
- 의료, 제조, 서비스업에서의 본격적인 로봇 도입

**장기 전망 (5년+)**

- 일반적인 물리적 추론 능력을 가진 AGI(Artificial General Intelligence) 로봇
- 인간과 자연스럽게 협업하는 로봇 파트너 시대
- 로봇 AI의 민주화로 개인용 로봇 어시스턴트 보편화

## 8. 실전 프로젝트 커리큘럼: 12주 로드맵

### 프로젝트 개요

**목표**: ROBOTIS OMX 매니퓰레이터로 모방학습 데이터를 수집하고, ACT 베이스라인 및 GR00T N1.6 VLA 파인튜닝을 수행하며, Cosmos Policy와 DreamDojo는 추론 기반 비교 분석

**접근 방식**: Option C 하이브리드 (OMX 기반)
- **Week 1-2**: OMX 하드웨어 조립 + Physical AI Tools/LeRobot 환경 구축
- **Week 3-4**: GR00T N1.6 환경 설정 + OMX로 모방학습 데이터 수집
- **Week 5-6**: ACT 베이스라인 + GR00T 파인튜닝 + OMX 배포 비교
- **Week 7-8**: Cosmos Policy 추론 전용 (사전 훈련 모델 평가)
- **Week 9-10**: DreamDojo 추론 전용 (세계 모델 시뮬레이션 체험)
- **Week 11-12**: 세 모델 비교 분석 및 프로젝트 마무리

**환경 및 도구:**

- **하드웨어**: ROBOTIS OMX-AI (OMX-L 리더 + OMX-F 팔로워, 5-DOF + 1 Gripper)
- **소프트웨어**: ROS 2 Jazzy + Physical AI Tools (Docker) + ROBOTIS 포크 LeRobot
- **시뮬레이션**: Gazebo/RViz (URDF), MuJoCo (옵션)
- **AI 모델**: ACT (베이스라인), GR00T N1.6 (파인튜닝), Cosmos Policy (추론), DreamDojo (추론)
- **프로젝트 목표 작업**: 객체 조작 및 배치 작업 (Pick and Place++)
- **데이터 수집**: 15 FPS, HuggingFace datasets 포맷, 6차원 액션 (5관절 + 그리퍼)
- **GPU 요구사항**: GR00T 파인튜닝용 1x H100/L40, ACT 학습/Cosmos/DreamDojo 추론용 1x RTX 4090+

**OMX 하드웨어 사양:**

| 구분 | OMX-L (리더) | OMX-F (팔로워) |
|------|-------------|---------------|
| DOF | 5 + 1 Gripper | 5 + 1 Gripper |
| 작업 반경 | 335mm | 400mm |
| 무게 | 360g | 560g |
| 페이로드 | - | 100~250g |
| 전원 | 5V USB-C | 12VDC |
| 모터 | XL330 | XL430 + XL330 |
| 통신 | TTL, 1Mbps | TTL, 1Mbps |

---

### Week 1-2: 환경 설정 및 기초 인프라 구축

#### Week 1: OMX 하드웨어 조립 및 Physical AI Tools 환경 구축

**학습 목표:**

- OMX-AI (리더 + 팔로워) 하드웨어 조립 및 캘리브레이션
- Physical AI Tools Docker 환경 구축
- ROS 2 Jazzy 기반 제어 아키텍처 이해

**실습 내용:**

1. **OMX 하드웨어 조립**

   - OMX-L (리더) 및 OMX-F (팔로워) 조립 (공장 캘리브레이션 완료)
   - USB-C 연결 및 DYNAMIXEL 모터 ID 확인
   - 안전 홈밍 기능 테스트 (초기 자세 복귀)

2. **Physical AI Tools Docker 환경 구축**

   ```bash
   # 사전 요구사항: Docker Engine + NVIDIA Container Toolkit
   sudo usermod -aG docker $USER
   sudo systemctl enable docker

   # Open Manipulator 컨테이너
   git clone https://github.com/ROBOTIS-GIT/open_manipulator
   cd open_manipulator/docker && ./container.sh start

   # Physical AI Tools 컨테이너
   git clone --recurse-submodules https://github.com/ROBOTIS-GIT/physical_ai_tools.git
   cd physical_ai_tools/docker && ./container.sh start
   ```

3. **USB 포트 및 카메라 설정**

   ```bash
   # 컨테이너 진입 후 USB 디바이스 확인
   ./container.sh enter
   ls -al /dev/serial/by-id/

   # Leader/Follower 포트 설정
   # omx_l_leader_ai.launch.py → Leader serial ID 입력
   # omx_f_follower_ai.launch.py → Follower serial ID 입력

   # 카메라 토픽 설정 (compressed 필수)
   sudo nano ~/ros2_ws/src/physical_ai_tools/physical_ai_server/config/omx_f_config.yaml
   # 토픽 예: camera1/image_raw/compressed
   ```

4. **ROS 2 제어 아키텍처 확인**

   - `ros2_control` 프레임워크 (100Hz 조인트 제어)
   - 파이프라인: 입력 소스 → ROS 2 JointTrajectory → controller_manager → DynamixelHardwareInterface → TTL
   - arm_controller (5-DOF), gpio_command_controller (그리퍼)

**결과물:**

- ✅ OMX-AI 하드웨어 조립 및 동작 확인
- ✅ Physical AI Tools + Open Manipulator Docker 컨테이너 실행
- ✅ 카메라 스트리밍 및 ROS 2 토픽 확인

**학습 자료:**

- https://ai.robotis.com/omx/assembly_guide_omx.html
- https://ai.robotis.com/omx/setup_guide_physical_ai_tools.html
- https://ai.robotis.com/omx/hardware_omx.html

---

#### Week 2: OMX 텔레오퍼레이션 및 LeRobot 설정

**학습 목표:**

- OMX 리더-팔로워 텔레오퍼레이션 실습
- ROBOTIS 포크 LeRobot 설치 및 CLI 워크플로우 이해
- 데이터 수집 파이프라인 이해 (Physical AI Tools + LeRobot 두 경로)

**실습 내용:**

1. **ROBOTIS 포크 LeRobot 설치**

   ```bash
   # ⚠️ 원본 LeRobot이 아닌 ROBOTIS 포크 사용 필수
   conda create -y -n lerobot python=3.10
   conda activate lerobot
   conda install -c conda-forge ffmpeg=6.1.1 -y

   git clone https://github.com/ROBOTIS-GIT/lerobot.git
   cd lerobot
   pip install -e ".[dynamixel]"
   ```

2. **OMX 텔레오퍼레이션 테스트 (LeRobot CLI)**

   ```bash
   # USB 포트 확인
   lerobot-find-port
   # 예: 팔로워 /dev/ttyACM0, 리더 /dev/ttyACM1

   # 텔레오퍼레이션 실행
   python -m lerobot.teleoperate \
       --robot.type=omx_follower \
       --robot.port=/dev/ttyACM0 \
       --teleop.type=omx_leader \
       --teleop.port=/dev/ttyACM1

   # 카메라 통합 텔레오퍼레이션 (OpenCV, 640x480, 30fps)
   python -m lerobot.teleoperate \
       --robot.type=omx_follower \
       --robot.port=/dev/ttyACM0 \
       --teleop.type=omx_leader \
       --teleop.port=/dev/ttyACM1 \
       --robot.cameras='[{type=opencv, key=cam1, index=0, width=640, height=480, fps=30}]'
   ```

3. **Physical AI Tools 웹 UI 텔레오퍼레이션**

   ```bash
   # Physical AI Server 실행 (Docker 내부)
   cd physical_ai_tools/docker
   ./container.sh enter
   ai_server

   # OMX Follower 노드 실행 (별도 터미널)
   cd open_manipulator/docker && ./container.sh enter
   ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py
   ```
   - 브라우저에서 `http://localhost` 접속 → 로봇 타입 선택 → 텔레오퍼레이션

4. **샘플 데이터 녹화 테스트 (LeRobot CLI)**

   ```bash
   # HuggingFace 인증
   huggingface-cli login
   export HF_USER=$(huggingface-cli whoami | head -1)

   # 5개 에피소드 샘플 녹화
   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-test \
       --dataset.num_episodes=5 \
       --dataset.single_task="Pick up the red cube"
   # 제어키: → 에피소드 완료, ← 재녹화, ESC 종료
   ```

**데이터 스키마 (OMX 6차원):**

```
action: List[float32]              # 리더 상태 (5관절 + 그리퍼 = 6차원)
observation.state: List[float32]   # 팔로워 상태 (6차원)
observation.images.camera1: Image  # RGB 이미지
timestamp: float32
frame_index: int64
episode_index: int64
```

**결과물:**

- ✅ OMX 리더-팔로워 텔레오퍼레이션 동작 확인
- ✅ ROBOTIS 포크 LeRobot CLI 워크플로우 숙지
- ✅ Physical AI Tools 웹 UI 데이터 수집 경로 확인
- ✅ 샘플 시연 데이터 (5-10개)

**과제:**

- 3가지 간단한 작업에 대해 각 5개씩 시연 데이터 수집 (총 15개)
  - 작업 1: 객체 잡기 (Pick)
  - 작업 2: 객체를 목표 위치로 이동 (Place)
  - 작업 3: 객체 쌓기 (Stack)

**학습 자료:**

- https://ai.robotis.com/omx/setup_guide_lerobot.html
- https://ai.robotis.com/omx/operation_omx.html
- https://ai.robotis.com/omx/lerobot_imitation_learning_omx.html

---

### Week 3-4: GR00T N1.6 기초 및 데이터 수집

#### Week 3: GR00T N1.6 이해 및 OMX 연동 설정

**학습 목표:**

- GR00T N1.6 아키텍처 이해
- 사전 훈련된 모델 로드 및 추론
- OMX용 `modality_config.json` 작성 (5-DOF + gripper = 6차원 액션 공간)

**실습 내용:**

1. **GR00T N1.6 설치**

   ```bash
   # GitHub 클론
   git clone https://github.com/NVIDIA/Isaac-GR00T
   cd Isaac-GR00T

   # uv 기반 환경 설정 (conda가 아닌 uv 사용)
   pip install uv
   uv venv .venv --python 3.10
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```
2. **사전 훈련 모델 테스트**

   ```python
   from gr00t.policy.gr00t_policy import Gr00tPolicy
   from gr00t.data.embodiment_tags import EmbodimentTag
   import numpy as np

   # 실제 API: Gr00tPolicy 생성자 사용
   policy = Gr00tPolicy(
       model_path="nvidia/GR00T-N1.6-3B",
       embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
       device="cuda"
   )

   # Observation은 중첩 딕셔너리 형태
   observation = {
       "video": {
           "cam": np.random.rand(1, 1, 224, 224, 3).astype(np.float32)
       },
       "state": {
           "joints": np.random.rand(1, 1, 6).astype(np.float32)  # OMX: 5-DOF + 1 gripper = 6-dim
       },
       "annotation": {
           "task": [["pick up the red cube"]]
       }
   }

   # 추론: get_action() 메서드 사용
   action = policy.get_action(observation)
   ```
3. **OMX용 modality_config.json 작성 (핵심)**

   ```json
   // configs/omx_modality_config.json
   // OMX: 5-DOF + 1 gripper = 6차원 액션/상태 공간
   {
     "video": {
       "cam1": {"resolution": [224, 224], "num_frames": 1}
     },
     "state": {
       "joints": {"dim": 6}
     },
     "action": {
       "joints": {"dim": 6, "action_horizon": 16}
     },
     "annotation": {
       "task": {"type": "string"}
     }
   }
   ```
   - `EmbodimentTag.NEW_EMBODIMENT`로 OMX 등록
   - OMX의 6차원(5관절 + 그리퍼) → GR00T 액션 공간 매핑
   - 상대 행동(relative action chunk) 방식 이해
   - ROBOTIS 포크 LeRobot 데이터 → GR00T LeRobot v2 포맷 변환 확인

**결과물:**

- ✅ 사전 훈련 GR00T 모델 실행 (RTX 4090: ~44ms, H100: ~38ms)
- ✅ OMX용 modality_config.json 완성 (6차원 액션)
- ✅ 추론 성능 벤치마크

**이론 학습:**

- VLA 모델 아키텍처 논문 리뷰
- Diffusion Policy vs VLA 비교
- 상대 행동 vs 절대 행동 이해

---

#### Week 4: OMX로 체계적 데이터 수집

**학습 목표:**

- OMX Physical AI Tools 웹 UI와 LeRobot CLI 두 경로로 대량 데이터 수집
- 데이터 다양성 확보 전략
- 데이터 품질 관리 및 GR00T 포맷 변환

**실습 내용:**

1. **Physical AI Tools 웹 UI로 대량 수집 (경로 1)**

   - 브라우저에서 Record 페이지 접속
   - Task Info 설정:
     - **FPS**: 15 (OMX 권장)
     - **Episode Time**: 30-60초
     - **Reset Time**: 15-30초
     - **Num Episodes**: 50개/작업
   - Start 클릭 → 워밍업 → 녹화 → 리셋 자동 반복
   - 제어: Stop(저장 후 중단), Retry(재시작), Next(다음으로)

2. **LeRobot CLI로 대량 수집 (경로 2)**

   ```bash
   # 작업별 50개 에피소드 수집
   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-pick \
       --dataset.num_episodes=50 \
       --dataset.single_task="Pick up the object" \
       --dataset.fps=15

   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-place \
       --dataset.num_episodes=50 \
       --dataset.single_task="Place the object on the target"

   lerobot-record \
       --dataset.repo_id=${HF_USER}/omx-stack \
       --dataset.num_episodes=50 \
       --dataset.single_task="Stack the cubes"
   ```

3. **데이터 다양성 확보**

   - 객체 위치/종류 변경 (OMX 페이로드 100~250g 이내)
   - 조명 조건 변경
   - 카메라 각도 변경
   - 배경 변경

4. **데이터 품질 관리**

   - Physical AI Tools 웹 UI에서 에피소드별 시각화 검증
   - 불량 에피소드 제거 (Data Tools → 에피소드 삭제, 자동 재색인)
   - 데이터셋 병합 (여러 세션의 데이터를 하나로 통합)

5. **GR00T LeRobot v2 포맷 변환**

   ```bash
   # OMX HuggingFace 데이터 → GR00T-flavored LeRobot v2 변환
   # parquet (상태/액션) + mp4 (비디오) + meta/modality.json
   python scripts/convert_to_lerobot_v2.py \
       --input_dir data/omx_recordings \
       --output_dir data/omx_groot_v2 \
       --modality_config configs/omx_modality_config.json
   ```

**결과물:**

- ✅ 150+ 고품질 시연 데이터 (작업당 50개 × 3작업)
- ✅ HuggingFace Hub에 데이터셋 업로드
- ✅ GR00T LeRobot v2 포맷 변환 완료
- ✅ 데이터 품질 검증 보고서

**학습 자료:**

- https://ai.robotis.com/omx/dataset_preparation_recording_omx.html
- https://ai.robotis.com/omx/data_tools_omx.html
- https://ai.robotis.com/omx/dataset_preparation_visualization_omx.html

---

### Week 5-6: ACT 베이스라인 & GR00T 파인튜닝 및 OMX 배포

#### Week 5: ACT 베이스라인 학습 + GR00T 파인튜닝

**학습 목표:**

- ACT (Action Chunking Transformers) 베이스라인 학습으로 OMX 데이터 검증
- GR00T N1.6 후훈련(post-training) 수행
- 두 모델의 학습 과정 비교 (학습 곡선, 수렴 속도)

**실습 내용:**

1. **ACT 베이스라인 학습 (ROBOTIS fork LeRobot)**

   ACT는 OMX 공식 지원 정책으로, GR00T 비교의 베이스라인 역할을 합니다.

   ```bash
   # ACT 정책 학습 (Physical AI Tools Docker 내부 또는 LeRobot CLI)
   # 방법 1: Physical AI Tools Web UI
   # http://localhost:7860 → Training 탭 → ACT 선택 → 학습 시작

   # 방법 2: ROBOTIS fork LeRobot CLI
   cd ~/lerobot  # ROBOTIS-GIT/lerobot
   lerobot-train \
       --dataset.repo_id=${HF_USER}/omx_pick_cube \
       --policy.type=act \
       --policy.device=cuda \
       --training.num_epochs=100 \
       --training.batch_size=64
   ```

   ACT 학습 결과물:
   - `outputs/train/` 디렉토리에 `config.json` + `model.safetensors` 생성
   - 학습 로그 및 loss 추이 기록

2. **OMX용 modality_config 작성 (GR00T용)**

   ```json
   // configs/omx_modality_config.json
   // OMX: 5-DOF + 1 gripper = 6-dim action/state
   {
     "video": {"cam1": {"resolution": [224, 224], "num_frames": 1}},
     "state": {"joints": {"dim": 6}},
     "action": {"joints": {"dim": 6, "action_horizon": 16}},
     "annotation": {"task": {"type": "string"}}
   }
   ```

3. **OMX 데이터를 GR00T LeRobot v2 포맷으로 변환**

   ```bash
   # Week 4에서 수집한 OMX HuggingFace 데이터셋을
   # GR00T-flavored LeRobot v2 포맷으로 변환
   # parquet (상태/액션) + mp4 (비디오) + meta/modality.json 필수
   python scripts/convert_to_lerobot_v2.py \
       --input_dir data/omx_hf_dataset \
       --output_dir data/omx_lerobot_v2 \
       --modality_config configs/omx_modality_config.json
   ```

   > **주의**: OMX LeRobot 데이터는 HuggingFace datasets 포맷이며,
   > GR00T는 LeRobot v2 (parquet + mp4 + modality.json) 포맷을 요구합니다.
   > 6-dim action space가 modality_config와 일치하는지 반드시 확인하세요.

4. **GR00T N1.6 파인튜닝 실행**

   ```bash
   # 실제 launch_finetune.py 사용 (OMX 6-dim 설정)
   CUDA_VISIBLE_DEVICES=0 uv run python \
       gr00t/experiment/launch_finetune.py \
       --base-model-path nvidia/GR00T-N1.6-3B \
       --dataset-path data/omx_lerobot_v2 \
       --embodiment-tag NEW_EMBODIMENT \
       --modality-config-path configs/omx_modality_config.json \
       --num-gpus 1 \
       --max-steps 10000 \
       --batch-size 32
   ```

5. **학습 모니터링 및 비교**

   - TensorBoard 로그 분석 (ACT vs GR00T 동시 비교)
   - 손실 함수 수렴 속도 비교
   - ACT: epoch 기반 학습 곡선 / GR00T: step 기반 학습 곡선
   - 하이퍼파라미터 최적화 (learning rate, batch size, augmentation)

**결과물:**

- ✅ ACT 베이스라인 체크포인트 (`config.json` + `model.safetensors`)
- ✅ 파인튜닝된 GR00T 체크포인트
- ✅ ACT vs GR00T 학습 과정 비교 리포트
- ✅ 최적 하이퍼파라미터 문서

**예상 GPU 시간:**

- ACT 학습: 1x RTX 3090/4090 충분, 100 epoch 기준 약 1-3시간
- GR00T 파인튜닝: 1x H100 또는 L40 권장, 10K 스텝 기준 약 4-8시간
- 다중 GPU 사용 시 GR00T `--num-gpus` 조정

---

#### Week 6: OMX 실제 로봇 배포 및 ACT vs GR00T 비교 평가

**학습 목표:**

- ACT 및 GR00T 모델을 OMX 실제 로봇에 배포
- 동일 작업에서 두 정책의 성능 비교
- 정량적/정성적 평가 및 실패 분석

**실습 내용:**

1. **ACT 모델 OMX 배포 (Physical AI Tools / LeRobot)**

   ```bash
   # 방법 1: Physical AI Tools Web UI로 배포
   # http://localhost:7860 → Inference 탭 → 학습된 ACT 모델 선택 → 실행

   # 방법 2: LeRobot CLI로 배포
   cd ~/lerobot
   python -m lerobot.scripts.control_robot \
       --robot.type=omx_follower \
       --robot.port=/dev/ttyACM0 \
       --control.type=record \
       --control.policy.path=outputs/train/act_omx_pick \
       --control.single_task="Pick up the red cube"
   ```

2. **GR00T 모델 OMX 배포**

   ```python
   import numpy as np
   from gr00t.policy.gr00t_policy import Gr00tPolicy
   from gr00t.data.embodiment_tags import EmbodimentTag

   # 파인튜닝된 GR00T 체크포인트 로드
   policy = Gr00tPolicy(
       model_path="checkpoints/groot_omx/best_model",
       embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
       device="cuda"
   )

   # OMX 실시간 추론 루프 (ROS 2 연동)
   # ros2_control 100Hz 제어 주기와 동기화 필요
   while True:
       obs = get_omx_observation()  # 카메라 224x224 + 6-dim joints
       action = policy.get_action(obs)  # 6-dim action 출력
       omx_robot.execute_action(action)  # OMX follower에 명령 전송
   ```

   > **참고**: GR00T 추론은 GPU 필요 (CUDA), ACT는 CPU에서도 동작 가능.
   > OMX의 `ros2_control`은 100Hz로 동작하므로, 추론 지연시간(latency) 측정 필수.

3. **동일 조건 비교 평가 프로토콜**

   | 평가 항목 | ACT | GR00T | 비고 |
   |----------|-----|-------|------|
   | 작업 성공률 | - | - | 각 20회 시행 |
   | 평균 실행 시간 | - | - | 초 단위 |
   | 추론 지연시간 | - | - | ms 단위 |
   | Smoothness | - | - | 궤적 jerk 측정 |
   | 일반화 능력 | - | - | 물체 위치 변경 시 |

   ```bash
   # 각 모델별 동일 작업 20회 반복 실험
   # 작업 예시: pick_cube, place_on_plate, stack_blocks
   for task in pick_cube place_on_plate stack_blocks; do
       echo "=== Evaluating ACT on ${task} ==="
       python eval_omx.py --policy=act --task=${task} --num_trials=20
       echo "=== Evaluating GR00T on ${task} ==="
       python eval_omx.py --policy=groot --task=${task} --num_trials=20
   done
   ```

4. **실패 케이스 분석 및 개선**

   - 실패 에피소드 영상 녹화 (LeRobot `--control.type=record`)
   - ACT vs GR00T 실패 패턴 비교 (grasping 실패, 위치 오차, 충돌 등)
   - 필요시 추가 데이터 수집 → 재학습 반복

**결과물:**

- ✅ ACT 베이스라인 OMX 배포 완료
- ✅ GR00T OMX 배포 완료
- ✅ ACT vs GR00T 성능 비교 보고서 (표 + 그래프)
- ✅ 실패 케이스 분석 및 개선 방안 문서

**평가 메트릭:**

- 작업 성공률 (목표: ACT ≥70%, GR00T ≥80%)
- 추론 지연시간 (목표: <100ms per step)
- 궤적 부드러움 (Smoothness/Jerk 지표)
- 일반화 점수 (물체 위치 변경 시 성공률 변화)

---

### Week 7-8: Cosmos Policy 추론 및 평가

> **Note**: Cosmos Policy 파인튜닝에는 8x H100 80GB가 필요하므로, 본 프로젝트에서는 사전 훈련된 모델의 추론과 평가에 집중합니다.

#### Week 7: Cosmos Policy 환경 구축 및 추론

**학습 목표:**

- Cosmos Policy 아키텍처 이해 (Latent Action Encoding, 통합 생성 프로세스)
- Docker 기반 환경 구축
- 사전 훈련 모델로 LIBERO 벤치마크 추론

**실습 내용:**

1. **Cosmos Policy 설치 (Docker 필수)**

   ```bash
   git clone https://github.com/NVlabs/cosmos-policy
   cd cosmos-policy

   # Docker 환경 빌드 및 실행 (pip 설치 미지원)
   docker build -t cosmos-policy -f docker/Dockerfile .
   docker run --gpus all -it --rm \
       -v $(pwd):/workspace \
       cosmos-policy bash
   ```
2. **LIBERO 벤치마크 추론 (사전 훈련 모델)**

   ```python
   from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
   from cosmos_policy.experiments.robot.cosmos_utils import (
       get_action, get_model, load_dataset_stats
   )

   # Config 기반 모델 로딩 (from_pretrained 패턴 아님)
   cfg = PolicyEvalConfig(
       config="cosmos_predict2_2b_480p_libero__inference_only",
       ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
       libero_task_suite="libero_spatial",
       num_episodes_per_task=20,
   )

   model, cosmos_config = get_model(cfg)
   dataset_stats = load_dataset_stats(cfg)

   # 추론 실행 (6-10 GB VRAM, 단일 GPU 충분)
   action_return_dict = get_action(
       cfg, model, dataset_stats,
       observation, task_description,
       cosmos_config=cosmos_config
   )
   # 반환: 행동 + 미래 상태 예측 + 가치 추정
   ```
3. **LIBERO 4개 서브스위트 전체 평가**

   ```bash
   # 실제 평가 스크립트 (Docker 내부)
   bash eval.sh cosmos_predict2_2b_480p_libero__inference_only \
       nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
       libero_spatial 20

   # 4개 서브스위트: spatial, object, goal, long
   ```
4. **미래 예측 시각화**

   - Cosmos Policy의 latent diffusion 기반 미래 상태 예측 시각화
   - 행동 + 미래 프레임 + 가치 함수를 동시 분석
   - 다양한 작업에서의 예측 품질 비교

**결과물:**

- ✅ Docker 기반 Cosmos Policy 환경 구축
- ✅ LIBERO 벤치마크 추론 결과 (목표: 논문 재현 ~98.5%)
- ✅ 미래 예측 시각화 자료

**이론 학습:**

- Cosmos-Predict2 아키텍처 논문 리뷰
- Latent Action Encoding 원리 이해
- Video-to-Policy 전환 메커니즘

---

#### Week 8: Cosmos Policy 심화 분석 및 GR00T 비교

**학습 목표:**

- Cosmos Policy의 데이터 효율성 분석
- GR00T N1.6 파인튜닝 결과와 정량적 비교
- Test-time Planning 메커니즘 이해

**실습 내용:**

1. **데이터 포맷 차이 분석**

   ```python
   # Cosmos Policy 데이터 포맷 (GR00T LeRobot v2와 다름)
   # Cosmos: pickle observation dict
   cosmos_obs = {
       "primary_image": np.array(...),   # [H, W, 3]
       "wrist_image": np.array(...),     # [H, W, 3]
       "proprio": np.array(...),         # [D]
   }
   # + T5 text embeddings 필요

   # GR00T: LeRobot v2 포맷
   groot_obs = {
       "video": {"cam": np.array(...)},  # [B, T, H, W, C]
       "state": {"joints": np.array(...)},
       "annotation": {"task": [["..."]]}
   }
   ```
2. **GR00T vs Cosmos Policy 정량적 비교**

   | 메트릭 | GR00T N1.6 (파인튜닝) | Cosmos Policy (추론) |
   |--------|----------------------|---------------------|
   | 성공률 (단순 작업) | Week 6 결과 | LIBERO 결과 |
   | 추론 속도 | ~44ms (RTX 4090) | 측정 |
   | 데이터 요구량 | 50-100 demos | 50-200 demos |
   | 미래 예측 | 불가 | 가능 |
   | 언어 이해 | 자연어 명령 | 제한적 |

3. **Cosmos Policy 강점/약점 실증**

   - 강점: 데이터 효율성, SOTA 벤치마크 성능, 미래 예측
   - 약점: Docker 전용, 커스텀 작업 파인튜닝 시 8x H100 필요
   - GR00T 대비: 언어 이해 제한, 크로스 플랫폼 호환성 부재

4. **비교 분석 보고서 작성**

   - 두 모델의 아키텍처적 차이 (VLA vs Video-to-Policy)
   - 각 모델이 우수한 작업 유형 분석
   - 실용적 배포 관점에서의 트레이드오프

**결과물:**

- ✅ GR00T vs Cosmos Policy 비교 보고서
- ✅ 데이터 포맷 변환 가이드
- ✅ 각 모델 적합 시나리오 분석

---

### Week 9-10: DreamDojo 추론 및 세계 모델 체험

> **Note**: DreamDojo 파인튜닝에는 8 GPU 노드(H100 80GB)가 필요하고, 증류 파이프라인과 텔레오퍼레이션 코드가 아직 미공개 상태입니다. 사전 훈련 모델의 추론과 롤아웃 생성에 집중합니다.

#### Week 9: DreamDojo 환경 구축 및 롤아웃 생성

**학습 목표:**

- DreamDojo 세계 파운데이션 모델 아키텍처 이해
- 사전 훈련 모델 로드 및 롤아웃 생성
- 물리 시뮬레이션 품질 평가

**실습 내용:**

1. **DreamDojo 설치**

   ```bash
   git clone https://github.com/dreamdojo-world/dreamdojo
   cd dreamdojo

   # 주의: 패키지명이 cosmos-predict2 (DreamDojo가 아님)
   pip install -e .

   # launch.sh의 내부 경로 하드코딩 수정 필요
   # /mnt/amlfs-01/shared/... → 로컬 경로로 변경
   ```
2. **사전 훈련 모델 로드 및 추론**

   ```python
   # DreamDojo는 Python 클래스가 아닌 config + torchrun 기반
   # 추론은 scripts 디렉토리의 스크립트 활용

   # 사전 훈련 모델 다운로드
   # nvidia/DreamDojo-2B-480p-GR1 등 HuggingFace 체크포인트

   # 추론 실행 (config 기반)
   # configs/2b_480_640_gr1.yaml 참조
   ```
3. **롤아웃 생성 및 시각화**

   - 초기 상태 + 행동 시퀀스 입력 → 미래 픽셀 시퀀스 생성
   - 다양한 로봇 플랫폼(GR-1, Unitree G1, YAM)에서의 롤아웃 비교
   - 물리적 일관성 평가 (중력, 충돌, 마찰)

4. **세계 모델 품질 분석**

   - 생성된 비디오의 시각적 품질 (FID, SSIM 등)
   - 장기 롤아웃 안정성 (1분+ 연속 생성)
   - 물리 법칙 준수 여부 정성적 평가

**결과물:**

- ✅ DreamDojo 추론 환경 구축
- ✅ 다양한 시나리오의 롤아웃 비디오
- ✅ 세계 모델 품질 평가 보고서

**이론 학습:**

- World Foundation Model 논문 리뷰
- Latent Action Pre-training 메커니즘
- Autoregressive Generation vs Diffusion 비교

**제약사항:**

- 증류 파이프라인 미공개 → 실시간 10 FPS 생성 불가
- 텔레오퍼레이션 코드 미공개 → VR 연동 불가
- 후훈련(Post-training)은 8 GPU 노드 필요 → 추론만 수행

---

#### Week 10: DreamDojo 정책 평가 및 세 모델 크로스 분석

**학습 목표:**

- DreamDojo를 활용한 GR00T 정책 시뮬레이션 평가 (가능 범위 내)
- 세 모델의 아키텍처적 차이와 상호 보완성 분석
- 통합 파이프라인의 이론적 설계

**실습 내용:**

1. **DreamDojo 기반 정책 평가 (제한적)**

   - 사전 훈련 모델의 지원 로봇 플랫폼에서 GR00T 정책 행동을 입력으로 롤아웃 생성
   - 생성된 미래 프레임의 시각적 품질로 정책 품질 간접 평가
   - 실제 로봇 결과와의 상관관계 분석

2. **세 모델 종합 비교 분석**

   ```
   ┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
   │ 비교 항목       │ GR00T N1.6       │ Cosmos Policy    │ DreamDojo        │
   ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
   │ 모델 타입       │ VLA              │ Video-to-Policy  │ World Model      │
   │ 본 프로젝트     │ 파인튜닝 완료    │ 추론 전용        │ 추론 전용        │
   │ 파인튜닝 GPU    │ 1x H100/L40     │ 8x H100 80GB    │ 8x H100 80GB    │
   │ 추론 GPU        │ 1x RTX 4090     │ 1x (6-10GB)     │ 1x A100+        │
   │ 출력            │ 로봇 행동       │ 행동+미래+가치   │ 미래 비디오      │
   │ 언어 이해       │ ✅               │ 제한적           │ ❌               │
   │ 크로스 플랫폼   │ ✅               │ ❌               │ ✅               │
   │ 실시간 제어     │ ✅               │ ✅               │ ❌ (시뮬레이션)  │
   └─────────────────┴──────────────────┴──────────────────┴──────────────────┘
   ```

3. **이론적 통합 파이프라인 설계**

   ```
   [실제 구현 가능 파이프라인]

   자연어 명령 → GR00T N1.6 (실시간 제어)
                      ↓ 행동 시퀀스
   정책 검증 → DreamDojo (미래 예측으로 안전성 사전 검증)
                      ↓ 피드백
   재계획   → Cosmos Policy (대안 행동 제안)

   ※ 세 모델의 I/O 포맷이 상이하므로
      데이터 변환 레이어가 필수:
      - GR00T LeRobot v2 ↔ Cosmos pickle ↔ DreamDojo MP4
   ```

4. **프로젝트 인사이트 정리**

   - VLA vs Video-Policy vs World Model 패러다임 비교
   - 각 접근법의 이론적/실용적 장단점
   - 향후 통합 연구 방향 제안

**결과물:**

- ✅ 세 모델 크로스 분석 보고서
- ✅ 이론적 통합 파이프라인 설계 문서
- ✅ 데이터 포맷 변환 명세서

---

### Week 11-12: 비교 분석 보고서 및 프로젝트 마무리

#### Week 11: 종합 벤치마킹 및 분석 보고서

**학습 목표:**

- 12주간의 실험 결과 종합 분석
- 각 모델의 실용적 배포 시나리오 도출
- 재현 가능한 실험 환경 문서화

**실습 내용:**

1. **종합 벤치마킹 정리 (OMX 실측 기준)**

   ```python
   # ACT 베이스라인 (OMX 공식 정책) - 실측 데이터
   act_results = {
       "simple_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "complex_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "training_data": "50-100 demos (OMX 수집)",
       "training_gpu": "1x RTX 4090, ~1-3h",
       "inference_speed": "CPU 가능, GPU 시 <10ms",
       "note": "OMX 기본 지원, Physical AI Tools 통합",
   }

   # GR00T N1.6 (파인튜닝) - OMX 실측 데이터
   groot_results = {
       "simple_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "complex_tasks": {"success_rate": ..., "avg_time_ms": ...},
       "training_data": "150+ demos (OMX 수집, LeRobot v2 변환)",
       "training_gpu": "1x H100, ~8h",
       "inference_speed": "~44ms (RTX 4090)",
       "action_dim": 6,  # OMX 5-DOF + 1 gripper
       "note": "ACT 대비 성능 차이 분석",
   }

   # Cosmos Policy (추론 전용) - LIBERO 벤치마크
   cosmos_results = {
       "libero_spatial": ...,
       "libero_object": ...,
       "libero_goal": ...,
       "libero_long": ...,
       "inference_gpu": "1x (6-10GB VRAM)",
       "note": "파인튜닝 미수행 (8x H100 필요), OMX 직접 배포 불가",
   }

   # DreamDojo (추론 전용) - 롤아웃 품질
   dreamdojo_results = {
       "rollout_quality": ...,  # FID, SSIM
       "physics_consistency": ...,
       "long_horizon_stability": ...,
       "inference_gpu": "1x A100+",
       "note": "증류/텔레오퍼레이션 미공개, 시뮬레이션 전용",
   }
   ```

2. **실용적 배포 시나리오 매트릭스 (OMX 기준)**

   | 시나리오 | 권장 모델 | 이유 |
   |---------|----------|------|
   | OMX 빠른 프로토타이핑 | ACT | OMX 공식 지원, 빠른 학습, CPU 추론 가능 |
   | OMX 고성능 조작 | GR00T N1.6 | 언어 이해, 크로스 플랫폼, 대규모 사전학습 |
   | 범용 로봇 제어 (언어 명령) | GR00T N1.6 | 유일하게 NL 명령 직접 지원 |
   | 정밀 조작 (소량 데이터) | Cosmos Policy | 50개 시연으로 SOTA, 데이터 효율적 |
   | 정책 안전성 사전 검증 | DreamDojo | 실제 배포 전 시뮬레이션 평가 |
   | 합성 데이터 생성 | DreamDojo | 44K시간 사전 훈련 물리 지식 |
   | 저비용/교육용 배포 | ACT | GPU 불필요, 학습 곡선 낮음 |

3. **학습된 교훈 (Lessons Learned)**

   - 논문 의사코드 vs 실제 API의 괴리
   - GPU 요구사항의 현실적 제약
   - 데이터 포맷 표준화의 중요성
   - Docker 기반 ML 환경의 장단점

4. **코드 정리 및 재현 가능성 확보**

   - 전체 실험 스크립트 정리
   - Docker/환경 설정 문서화
   - 데이터 변환 유틸리티 코드

**결과물:**

- ✅ 종합 비교 분석 보고서 (30+ 페이지)
- ✅ 재현 가능한 실험 코드 (GitHub)
- ✅ 실용적 배포 가이드

---

#### Week 12: 최종 발표 및 프로젝트 마무리

**학습 목표:**

- 최종 발표 자료 준비
- 향후 연구 방향 제안
- 프로젝트 회고

**실습 내용:**

1. **OMX 라이브 데모 준비 (ACT vs GR00T)**

   ```python
   from gr00t.policy.gr00t_policy import Gr00tPolicy
   from gr00t.data.embodiment_tags import EmbodimentTag

   # GR00T 최적 체크포인트 로드
   groot_policy = Gr00tPolicy(
       model_path="checkpoints/groot_omx/best_model",
       embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
       device="cuda"
   )

   # ACT 모델은 Physical AI Tools 또는 LeRobot CLI로 로드
   # lerobot-record --control.policy.path=outputs/train/act_omx_best

   # 라이브 데모: OMX에서 동일 작업으로 ACT vs GR00T 비교
   demo_tasks = [
       "pick up the red cube",        # 단순 Pick
       "place the cube on the plate",  # Pick & Place
       "stack the cubes"               # 다단계 조작
   ]
   ```

2. **데모 비디오 제작**

   - ACT vs GR00T: OMX 동일 작업 나란히 비교 영상
   - GR00T 파인튜닝 전/후 성능 비교 영상
   - Cosmos Policy LIBERO 추론 시연
   - DreamDojo 롤아웃 생성 시연
   - 네 모델(ACT, GR00T, Cosmos, DreamDojo)의 출력 비교 영상

3. **발표 자료 준비**

   - 프로젝트 개요: OMX 하드웨어 + Option C 하이브리드 접근법
   - OMX 데이터 수집 파이프라인 (Physical AI Tools + LeRobot)
   - ACT vs GR00T 비교: 학습 효율, 성공률, 추론 속도, 일반화 능력
   - 네 모델의 기술적 차이 (아키텍처 다이어그램)
   - Cosmos Policy / DreamDojo 추론 분석
   - 실용적 배포 시나리오 및 향후 방향

4. **향후 연구 방향 제안**

   - OMX 멀티태스크 학습 확장 (10+ 작업)
   - ACT의 다른 정책 (Diffusion Policy, VQ-BeT) 대체 실험
   - 8x H100 접근 시 Cosmos Policy OMX 데이터 파인튜닝
   - DreamDojo 증류 파이프라인 공개 후 실시간 시뮬레이션
   - GR00T + ACT 앙상블 정책 연구
   - OMX 듀얼암(Bimanual) 확장

**최종 결과물:**

- ✅ 최종 발표 슬라이드 (30분)
- ✅ 데모 비디오 (OMX ACT vs GR00T 라이브 + 비교 분석)
- ✅ 완전한 코드베이스 (GitHub, 재현 가능)
- ✅ 기술 보고서 패키지
- ✅ 프로젝트 회고 문서

**발표 구성:**

1. **도입** (5분): OMX 하드웨어 소개 + Option C 접근법 선택 이유
2. **데이터 파이프라인** (5분): OMX 텔레오퍼레이션 → 데이터 수집 → 포맷 변환
3. **ACT vs GR00T** (10분): 학습/배포/성능 비교 실험 결과 (OMX 실측)
4. **추론 비교** (5분): Cosmos Policy LIBERO + DreamDojo 롤아웃 분석
5. **라이브 데모** (5분): OMX에서 ACT와 GR00T 동일 작업 실시간 시연
6. **결론** (5분): 배운 점, 향후 방향, 통합 파이프라인 비전

---

## 9. 프로젝트 성공을 위한 팁

### 리소스 관리

**GPU 리소스 (Option C 하이브리드 + OMX 기준):**

- ACT 베이스라인 학습: 1x RTX 3090/4090 충분 (~1-3시간)
- ACT 추론: CPU 가능 (GPU 불필요), GPU 시 <10ms
- GR00T N1.6 파인튜닝: 1x H100 또는 L40 권장 (A6000도 가능하나 느림)
- GR00T N1.6 추론: RTX 4090 (44ms, 22.8 Hz) 또는 H100 (38ms, 26.3 Hz)
- Cosmos Policy 추론: 1x GPU, 6-10 GB VRAM (RTX 3090 이상)
- DreamDojo 추론: 1x A100 40GB+ 권장
- ⚠️ Cosmos Policy 파인튜닝: 8x H100 80GB (본 프로젝트 범위 외)
- ⚠️ DreamDojo 후훈련: 8 GPU 노드 H100 80GB (본 프로젝트 범위 외)
- 클라우드 GPU 옵션: Lambda Labs, RunPod, AWS p4d/p5

**시간 관리:**

- 훈련 시간을 고려한 스케줄링
- 주말에 장시간 훈련 작업 배치
- 병렬 실험으로 시간 단축

### 일반적인 함정 및 해결책

**문제 1: Sim-to-Real 갭**

- 해결: 도메인 랜덤화, 실제 데이터 혼합, 점진적 적응

**문제 2: 데이터 품질 불량**

- 해결: 체계적 데이터 검증, 이상치 제거, 재수집

**문제 3: 과적합**

- 해결: 데이터 증강, 정규화, 조기 종료

**문제 4: 훈련 불안정**

- 해결: Learning rate 조정, 그래디언트 클리핑, 배치 크기 증가

### 확장 아이디어 (Option C 완료 후)

**GPU 리소스 확보 시:**

1. **Cosmos Policy 파인튜닝**: 8x H100 클라우드 인스턴스로 LeRobot 데이터 직접 파인튜닝
2. **DreamDojo 후훈련**: 증류 파이프라인 공개 후 실시간 10 FPS 시뮬레이션 구현
3. **세 모델 I/O 통합 어댑터**: GR00T LeRobot v2 ↔ Cosmos pickle ↔ DreamDojo MP4 자동 변환

**기본 프로젝트 완료 후:**

4. **멀티모달 입력 추가**: 촉각, 힘 센서 통합
5. **언어 명령 확장**: 복잡한 자연어 지시 처리
6. **다중 로봇 협업**: 여러 LeRobot 협력 작업
7. **온라인 학습**: 실시간 피드백 기반 개선
8. **실제 응용**: 특정 산업 작업 적용 (분류, 포장 등)

### 학습 리소스

**논문:**

- GR00T N1: "An Open Foundation Model for Generalist Humanoid Robots"
- Cosmos Policy: "Fine-Tuning Video Models for Visuomotor Control"
- DreamDojo: "A Generalist Robot World Model from Large-Scale Human Videos"

**코드 저장소:**

- https://github.com/NVIDIA/Isaac-GR00T
- https://github.com/NVlabs/cosmos-policy
- https://github.com/dreamdojo-world/dreamdojo
- https://github.com/huggingface/lerobot

**커뮤니티:**

- NVIDIA Developer Forums
- Isaac Sim Discord
- LeRobot Community
- Robotics StackExchange

---

## 10. 예상 프로젝트 결과

### 정량적 목표 (Option C 하이브리드)

| 메트릭 | 목표 | 우수 | 비고 |
| ------ | ---- | ---- | ---- |
| GR00T 단순 작업 성공률 | >80% | >90% | 파인튜닝 후 실측 |
| GR00T 복잡 작업 성공률 | >60% | >75% | 파인튜닝 후 실측 |
| Cosmos LIBERO 재현율 | >95% | >98% | 논문 결과 재현 |
| DreamDojo 롤아웃 안정성 | 30초+ | 60초+ | 연속 생성 유지 |
| GR00T 추론 속도 | <100ms | <50ms | RTX 4090 기준 |
| 비교 분석 보고서 | 20+ 페이지 | 30+ 페이지 | 세 모델 종합 |

### 학습 성과

**기술적 역량:**

- ✅ VLA 모델 파인튜닝 및 배포 (GR00T N1.6)
- ✅ Video-to-Policy 모델 추론 및 평가 (Cosmos Policy)
- ✅ 세계 파운데이션 모델 추론 및 롤아웃 분석 (DreamDojo)
- ✅ 실제 API vs 논문 의사코드 갭 분석 능력
- ✅ Docker 기반 ML 환경 구축

**실무 경험:**

- ✅ 대규모 딥러닝 파인튜닝 (GR00T)
- ✅ 로봇 하드웨어 제어 및 데이터 수집
- ✅ 다양한 데이터 포맷 변환 (LeRobot v2, pickle, MP4)
- ✅ 체계적 벤치마킹 및 비교 분석
- ✅ 실험 설계 및 재현 가능한 연구

**Option C 접근법의 장점:**

- 12주 내 현실적으로 달성 가능한 목표 설정
- GR00T 파인튜닝을 통한 심화 실습 경험
- GPU 비용 최적화 (추론만으로도 모델 특성 충분히 이해)
- 향후 GPU 리소스 확보 시 Cosmos Policy/DreamDojo 파인튜닝으로 확장 가능

이 12주 커리큘럼을 통해 NVIDIA의 세 가지 로봇 AI 모델을 실제 소스코드 수준에서 이해하고, GR00T N1.6 파인튜닝의 실전 경험과 함께 Cosmos Policy, DreamDojo의 추론 기반 비교 분석 역량을 확보할 수 있습니다.

NVIDIA의 세 가지 모델은 각각의 고유한 강점을 통해 로봇공학의 미래를 재정의하고 있습니다. Option C 하이브리드 접근법을 통해 GPU 리소스 제약 하에서도 세 모델의 핵심 특성을 실증적으로 파악하고, 향후 리소스 확보 시 통합 시스템으로의 확장 기반을 마련할 수 있습니다.

---

## 11. GR00T-IDM 연계 활용

> **참조 프로젝트**: [GR00T-IDM-Documentation](../GR00T-IDM-Documentation/)
>
> GR00T-IDM (Inverse Dynamics Model)은 비디오에서 로봇 action을 추출하는 모델입니다.
> DreamDojo 합성 비디오와 결합하면 VLA 학습 데이터를 크게 증강할 수 있습니다.

### VLA ↔ IDM 데이터 포맷 차이

| 항목 | GR00T N1.6 VLA | GR00T IDM |
|------|---------------|-----------|
| 비디오 dtype | `float32` [0.0, 1.0] | `uint8` [0, 255] |
| 비디오 해상도 | 224×224 | 256×256 |
| 관절 이름 | `joint1~5, gripper` (기능 무관) | `shoulder_pan, shoulder_lift, ...` (해부학적) |
| Action 표현 | joints=RELATIVE, gripper=ABSOLUTE | 모두 absolute |
| 설정 포맷 | Python (`register_modality_config`) | JSON (`modality.json`) |
| 입력 프레임 | 1 프레임 | 2 프레임 (t, t+1) |

### 관절 이름 매핑 테이블

| VLA (GR00T N1.6) | IDM (GR00T-Dreams) | 기능 |
|-------------------|-------------------|------|
| `joint1` | `shoulder_pan` | 어깨 수평 회전 |
| `joint2` | `shoulder_lift` | 어깨 수직 회전 |
| `joint3` | `elbow_flex` | 팔꿈치 굴곡 |
| `joint4` | `wrist_flex` | 손목 굴곡 |
| `joint5` | `wrist_roll` | 손목 회전 |
| `gripper` | `gripper` | 그리퍼 개폐 |

매핑 유틸리티: `utils/omx_constants.py`의 `OMX_JOINT_MAPPING` / `OMX_JOINT_MAPPING_INV`

### DreamDojo + IDM 시너지 파이프라인

```
DreamDojo 합성 비디오 (uint8, 480×640)
    ↓ resize + 2-frame pairing
GR00T-IDM pseudo action labeling (uint8, 256×256)
    ↓ 관절 이름 매핑 (IDM→VLA)
품질 평가 (jerk, temporal consistency → grade ≥ B)
    ↓ dtype 변환 (uint8→float32)
GR00T N1.6 VLA 파인튜닝 데이터 증강
```

이 파이프라인의 핵심 가치:
- **데이터 증강**: 실제 로봇 시연 없이 합성 데이터로 VLA 학습 데이터 확보
- **비용 절감**: 텔레오퍼레이션 대비 대규모 데이터 생성 가능
- **품질 보장**: IDM pseudo label 품질 평가로 저품질 데이터 필터링

상세 구현: `scripts/week10_cross_model_analysis.py`

### 비디오 dtype 변환

```python
from utils.omx_constants import convert_video_vla_to_idm, convert_video_idm_to_vla

# VLA float32 [0,1] → IDM uint8 [0,255]
idm_video = convert_video_vla_to_idm(vla_video)

# IDM uint8 [0,255] → VLA float32 [0,1]
vla_video = convert_video_idm_to_vla(idm_video)
```

