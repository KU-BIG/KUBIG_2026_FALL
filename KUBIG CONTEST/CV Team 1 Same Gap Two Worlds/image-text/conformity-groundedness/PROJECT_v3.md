# Conformity × Groundedness: 미술 도메인에서 Modality Gap 재해석

> Double-Ellipsoid(ICML 2025)의 conformity 개념을 SemArt에 적용하되, **텍스트 내부 독특성(conformity)과 이미지-텍스트 결부성(groundedness)을 분리**하여 2차원으로 분석한다.

---

## 0. 이 문서를 읽는 Claude Code에게

1. **Phase 0(재현)과 Phase 1(전제 검증)을 건너뛰지 마라.** Phase 1에서 thin shell 가정이 깨지면 이후 설계를 수정해야 한다. **결과가 나오면 멈추고 보고할 것.**
2. **학습은 하지 않는다.** 전부 frozen CLIP에서 임베딩을 뽑는 사후 분석이다.
3. **정규화 전 임베딩과 정규화 후 임베딩을 모두 저장하라.** 이 프로젝트는 두 공간을 오간다. 나중에 다시 뽑으면 낭비다.
4. **결과를 예단하지 마라.** 가설이 반증되는 경우도 유효한 결과다(§8).
5. 데이터 접근, 라이선스, 이미지 링크 문제는 임의 우회 금지. 막히면 보고.

---

## 1. 배경

### 1.1 Modality gap

CLIP은 이미지 인코더와 텍스트 인코더를 독립적으로 두고 contrastive 학습을 한다. 두 출력은 공유 임베딩 공간에 놓이는데, 학습 후 **이미지 무리와 텍스트 무리가 분리된 두 영역에 놓인다.** 짝짓기는 성공했는데 두 무리가 통째로 떨어져 있다.

- CLIP: https://arxiv.org/abs/2103.00020
- Mind the Gap (Liang et al., NeurIPS 2022): https://openreview.net/pdf?id=S7Evzt9uit3

**L2M 지표:**

```
Δ_gap = mean(L2 정규화된 이미지 임베딩) - mean(L2 정규화된 텍스트 임베딩)
L2M   = ||Δ_gap||
```

개별 임베딩을 **먼저 정규화한 뒤 평균**낸다. CLIP 기준값 **0.82**.

### 1.2 Two Effects, One Trigger (Schrodi et al., ICLR 2025)

- 논문: https://arxiv.org/abs/2404.07983
- 코드: https://github.com/lmb-freiburg/two-effects-one-trigger

**주장:** gap의 원인은 이미지와 캡션의 **정보 불균형**이다. 캡션은 이미지 정보의 일부만 담고, 이미지 인코더는 캡션에 뭐가 쓰일지 모르므로 정렬이 원천적으로 제한된다. 그래서 모델은 정렬 대신 균일성을 택해 두 모달리티를 서로 멀어지게 만든다.

**제안 지표 RMG** (L2M은 짝 정보와 공간 크기를 무시한다는 비판):

```
        mean_i d(x_i, y_i)
RMG = ──────────────────────────────────────
      [모달리티 내부 평균 거리] + mean_i d(x_i, y_i)
```

**중요한 부수 발견:** 소수 차원만 gap을 만든다. 사후 조작(차원 제거·평행이동)은 gap을 닫아도 성능을 못 올린다. 두 모달리티의 이웃 순서가 다르기 때문(정규화 Kendall-τ 거리 ≈ 0.5, 사실상 무작위).

> ⚠️ 이 마지막 결론은 **현재 논쟁 중**이다. 일부 후속 연구는 원칙적 지표로 gap을 줄이면 retrieval과 zero-shot이 개선된다고 보고했다. 단정적으로 쓰지 말 것.

### 1.3 Double-Ellipsoid Geometry of CLIP (Levi & Gilboa, ICML 2025) ← 직접적 출발점

- 논문: https://arxiv.org/abs/2411.14517

**핵심 발견:**

1. **정규화 전** 임베딩에서 이미지와 텍스트가 **원점에서 벗어난 두 개의 기울어진 타원체 껍질** 위에 놓인다. 단 두 개의 feature로 100% 선형 분리 가능(MS-COCO).
2. **Thin shell 현상:** 질량이 평균으로부터 특정 반경 범위에 집중되며, 평균 근처에는 질량이 없다.
3. **Conformity 정의:**

```
C(v) = E_{u ∈ S, u≠v} [cos(v, u)]
```

즉 **그 샘플이 집합 내 다른 모든 샘플과 갖는 평균 코사인 유사도.** 낮으면 독특하고, 높으면 전형적이다.

4. **추정 정리:** thin shell 가정 하에서 conformity는 모달리티 평균 벡터와의 코사인 유사도로 근사된다.

```
Ĉ(v) = a · cos(m, v) + b
```

MS-COCO에서 피어슨 상관 **0.9998**. (텍스트 a=1.411, b=−0.008 / 이미지 a=1.461, b=−0.002)

5. **메커니즘:** 자주 등장하는 개념일수록 false negative가 많아 불확실성이 크고, 그런 샘플은 평균 벡터 가까이 임베딩된다(**semantic blur**). 원점에서 벗어난 타원체가 이 blur를 가능하게 한다.

6. **Gap에 대한 설명:** modality gap은 **이미지와 텍스트의 conformity 분포를 정합시키도록** 최적화되어 있다. 오프셋 파라미터 α로 실험했을 때 KL divergence가 α≈0(학습된 상태)에서 최소(≈0.14).

```
v' = v − α · m      (α=0: 현재 CLIP, α=1: 원점 중심)
```

7. 노름은 코사인 유사도에 무관하지만 의미를 담는다. MS-COCO에서 가장 큰 노름은 "이게 무슨 이미지인지 모르겠다" 같은 이상한 캡션이었다.

---

## 2. 문제의식: conformity는 이미지를 보지 않는다

### 2.1 결정적 관찰

**conformity 계산에는 이미지가 전혀 들어가지 않는다.** 텍스트 집합 내부에서만 계산된다. 따라서 conformity는 "이 텍스트가 다른 텍스트에 비해 얼마나 독특한가"만 잰다.

그러면 conformity가 낮은(=독특한) 텍스트는 **두 종류가 섞여 있다.**

| 유형 | 예시 | 이미지 결부 |
|---|---|---|
| 독특 + **접지** | "왼쪽 창가의 인물이 백합을 들고 있다" | 그림을 봐야 쓸 수 있음 |
| 독특 + **비접지** | "1520년 로마 약탈 직후 그려졌다" | 그림을 안 봐도 쓸 수 있음 |

**둘 다 conformity가 낮게 나온다.** 그러나 modality gap 연구에서 이 둘은 완전히 다른 의미를 갖는다.

### 2.2 MS-COCO에서는 이 문제가 드러나지 않는다

COCO 캡션은 전부 이미지를 서술한다. 따라서 **conformity가 낮은 캡션은 대체로 접지성도 높다.** 두 축이 강하게 상관되어 있어 구분할 필요가 없었고, 원 논문도 구분하지 않았다.

**SemArt에서는 그 상관이 깨진다.** 미술 해설에는 독특하면서 접지되지 않은 텍스트가 대량으로 존재한다.

### 2.3 2차원 프레임

|  | 접지 높음 | 접지 낮음 |
|---|---|---|
| **conformity 높음** (전형적) | "성모와 아기 예수" | "16세기 이탈리아 회화" |
| **conformity 낮음** (독특) | "왼쪽 창가의 백합" | **"1520년 로마 약탈 직후"** ← 미탐색 영역 |

우측 하단이 **MS-COCO에는 거의 없고 SemArt에는 풍부한 영역**이다.

---

## 3. 연구 질문

> **RQ1.** SemArt에서 conformity와 groundedness는 얼마나 상관되는가? COCO와 비교하면?
>
> **RQ2.** Double-Ellipsoid의 conformity 근사식은 이질적인 긴 텍스트에서도 성립하는가? (thin shell 가정 검증)
>
> **RQ3.** "gap = conformity 분포 정합"이라는 설명은 미술 도메인에서도 성립하는가? KL 최소점이 여전히 α=0인가?
>
> **RQ4 (핵심).** conformity 분포는 정합인데 접지성이 어긋난 상태가 존재하는가? 그때 gap과 실제 검색 성능은 어떻게 되는가?

### 3.1 RQ4가 핵심인 이유

Double-Ellipsoid의 주장대로면 **conformity 분포만 맞으면 gap이 최적**이다. 그런데 접지성이 없으면 실제 검색 성능은 나쁠 것이다.

> **"기하학적으로는 정합인데 의미적으로는 어긋난" 상태**가 관찰되면, 논문의 설명이 불완전하다는 증거가 된다.

**MS-COCO에서는 두 축이 얽혀 있어 이를 관찰할 수 없다. SemArt여야만 가능하다.** 이것이 데이터 선택의 진짜 근거다.

---

## 4. 데이터셋

### 4.1 SemArt (주력)

- 프로젝트: https://noagarcia.github.io/SemArt/
- 다운로드(DOI): https://doi.org/10.17036/researchdata.aston.ac.uk.00000380
- 코드: https://github.com/noagarcia/SemArt
- 논문: https://noagarcia.github.io/docs/VISART2018.pdf

| 항목 | 내용 |
|---|---|
| 규모 | 21,384점 (train 19,244 / val 1,069 / test 1,069) |
| 모달리티 | 유럽 고전 회화 ↔ 예술 해설(artistic comment) |
| 페어링 | 1:1 |
| 속성 | Author, Title, Date, Technique, Type, School, Timeframe |
| 내장 벤치마크 | Text2Art challenge (양방향 retrieval) |
| 이미지 출처 | Web Gallery of Art |

**확인 필요 (임의 판단 금지):**

- [ ] DOI 페이지 라이선스 문구
- [ ] Web Gallery of Art 이용 약관
- [ ] **이미지 링크 생존율** (2018년 데이터셋)
- [ ] 죽은 링크가 많으면 대안 논의 (https://github.com/georgeblck/art-datasets)

### 4.2 MS-COCO (필수 대조군)

- https://cocodataset.org/

**두 가지 역할:**
1. Phase 0 검산 (L2M = 0.82, conformity 피어슨 = 0.9998)
2. **conformity-groundedness 상관이 높은 대조 사례** — SemArt와의 대비가 논지의 핵심

### 4.3 CUB-200-2011 (선택)

- https://data.caltech.edu/records/65de6-vp158

극단적으로 지시적인 캡션. 접지성이 최대인 끝점.

---

## 5. 인코더

| 구분 | 모델 | 근거 |
|---|---|---|
| 기본 | OpenCLIP **ViT-B/32**, `pretrained='openai'` | Double-Ellipsoid가 ViT-B/32(n=512)로 분석. 수치 직접 대조 |
| 보조 | ViT-B/16 | Schrodi가 사용. RMG 등 비교 |
| 보조 | ViT-L/14 (n=768) | 원 논문이 부록에서 확인한 설정 |
| 비교 | LAION-2B, DataComp | 사전학습 데이터별 차이 |

- OpenCLIP: https://github.com/mlfoundations/open_clip

> **주의:** 논문마다 기준 모델이 다르다. Double-Ellipsoid는 ViT-B/32, Schrodi는 ViT-B/16. **어느 수치와 대조할지에 따라 모델을 맞출 것.** 주력은 ViT-B/32로 하고, RMG 비교 시 ViT-B/16을 병행한다.

**학습하지 않는다.** 전부 frozen 모델의 사후 분석이다.

---

## 6. 측정

### 6.1 두 개의 공간, 두 개의 저장

이 프로젝트는 **정규화 전후 두 공간을 모두 사용한다.**

```python
import torch, torch.nn.functional as F, open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

with torch.no_grad():
    img_raw = model.encode_image(images)   # 정규화 전 — 타원체 분석용
    txt_raw = model.encode_text(tokens)    # 정규화 전

img_emb = F.normalize(img_raw, dim=-1)     # 정규화 후 — L2M, RMG용
txt_emb = F.normalize(txt_raw, dim=-1)

# 둘 다 저장할 것
```

| 공간 | 용도 |
|---|---|
| **정규화 전** | 타원체 기하학, thin shell, conformity, α 스윕 |
| **정규화 후** | L2M, RMG, Kendall-τ, retrieval |

> **흔한 실수:** Mind the Gap의 cone effect 실험과 L2M 계산은 서로 다른 레이어를 쓴다. cone effect는 ResNet의 경우 마지막 linear 직전, CLIP의 ViT/Text Transformer는 최종 레이어. Δ_gap은 projection 이후 정규화된 공유 공간.

### 6.2 Conformity (독특성 축)

```python
def conformity_true(embs):
    """정의 그대로. O(N^2)이므로 검증용으로만."""
    E = F.normalize(embs, dim=-1)
    S = E @ E.T
    N = S.shape[0]
    S.fill_diagonal_(0)
    return S.sum(1) / (N - 1)

def conformity_est(embs):
    """추정식. cos(mean, v)에 비례."""
    m = embs.mean(0)
    return F.cosine_similarity(embs, m.unsqueeze(0), dim=-1)
```

**두 값의 피어슨 상관을 반드시 보고할 것.** 원 논문 MS-COCO 기준 0.9998. SemArt에서 이보다 낮으면 thin shell 가정이 약화된 것이며, 그 자체가 결과다.

계수 a, b는 선형회귀로 추정한다.

### 6.3 Groundedness (접지성 축)

**절대 코사인 유사도는 신뢰하기 어렵다. 순위로 변환할 것.**

```
groundedness(i) = 텍스트 i가 자기 짝 이미지를 전체 갤러리에서 몇 등으로 찾는가
```

- 순위를 [0,1]로 정규화 (1 = 완벽히 특정, 0 = 전혀 못 찾음)
- 갤러리 크기를 데이터셋 간에 고정 (예: 5,000)
- 양방향 모두 계산: text→image, image→text

보조 지표로 짝 코사인 유사도 원값도 함께 기록하되, 주 분석은 순위 기반으로 한다.

### 6.4 Gap 지표

1. **L2M** — Liang 기준, 비교 가능성
2. **RMG** — Schrodi. L2M만 쓰면 지적당한다
3. **차원별 기여도** — |평균 차이|를 차원별 정렬
4. **이웃 순서 Kendall-τ 거리** — Schrodi Takeaway 3과 대화

### 6.5 타원체 기하학

정규화 전 공간에서:

- 모달리티별 평균 벡터 m, 표준편차 벡터 σ, ‖m‖/‖σ‖ 비율
- 노름 분포 히스토그램 (thin shell 확인 — 평균 근처에 질량이 없는지)
- feature별 분리도: `Sep(l) = |m_i(l) − m_t(l)| / sqrt(var_i(l) + var_t(l))`
- 상위 2개 feature로 선형 SVM 분리 정확도 (원 논문은 MS-COCO에서 100%)
- 공분산 off-diagonal dominance (타원체 기울기)

### 6.6 표본 처리

- n = 5,000 고정 (원 논문들과 동일)
- 데이터셋 간 비교 시 **반드시 샘플 수 고정**
- 이미지당 캡션 1개로 통일 (COCO는 5개)
- 부트스트랩 95% 신뢰구간
- N별 수렴 곡선 (N = 100, 500, 1000, 5000)

---

## 7. 텍스트 변이체 사다리

SemArt의 강점은 **같은 그림에 대해 성격이 다른 텍스트를 만들 수 있다는 것**이다. 이미지는 한 장도 바뀌지 않는다.

| 변이체 | 구성 | 예상 conformity | 예상 groundedness |
|---|---|---|---|
| V0 | `a {Type} painting by {Author}, {School} school, {Timeframe}` | 매우 높음 | 낮음 |
| V1 | Title 필드만 | 높음 | 중간 |
| V2 | 해설 중 **시각 서술문**만 | 중간 | **높음** |
| V3 | 해설 중 **해석·맥락문**만 | 낮음 | **낮음** |
| V4 | 해설 전문 | 매우 낮음 | 중간 |

**V2 대 V3가 핵심 비교다.** 둘 다 문장 단위 추출이라 길이가 비슷하다 → 길이 교란 제거. 예상되는 conformity 차이는 작지만 **groundedness 차이는 클 것**이다. 이 대비가 2차원 프레임의 존재 이유를 입증한다.

**문장 분류 절차:**

1. 500문장 수동 라벨링 → 기준 확립
2. 그 기준으로 나머지 자동 분류
3. 분류기 정확도 보고

완벽할 필요 없다. 분류 노이즈는 두 그룹을 섞어 **차이를 줄이는 방향으로만** 작용하므로, 그럼에도 차이가 보이면 보수적 증거다.

**V0과 V4는 주관이 개입하지 않는 끝점**이므로, 이 둘만으로도 최소한의 주장은 성립한다.

### 7.1 CLIP의 실효 토큰 길이

CLIP 텍스트 인코더의 명목 상한은 77토큰이지만, **실효 길이는 약 20토큰**이다. 뒤쪽 positional embedding은 학습 중 거의 활성화되지 않는다. "77 이내면 안전"이라는 전제는 틀렸다.

V4(해설 전문)는 이를 초과할 가능성이 높다. 세 가지 처리 방식을 비교할 것:

1. Truncation (앞 77토큰) — 기준선
2. 문장 분할 후 임베딩 평균 — 정보 손실 최소
3. (선택) Long-CLIP — 단, 파인튜닝된 다른 모델이므로 별도 트랙

---

## 8. 실행 순서

### Phase 0 — 재현 (필수)

**목표 세 가지를 COCO에서 확인:**

| 항목 | 기대값 |
|---|---|
| L2M | 0.82 |
| conformity 실제 vs 추정 피어슨 | 0.9998 |
| 상위 2 feature 선형 SVM 분리 | 100% |

안 나오면 의심할 것: 정규화 순서, 체크포인트, 모델(B/32 vs B/16), 샘플 수.

**통과해야 다음으로 간다.**

### Phase 1 — Thin shell 검증 ⭐ 최우선

**SemArt에서 conformity 근사식이 성립하는가?**

- artistic comment 토큰 길이 분포 히스토그램 (20/40/77 초과 비율)
- 정규화 전 노름 분포 — 평균 근처에 질량이 없는가
- conformity 실제 vs 추정 피어슨 상관

**결과에 따라 설계가 갈린다. 보고하고 논의할 것.**

| 결과 | 의미 | 다음 |
|---|---|---|
| 피어슨 ≈ 0.99 | thin shell 성립 | 추정식 사용, 계속 진행 |
| 피어슨 0.9~0.99 | 약화됨 | 실제 conformity를 쓰되 한계 명시 |
| 피어슨 < 0.9 | **가정 붕괴** | 그 자체가 결과. 설계 재논의 |

### Phase 2 — 2차원 산점도 (핵심 그림)

**x축 conformity, y축 groundedness로 산점도를 그린다.**

- SemArt V4 전체
- COCO 전체
- 두 축의 상관계수를 각각 보고

**예측:** COCO는 강한 음의 상관(독특할수록 접지 높음), SemArt는 약한 상관 또는 무상관.

**이 한 장이 프로젝트의 얼굴이다.**

이어서 사다리 V0~V4를 같은 평면에 올려 궤적을 그린다. V2와 V3가 실제로 groundedness 축에서 갈리는지 확인.

### Phase 3 — 타원체 기하학

SemArt에서 §6.5 항목 전부 측정. COCO와 대조.

- 두 모달리티가 여전히 선형 분리되는가? 몇 개 feature로?
- 타원체 오프셋 ‖m‖/‖σ‖가 COCO와 다른가?
- gap을 만드는 feature가 COCO와 같은 feature인가? ← Schrodi Takeaway 2와 연결

### Phase 4 — α 스윕과 conformity 정합 (RQ3)

원 논문 Fig 11 재현. `v' = v − α·m`으로 α를 −1에서 1까지 쓸면서:

- 이미지·텍스트 conformity 분포의 KL divergence
- contrastive loss
- **동시에 retrieval 성능** ← 원 논문에 없는 축

**핵심 확인:** KL 최소점이 α=0인가?

| 결과 | 의미 |
|---|---|
| α=0에서 최소 | 논문 주장이 도메인을 넘어 성립 |
| α≠0에서 최소 | **CLIP의 gap이 미술 도메인에서 잘못 보정됨.** gap은 사전학습 분포의 함수 |

### Phase 5 — RQ4: 정합인데 어긋난 상태 (핵심)

Phase 4에서 **KL이 최소인 α**와 **retrieval이 최대인 α**를 비교한다.

> 두 지점이 다르면, "conformity 분포 정합"만으로는 좋은 표현을 설명할 수 없다는 직접적 증거다.

COCO에서는 두 지점이 일치할 것으로 예상되고, SemArt에서는 어긋날 것으로 예상된다. **그 어긋남의 크기가 이 프로젝트의 주요 결과가 된다.**

### Phase 6 — 빈도 분해 (선택, 통제 실험 겸용)

원 논문 Prediction 1은 "자주 등장하는 개념일수록 평균 벡터 가까이 임베딩된다"이다. MS-COCO에서는 **평가셋 빈도와 사전학습 빈도가 거의 같아** 구분되지 않았다.

**SemArt에서는 분리 가능:**

- 평가셋 내 빈도: SemArt에 해당 School/Type이 몇 점인가
- 사전학습 빈도 대리: 작가 유명도

> **RQ: conformity는 평가셋의 통계인가, 모델의 성질인가?**

**이것은 동시에 "CLIP이 명화를 외웠지 않냐"는 반박에 대한 답이기도 하다.** 유명도 변수를 정면으로 다루게 되므로 반드시 포함할 것.

### Phase 7 — 노름의 의미 (부록용)

정규화 전 노름이 큰 텍스트와 작은 텍스트를 정성적으로 살펴본다. 원 논문은 MS-COCO에서 노름이 큰 것이 "이게 무슨 이미지인지 모르겠다" 류의 이상한 캡션이었다고 보고했다. **SemArt에서는 무엇인가?**

---

## 9. 결과 해석 가이드

| 결과 | 의미 | 가치 |
|---|---|---|
| COCO는 두 축 상관 높고 SemArt는 낮음 | 2차원 프레임의 필요성 입증 | 프로젝트 전제 확립 |
| V2/V3가 groundedness에서만 갈림 | conformity 단독으로는 부족 | 프레임 정당화 |
| SemArt에서 KL 최소점이 α≠0 | gap은 사전학습 분포에 맞춰진 것 | 강한 결과 |
| KL 최적 α ≠ retrieval 최적 α | conformity 정합만으로는 불충분 | **핵심 기여** |
| 두 지점이 일치 | Double-Ellipsoid 설명이 도메인 넘어 성립 | 확증. 여전히 발표 가치 |
| thin shell 붕괴 (피어슨 < 0.9) | 이질적 텍스트에서 근사식이 깨짐 | 원 논문 한계 규명 |

**어떤 결과가 나와도 논지가 선다.** 정직하게 측정하라.

---

## 10. 예상 반박과 대응

**"conformity는 원래 이미지를 안 보는 지표인데 왜 문제 삼나"**
→ 문제 삼는 게 아니라 **범위를 명확히 하는 것**이다. 원 논문은 conformity 분포 정합으로 gap을 설명했는데, 그 설명이 접지성을 고려하지 않는다는 점을 지적하고 반례가 존재하는 도메인을 제시한다.

**"groundedness를 순위로 재면 갤러리 구성에 의존한다"**
→ 맞다. 그래서 갤러리 크기를 고정하고, 여러 갤러리 샘플링으로 부트스트랩 CI를 보고한다. 절대값이 아니라 **데이터셋 간 상대 비교**로만 해석한다.

**"CLIP이 명화를 외우고 있지 않나"**
→ 가능성 있음. **다만 이 프로젝트에는 유리하다** — 이미지 쪽이 익숙할수록 변수가 텍스트로 격리된다. Phase 6이 이 반박에 대한 직접적 대응이다.

**"해석문과 서술문 분류가 주관적이다"**
→ 기준 문서화, 샘플 공개, 분류기 정확도 보고. V0과 V4는 주관이 개입하지 않는 끝점.

**"77토큰 잘림 때문 아닌가"**
→ 세 가지 처리 방식 비교. V2·V3는 문장 단위라 애초에 짧다. 길이 고정 부분집합에서 재측정.

**"L2M만 봤다"**
→ RMG 병행. Double-Ellipsoid의 KL도 함께 보고하므로 지표가 세 개다.

---

## 11. 리포지토리 구조 제안

```
.
├── configs/
├── data/
│   ├── semart/
│   ├── coco/
│   └── cub/                # 선택
├── src/
│   ├── encode.py           # 임베딩 추출 (정규화 전/후 모두 저장)
│   ├── ladders.py          # 텍스트 변이체 V0~V4 생성
│   ├── conformity.py       # 실제/추정 conformity, 피어슨 검증
│   ├── groundedness.py     # 순위 기반 접지성
│   ├── geometry.py         # 타원체, thin shell, 분리도, ODD
│   ├── metrics.py          # L2M, RMG, Kendall-τ, 차원별 기여도
│   ├── alpha_sweep.py      # v' = v - α·m, KL, loss, retrieval
│   └── eval.py             # Text2Art retrieval, zero-shot classification
├── notebooks/
├── results/
│   ├── embeddings/         # 캐시 — 필수
│   └── figures/
└── PROJECT.md
```

**임베딩 캐싱은 선택이 아니다.** 변이체 5종 × 모델 4종 × 데이터셋 3개 조합이 나온다. 정규화 전/후를 모두 저장할 것.

---

## 12. 원 논문과 맞출 상수

| 항목 | 값 | 출처 |
|---|---|---|
| 표본 수 | n = 5,000 | 세 논문 공통 |
| 손실 평가 batch | 50 | Mind the Gap |
| 기준 온도 τ | 1/100 (비교: 1/50, 1) | Mind the Gap |
| 시드 | 3개, 95% CI | Schrodi |
| conformity 계수 (COCO, 텍스트) | a=1.411, b=−0.008 | Double-Ellipsoid |
| conformity 계수 (COCO, 이미지) | a=1.461, b=−0.002 | Double-Ellipsoid |
| α 스윕 범위 | −1 ~ 1 | Double-Ellipsoid |
| shift 공식 | `x' = Normalize(x − λ·Δ_gap)`, `y' = Normalize(y + λ·Δ_gap)` | Mind the Gap |
| 오프셋 공식 | `v' = v − α·m` (원점 방향) | Double-Ellipsoid |

> 두 공식이 다르다. Mind the Gap은 **두 모달리티를 서로에게** 이동시키고, Double-Ellipsoid는 **원점 방향으로** 이동시킨다. 혼동하지 말 것.

---

## 13. 참고 링크

### 논문

| | |
|---|---|
| CLIP | https://arxiv.org/abs/2103.00020 |
| Mind the Gap | https://openreview.net/pdf?id=S7Evzt9uit3 |
| Mind the Gap 코드 | https://github.com/Weixin-Liang/Modality-Gap |
| Two Effects, One Trigger | https://arxiv.org/abs/2404.07983 |
| Two Effects 코드 | https://github.com/lmb-freiburg/two-effects-one-trigger |
| **Double-Ellipsoid** | https://arxiv.org/abs/2411.14517 |
| SemArt | https://noagarcia.github.io/docs/VISART2018.pdf |

### 데이터

| | |
|---|---|
| SemArt 프로젝트 | https://noagarcia.github.io/SemArt/ |
| SemArt 다운로드 | https://doi.org/10.17036/researchdata.aston.ac.uk.00000380 |
| SemArt 코드 | https://github.com/noagarcia/SemArt |
| MS-COCO | https://cocodataset.org/ |
| CUB-200-2011 | https://data.caltech.edu/records/65de6-vp158 |
| 미술 데이터셋 목록 | https://github.com/georgeblck/art-datasets |

### 도구

| | |
|---|---|
| OpenCLIP | https://github.com/mlfoundations/open_clip |
| CLIP_benchmark | https://github.com/LAION-AI/CLIP_benchmark |

---

## 14. 첫 주 체크리스트

- [ ] OpenCLIP 설치, ViT-B/32 `openai` 로드
- [ ] COCO validation 준비
- [ ] **Phase 0**: L2M 0.82, conformity 피어슨 0.9998, SVM 100% 재현
- [ ] SemArt 다운로드, 이미지 링크 생존율 확인
- [ ] SemArt 라이선스 문구 확인
- [ ] artistic comment 토큰 길이 히스토그램
- [ ] **Phase 1**: SemArt thin shell 검증 (피어슨 상관)
- [ ] **결과 보고 후 논의** ← 여기서 멈출 것

**Phase 0과 Phase 1이 이 프로젝트의 전제를 결정한다. 거기까지 가서 멈추고 보고하라.**
