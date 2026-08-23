# 실험 결과 및 의사결정 기록

> 이 문서는 파이프라인을 실제로 실행하면서 나온 실측 결과와, 그 과정에서 내려진 주요 의사결정을 단계 순서대로 정리한 것이다. "확인된 사실"과 "추정·가설"을 구분해 표기했다.

---

## 1. Step 4 — Stage A: H1 1차 검증

**질문**: 텍스트 길이를 통제했을 때(visual 44.16토큰 vs contextual 48.77토큰, 차이 9.5%), 정보 종류만으로 gap 크기가 갈리는가?

**결과** (`step4_stage_a_results.json`):

| 지표 | visual | contextual | 판정 |
| --- | --- | --- | --- |
| $\|\Delta_{gap}\|$ | 0.8541 (CI 0.8523–0.8558) | 0.8609 (CI 0.8594–0.8624) | visual < contextual, **H1 방향**, CI 겹치지 않음 |
| Linear separability | 1.0 | 1.0 | 완전 포화 — 두 조건 구분에 무정보 |
| Paired cosine similarity | 0.3028 | 0.2989 | visual > contextual (t=7.96, p=1.9e-15) |

**판단**: Linear separability를 제외한 두 지표 모두 H1 방향으로 수렴했다. 다만 이것은 점추정 방향이 일치한다는 것이지, 통계적으로 확증된 것은 아니다(`h1_supported=true`는 방향 플래그일 뿐 유의성 검정이 아님).

---

## 2. Step 5 — Stage B: 정보량을 벌렸을 때의 역전

**질문**: visual 조건을 첫 문장만(31.77토큰)으로 축소해 정보량 격차를 9.5%→34.9%로 벌리면, Stage A에서 관찰된 격차는 유지되는가(H1의 강한 형태), 아니면 좁혀지거나 역전되는가(정보량이 지배적)?

전체 9,356장 중 실제로 처치(문장 축소)가 발생한 표본은 3,854장(40.9%)이었다.

**결과** (`step5_stage_b_results.json`):

| | Stage A (contextual − visual) | Stage B, 전체표본 | Stage B, treated-only ($n=3{,}854$) |
| --- | --- | --- | --- |
| $\Delta_{gap}$ diff | **+0.0068** (H1 방향) | **−0.0161** (역전) | **−0.0285** (더 크게 역전) |
| paired cosine diff | **+0.0039** (H1 방향) | **−0.0045** (역전) | **−0.0082** (더 크게 역전) |
| Linear separability | 1.0 / 1.0 | 0.9995~1.0 (여전히 포화) | 0.9995~1.0 |

**판단**: 정보량 격차를 벌리자 부호가 뒤집혔다. 전체표본보다 처치가 실제로 발생한 부분표본에서 역전 폭이 더 크다는 것은, 이 역전이 무처치 표본에 의한 희석 때문이 아니라 처치 자체의 효과임을 뒷받침한다. **정보량(길이)이 정보 종류보다 gap에 더 지배적인 영향을 미친다**는 쪽으로 결과가 기운 것이나, 이는 이번 두 조건(Stage A/B)만으로 도출한 잠정 결론이다.

---

## 3. Step 6 — Temperature Loss Landscape: 체크마크 패턴

**전제**: Frozen 모델에서 temperature $\tau$는 임베딩 자체를 바꾸지 않으므로, 여기서는 다양한 gap 크기의 배치에서 $\tau$별 contrastive loss 값만 계산하는 loss landscape probing을 수행한다.

**결과** (`step6_temperature_results.json`):

- $\tau=1/100$에서 원래 gap 거리가 4개 조합(Stage A/B × visual/contextual) 전부 정확히 global minimum(diff=0.0000)이었다 — Mind the Gap이 MSCOCO에서 보고한 패턴이 SemArt(회화 도메인)에서도 재현됨을 의미한다. 세 참고 논문 중 어디에도 "정보 종류 × temperature landscape" 조합은 없어, 이 조합 자체가 독자적 기여점이다.
- Repulsive structure(반발 구조, back-to-back local minimum)는 $\tau \in \{1/100, 1/50, 1/30\}$에서 관찰되고 $\tau \geq 1/20$부터 사라진다. `critical_tau_bracket = "1/30~1/20"`으로 4개 조합 모두 동일했다.
- **정정 사항**: $\tau=1/30$에서는 반발 구조가 있지만, 원래 지점이 global minimum은 아니고 local minimum으로만 존재한다. "원래 지점=global minimum"이 정확히 성립하는 것은 $\tau=1/100, 1/50$뿐이다.
- visual/contextual 사이에 이 temperature 특성 자체의 차이는 없었다(둘 다 diff=0.0000). 차이가 없다는 것 자체도 결과로 취급했다.
- **Sanity check**: Stage A와 B의 contextual 곡선은 완전히 동일한 텍스트·이미지·모델을 쓰므로 결과가 일치해야 한다. 실제로 `max_abs_diff=0.0`로 완전히 일치해, 파이프라인 무결성을 확인했다.

---

## 4. Step 6b — Pair Margin: 페어링 검증과 두 번째 역전

**목적**: $\Delta_{gap}$은 centroid 간 거리만 보는 지표라 페어링(어떤 이미지가 어떤 텍스트와 진짜 쌍인지) 정보를 쓰지 않는다. Pair margin($margin_{i2t}[i] = S[i,i] - \text{mean}_{j\neq i}S[i,j]$)으로 "진짜 매칭이 가짜 매칭보다 실제로 더 유사한가"를 별도 검증했다.

**결과** (`step6b_pair_margin_results.json`):

- Stage A, image vs visual: margin 평균 0.0837, Cohen's d = 2.04 (매우 큰 효과 크기) — 진짜 쌍이 가짜 쌍보다 확실히 더 유사함을 재확인.
- Stage B treated-only에서 margin diff(contextual−visual)가 **+0.0010으로 소폭 역전**되었다 — Step 5에서 확인된 $\Delta_{gap}$·paired cosine의 역전과 같은 방향의 역전이 독립적인 세 번째 지표(margin)에서도 재현된 것이다.

**판단**: 서로 다른 세 지표($\Delta_{gap}$, paired cosine, pair margin)가 모두 Stage B에서 같은 방향으로 역전됐다는 것은, Step 5의 결과가 특정 지표의 우연이 아니라 일관된 현상임을 뒷받침한다.

---

## 5. Downstream 태스크 확정 — 라벨 선택과 클래스 필터링

**라벨 컬럼 선택**: `type`(10-class)과 `school`(26-class) 두 후보를 zero-shot classification baseline으로 먼저 검증했다.

| 라벨 | 프롬프트 | micro accuracy | above-majority | 판정 |
| --- | --- | --- | --- | --- |
| `type` (single-prompt) | `"a painting depicting {}."` | 0.5546 | **+0.1553** | informative → 채택 |
| `type` (4-template ensemble) | — | 0.5130 | +0.1137 | informative, but single-prompt보다 약함 |
| `school` (single-prompt) | `"a painting from the {} school."` | 0.2840 | **−0.1481** | **다수결보다 나쁨 → 제외** |

`school`이 제외된 이유는 이 컬럼이 작품의 시각적 내용이 아니라 화파·국적 귀속을 나타내, CLIP의 시각적 표현력으로는 애초에 예측하기 어려운 속성이기 때문으로 추정된다(검증되지 않은 해석).

**클래스 필터링 기준**: 유병률 1% 미만 클래스만 제외한다는 규칙을 정확도를 확인하기 전에 사전 결정했다. `study`(14장, 전체의 0.15%)만 이 기준에 해당해 제외되었다. `genre`는 단독 정확도가 0.51%로 극히 낮았지만 유병률(8.3%)이 기준보다 훨씬 높아 **제외되지 않고 그대로 유지**했다 — 정확도가 아니라 표본 수만으로 판단한다는 원칙을 실제로 지킨 사례다.

**최종 baseline** (`final_prevalence_filtered`, 9-class, $n=9{,}342$): micro=0.5841, macro=0.5683, majority baseline=0.3999, above-majority=+0.1842. 이 값이 이후 Step 7b 스윕의 $\lambda=0$ 지점과 정확히 일치함을 확인했다.

**부가 관찰**: `genre` 클래스의 오분류 상위 3개는 `study`(41.8%), `interior`(18.6%), `landscape`(10.3%)로, 이 세 클래스가 오분류의 71%를 차지했다. 원인은 규명하지 않았다.

---

## 6. Step 7a/7b — Embedding Shift 개입: 체크마크가 실제 성능에서도 재현

### 6.1 Step 7a — 4개 지표 진단

4개 콤보(Stage A/B × visual/contextual) 전부에서 $\lambda$에 따른 4개 지표(delta_gap_norm, linear_sep_accuracy, paired_cosine_mean, margin_i2t_mean)가 Step 6과 동일한 체크마크(비단조) 패턴을 보였다 — $\lambda \approx 0.5$ 부근에서 gap이 가장 닫히고, 그 이후 다시 벌어진다.

**미검증 관찰**: 사후 분석 과정에서, $\Delta_{gap}$이 최소가 되는 $\lambda$(약 0.5)와 R@k 등 실제 성능 지표가 최저가 되는 $\lambda$(약 1.0~1.3)가 정확히 일치하지 않고 뒤쪽으로 밀려 있는 경향이 관찰되었다. Centroid 거리라는 기하학적 지표와 개별 쌍의 상대 순위에 좌우되는 성능 지표가 다른 방식으로 움직인다는 가설을 세울 수 있으나, 검증하지는 않았다.

### 6.2 Step 7b — Downstream 성능 스윕

**Retrieval (I→T, $n=2{,}000$, $\lambda \in [0,2]$)**: 4개 조합 전부 $\lambda=0$(무개입)이 R@1 최고점이었다. 예를 들어 `stage_a_visual`은 $\lambda=0$의 0.142에서 $\lambda \approx 1.1$의 0.0175까지 급락했다.

**Classification (9-class, 단일 gap 벡터)**: 초기 스윕($\lambda \in [0,2]$)에서 peak가 경계에 걸려 자동 확장 규칙에 따라 $\lambda \in [0,4]$로 재실행했다. Micro accuracy peak는 $\lambda=2.2$(0.682), macro accuracy peak는 $\lambda=3.2$(0.590)로 **서로 다른 $\lambda$**에서 나타났다 — "최적 $\lambda$"는 어떤 지표를 보느냐에 따라 달라지는 값이지 단일한 상수가 아니다. Trough(국소 최저)는 $\lambda=0.8$(micro=0.619, macro=0.344)이었고, `above_majority`는 스윕 전 구간에서 한 번도 음수가 되지 않았다.

**판단**: Retrieval은 gap을 줄이자마자 급락하는 반면 classification은 gap을 오히려 원본보다 76% 더 벌린 지점($\lambda=2.2$, distance 1.802)에서 원본을 능가했다. 이는 **Fahim et al.의 "retrieval은 gap과 상관없다"는 결론과 다르다.** Fahim은 fine-tuning 기반 개입을 썼고 본 프로젝트는 shift 기반 개입을 썼다는 차이가 있어, **개입 방식이 다르면 gap-성능 관계 자체가 달라질 수 있다**는 것이 본 실험의 가장 강한 독자적 발견 후보다.

---

## 7. Step 7c — Retrieval 확장: 태스크 간 회복 패턴의 질적 차이

**결과** (`step7c_retrieval_extended_results.json`, $\lambda \in [0,4]$, T→I 추가, R@5/R@10 추가, 후보 풀 $n=2{,}000$ 고정 유지):

- 24개 조합(4콤보 × 2방향 × 3 R@k) 전부에서 $\lambda=0$이 peak — classification과 달리 원본을 능가하는 지점이 없다.
- 체크마크형 회복은 나타나지만 폭이 다르다. $\lambda=4$ 시점의 회복률(trough 대비):

| | I→T | T→I |
| --- | --- | --- |
| 회복률 범위 (4개 조합) | 27~42% | 63~85% |

- **I→T와 T→I는 비대칭이다.** 둘 다 $\lambda \approx 0.8$~$1.2$ 부근에서 바닥을 찍고 $\lambda=4$까지 회복하는 체크마크 모양은 같지만, T→I가 훨씬 강하게 회복한다. 이 비대칭은 4개 조합 모두에서 방향이 같아 우연으로 보이지 않지만, **원인은 검증하지 않았다.**
- R@1/R@5/R@10 사이에서는 $\lambda=0$이 최선이라는 결론이 공통이지만, $k$가 커질수록(R@10) 상대적 회복률이 약간 높아지는 경향(특히 I→T)이 있었다.

**최종 비교**: Classification은 $\lambda=2.2$에서 원본을 넘어서는 완전 회복+초과를 보인 반면, retrieval은 기하학적으로 도달 가능한 최대 거리 근방($\lambda=4$, distance $\approx 1.92/2$)까지 가도 T→I조차 원본의 63~85%까지만 회복하고 넘어서지 못했다. **"체크마크는 있지만 폭이 훨씬 얕고 완주하지 못하는" 패턴**이며, 설계 단계에서 예상했던 "태스크마다 gap-성능 관계가 다르다"는 것을 정량적으로 뒷받침하는 결과다.

---

## 8. 최종 결론 종합

| 발견 | 확인 수준 |
| --- | --- |
| Stage A에서 H1 방향(visual gap < contextual gap) 확인 | 점추정 방향 일치, 통계적 확증 아님 |
| Stage B(정보량 격차 확대)에서 세 지표($\Delta_{gap}$, paired cosine, margin) 모두 역전 | 확인된 사실 |
| $\tau=1/100$에서 원 논문과 동일한 global minimum 패턴 재현 | 확인된 사실 |
| Embedding shift에 대한 반응이 지표·태스크마다 다른 λ에서 체크마크(비단조) 형태로 나타남 | 확인된 사실 |
| Classification은 gap을 더 벌린 지점에서 원본을 능가, retrieval은 끝까지 원본을 못 넘음 | 확인된 사실 |
| I→T보다 T→I가 훨씬 강하게 회복 | 확인된 사실(4개 조합 일관), 원인 미검증 |
| 개입 방식(shift vs fine-tuning)에 따라 gap-성능 관계 자체가 달라짐 | 본 실험의 해석, Fahim과의 직접 비교에 근거 |
