# 실험 설계 및 방법론

> 데이터셋 구축·전처리 상세는 `01_데이터_소개_및_전처리.md` 참고.
> 본 문서는 최종적으로 수행된 파이프라인 전체(Step 1~7c)를 기준으로, 무엇을 왜 그렇게 설계했는지를 정리한다.

---

## 1. 개요

### 1.1 연구 질문

CLIP 계열 이미지-텍스트 인코더에서 관찰되는 modality gap이 (1) 여러 조건에서 어떻게 나타나는지, (2) retrieval·classification 성능과 어떤 관계를 갖는지, (3) gap을 인위적으로 조작했을 때 태스크별 성능이 어떻게 달라지는지를 실증적으로 검증한다.

### 1.2 핵심 가설 (H1)

$$\|\Delta_{gap}^{visual}\| < \|\Delta_{gap}^{contextual}\|$$

동일한 정보량(토큰 수) 하에서도, 시각적으로 관련성 높은 텍스트(visual)가 맥락적 정보(contextual)보다 modality gap이 작다는 가설이다. 이는 *Two Effects, One Trigger*의 $I(X;Y)$ 프레임워크를 frozen 모델에 확장 적용한 것으로, **검증된 사실이 아니라 본 실험의 확장 가설**이다.

### 1.3 참고 문헌과의 관계

| 문헌 | 본 실험과의 관계 |
| --- | --- |
| Radford et al. (2021), CLIP | Contrastive 구조, temperature parameter, zero-shot classification 프로토콜의 근거 |
| Liang et al. (2022), Mind the Gap | $\Delta_{gap}$ 정의, embedding shift, loss landscape probing 방법론을 그대로 채택 — 가장 직접적인 비교 기준 |
| Fahim et al. (2024), It's Not a Modality Gap | Linear separability 지표 채택. 단, 이 논문의 개입 방식(uniformity/alignment fine-tuning)과 본 실험의 개입 방식(embedding shift)은 다르며, 이 차이가 뒤에서 중요한 논점이 된다 |
| *Two Effects, One Trigger* | H1의 이론적 동기($I(X;Y)$ 정보 불균형). PDF 원문은 확보하지 못했고 2차 자료(강의 요약)로만 개념을 확인했다는 한계가 있다 |

---

## 2. 파이프라인 구조

| Step | 내용 | 산출물 |
| --- | --- | --- |
| 1 | Stage A/B 텍스트 구성, treated_mask 산출 | `step1_validation_summary.json` |
| 2 | Frozen CLIP 임베딩 추출 (GPU) | `clip_vitb32_embeddings.npz` |
| 4 | Stage A: $\Delta_{gap}$, linear separability, paired cosine 측정 | `step4_stage_a_results.json` |
| 5 | Stage B: 동일 지표를 길이 비대칭 조건에서 재측정 | `step5_stage_b_results.json` |
| 6 | Temperature loss landscape probing | `step6_temperature_results.json` |
| 6b | Pair margin 분석 (매칭 검증) | `step6b_pair_margin_results.json` |
| — | Downstream 라벨 분포·zero-shot baseline 확정 | `downstream_label_distribution.json`, `downstream_zeroshot_baseline.json` |
| 7a | Embedding shift에 따른 4개 지표(delta_gap/linear_sep/paired_cosine/margin) 반응 진단 | `step7a_shift_diagnostics_results.json` |
| 7b | Embedding shift → downstream(retrieval I→T, classification) 성능 스윕 | `step7b_downstream_sweep_results.json` |
| 7c | Retrieval 확장 (T→I 추가, R@5/R@10 추가, λ 범위 확장) | `step7c_retrieval_extended_results.json` |

---

## 3. 핵심 방법론 원칙

파이프라인 전반에 걸쳐 일관되게 적용된 설계 원칙이다.

**원칙 1 — Downstream task마다 독립적인 gap 벡터를 사용한다.** Mind the Gap Appendix B.1의 관례에 따라, retrieval은 자신의 이미지-캡션 쌍으로 계산한 $\Delta_{gap}$(콤보별 4개)을, classification은 이미지-라벨 쌍으로 별도 계산한 $\Delta_{gap}$(단일)을 각각 사용한다. **같은 $\lambda$ 값이라도 두 태스크에서 실제 gap 축소 폭이 다르므로, $\lambda$를 태스크 간에 직접 비교해서는 안 된다.**

**원칙 2 — 클래스/표본 제외 기준은 성능 지표를 보기 전에 사전에 확정한다.** Downstream classification의 9-class 확정 시, 유병률(prevalence) 1% 미만 클래스만 제외 대상으로 삼는 규칙을 정확도 확인 전에 정했다. 실제로 `genre` 클래스는 단독 정확도가 0.5%로 극히 낮았지만 유병률(8.3%)이 기준을 훨씬 웃돌아 제외되지 않았다 — 정확도가 낮다는 이유로 사후에 기준을 바꾸지 않았다는 뜻이다.

**원칙 3 — Retrieval의 후보 풀 크기는 고정하고, 결과는 상대적 추세로만 해석한다.** Retrieval 스윕은 $n=2{,}000$(seed=42)으로 서브샘플링한 후보 풀을 사용하며, 이는 pair margin 분석의 전체 표본($n=9{,}356$) 풀과 다르다. 따라서 retrieval의 절대적인 R@k 수치는 다른 후보 풀 크기를 쓴 연구와 직접 비교할 수 없고, **오직 $\lambda$에 따른 상대적 변화 추세만이 유효한 비교 대상**이다.

**원칙 4 — Embedding shift는 기하학적 개입이지 표현 개선이 아니다.** $x_i^{shift} = \text{Normalize}(x_i - \lambda \Delta_{gap})$, $y_i^{shift} = \text{Normalize}(y_i + \lambda \Delta_{gap})$ 형태의 shift는 centroid 간 거리라는 지표를 직접 움직일 뿐, 그 아래의 표현 품질을 개선하지 않는다. Classification에서 `above_majority`가 스윕 도중 음수로 바뀌는지 여부는 이 한계가 실제로 드러나는지를 보여주는 직접적인 증상으로 취급한다.

---

## 4. Temperature Loss Landscape Probing — 방법 요약

Frozen CLIP에서는 temperature $\tau$가 임베딩 자체에 영향을 주지 않는다. $\tau$는 contrastive loss의 softmax 스케일링 파라미터로 학습 과정에만 관여하기 때문이다. 따라서 여기서 "temperature를 바꿔가며 gap을 측정한다"는 것은 성립하지 않으며, 실제로 수행하는 것은 **loss landscape probing**이다 — 임베딩은 고정한 채, embedding shift로 인위적으로 만든 다양한 gap 크기의 배치에서 여러 $\tau$의 contrastive loss 값만 계산한다. Gradient step이 없어 CPU로 수행 가능하다.

$$\mathcal{L}_{I\to T} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(x_i \cdot y_i / \tau)}{\sum_{j=1}^{N}\exp(x_i \cdot y_j / \tau)}$$

---

## 5. Downstream 태스크 설계 근거

### 5.1 Retrieval

- **방향**: I→T, T→I 둘 다 측정한다. Mind the Gap의 cone effect 논지 자체가 두 모달리티의 기하학적 비대칭을 다루므로, 한쪽 방향만 보는 것은 절반의 그림이라 판단했다.
- **지표**: R@1, R@5, R@10. R@1은 가장 엄격한 지표라 특유의 아티팩트일 가능성을 배제할 수 없어, 완화된 기준까지 같이 봐서 결론의 강건성을 확인한다.
- **후보 풀**: $n=2{,}000$ 고정 서브샘플(원칙 3 참고).
- **$\lambda$ 스윕 범위**: 기본 $[0,2]$에서 시작해, classification에 적용한 것과 동일한 자동 확장 규칙(peak가 경계에 걸리면 범위를 배증, 최대 2회)을 그대로 적용해 $[0,4]$까지 관찰한다. 좁은 관찰 범위 때문에 "retrieval은 회복하지 않는다"는 결론이 성급했을 가능성을 배제하기 위함이다.

### 5.2 Classification

- **라벨 후보 검토**: `type`(10-class), `school`(26-class) 두 컬럼을 zero-shot baseline으로 먼저 검증했다.
  - `school`: `above_majority_accuracy = -0.148` (다수결보다 오히려 나쁨) → **not_informative 판정, 태스크에서 제외**
  - `type`: `above_majority_accuracy = +0.155` (single-prompt 기준) → informative 판정, 채택
- **프롬프트 비교**: single-prompt(`"a painting depicting {}."`)가 4-template ensemble보다 오히려 above-majority 마진이 컸음(+0.155 vs +0.114) — CLIP 프롬프트 앙상블이 일반적으로 유리하다는 통념과 반대되는 결과이나, 본 실험에서는 이 차이의 원인을 별도로 규명하지 않았다.
- **클래스 필터링**: 유병률 1% 미만인 `study`(14장, 0.15%)만 제외 → 최종 9-class, $n=9{,}342$ (원칙 2 참고).
