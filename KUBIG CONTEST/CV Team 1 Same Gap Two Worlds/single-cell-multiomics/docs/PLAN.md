# Modality Gap in Single-Cell Data — 업데이트된 실행 계획서

> 원본 계획서(이미지-텍스트 modality gap 연구를 GEX-ADT / GEX-ATAC single-cell 데이터로 재현+확장)를 코드 구현 전 단계에서 방법론적으로 보강한 버전. 기존 구조(Baseline → Confound 제거 → 정보량 조작 실험 3종 → 통합분석)는 유지하고, 각 단계에 확증 가능성(falsifiability)과 통계적 엄밀성을 높이는 control들을 추가했다.
>
> 이 파일은 구현의 기준이 되는 계획 원문이다. 실험 도중 계획을 바꿔야 하면 이 파일이 아니라 `docs/HISTORY.md`에 "왜 바꿨는지"를 기록하고, 계획 자체를 고치는 경우에만 이 파일을 갱신한다.

---

## 0. 연구 질문과 기여점 (명확화)

**핵심 질문:** Schrödi et al. (2024, "Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models")이 제안한 "모달리티 간 정보 비대칭(information imbalance)이 modality gap과 object bias를 유발한다"는 가설이, 이미지-텍스트가 아닌 single-cell multi-omics(GEX-ADT, GEX-ATAC) 도메인에서도 성립하는가?

**이 도메인이 좋은 테스트베드인 이유 (추가 논지):**
이미지-캡션 데이터셋은 "진짜 정답 매칭"조차 웹 스크래핑 기반이라 noisy하다. 반면 10x Multiome/CITE-seq은 **동일 세포에서 물리적으로 동시 측정**된 진짜 paired ground-truth를 갖고 있다. 즉 "매칭 품질" 자체의 노이즈를 배제하고 순수하게 정보 비대칭 효과만 분리해서 볼 수 있는, 오히려 이미지-텍스트보다 깨끗한 실험 조건이라는 점을 논문 기여점으로 명시한다.

**Object bias의 대응 개념 (명시적 정의):**
이미지-텍스트에서 object bias = 이미지 인코더가 배경/디테일을 버리고 캡션이 언급하는 주요 객체 위주로 정보를 압축하는 편향. Single-cell에서의 대응 가설 = GEX 인코더가 세포 전체의 이질성(heterogeneity) 정보를 버리고, ADT 패널이 구분하도록 설계된 주요 세포 계통(lineage) 정보 위주로 압축하는 편향. 이 대응관계를 실험 C에서 직접 검증한다.

---

## 1. Baseline Modality Gap 측정 (보강)

| 원본 계획 | 보강 내용 |
|---|---|
| Pretrained MatchCLOT 인코더 1종으로 임베딩 추출 | **인코더 3종 비교**: (a) pretrained MatchCLOT, (b) 각 데이터셋에 from-scratch 재학습한 동일 구조, (c) linear CCA/PCA + OT (단순 baseline). Gap 순위가 인코더 선택에 강건한지 확인 — 강건하지 않으면 "생물학적 정보 격차"가 아니라 "특정 모델의 inductive bias"라는 뜻이므로 반드시 먼저 확인 |
| 임베딩은 train 전체에서 추출 | **Train/test split 명시, held-out cell에서만 gap 측정** (data leakage 방지) |
| Centroid distance (raw) | **Unit-normalize 후 벡터 차이(Δgap, Liang et al. 방식)로 통일** — norm/scale 의존성 제거, GEX-ADT vs GEX-ATAC 직접 비교 가능하게 |
| (없음) | **모달리티별 signal quality 측정 추가**: 각 모달리티의 reconstruction error 또는 SNR 프록시를 별도로 계산. ATAC는 ADT보다 sparse/noisy하므로, 이후 "정보량 차이"와 "기술적 노이즈 차이"를 구분하는 공변량으로 사용 |
| 4개 지표 병렬 나열 | **지표 위계 확정**: Δgap(centroid)을 주지표로, Alignment-Uniformity / Linear separability / Top-5 retrieval을 보조지표로 지정. 이후 모든 실험에서 이 위계를 일관되게 사용 (multiple comparison 문제 완화) |

**산출물:** 인코더 3종 × 데이터셋 2종 × 지표 4개 표, signal quality 공변량 값

---

## 2. Batch Effect Confound 분리 (보강)

| 원본 계획 | 보강 내용 |
|---|---|
| Harmony on/off 비교로 배치효과 기여도 정성적 확인 | **Variance partitioning 추가**: 배치 라벨 vs 모달리티 라벨로 임베딩 공간의 분산을 PERMANOVA 또는 R² 분해로 정량화. "배치가 gap의 몇 %를 차지하는가"를 숫자로 제시 |
| Harmony 배치교정 적용 후 바로 사용 | **과교정(over-correction) sanity check 추가**: 배치교정 후에도 cell type 구조(silhouette score 등)가 유지되는지 확인. 생물학적 신호까지 지워버렸다면 이후 실험 결과가 왜곡됨 |
| 두 데이터셋 세포 수 차이(~7만 vs ~9만) 언급만 됨 | **Matched subsampling 조건 추가**: 동일 N으로 맞춘 버전도 병행 측정하여 세포 수 차이가 confound로 작용하지 않는지 확인 |

**산출물:** 배치 기여도(%) 수치, 과교정 체크 결과, matched-N 조건 결과

---

## 3. 정보 비대칭 조작 실험

### 3-1) 실험 A — 정보량 dial swipe (quantity / quality 축)

**Quantity 축**
- HVG 개수: 50 → 134 → 500 → 2,000 → 전체(~13,000)
- **추가:** 조건당 **최소 3 seed 반복**, gap 평균±표준편차 리포트 (특히 50/500처럼 적은 유전자 수는 HVG 선택 자체가 랜덤성에 민감)
- **추가:** 인코더 capacity를 입력 차원과 독립적으로 통제 (hidden dim 고정 또는 입력 차원 비례 스케일 버전 병행) — "정보량 증가 효과"와 "under/over-parameterization 효과"를 분리

**Quality 축**
- 무작위 HVG 134개 vs ADT-매칭 134개 비교
- **추가 control:** 발현량/분산/dropout rate 분포를 ADT-매칭 셋과 맞춘 "통계량-매칭 랜덤 셋"을 추가. 이게 없으면 "내용이 매칭되었는가"와 "유전자 집합의 통계적 성질이 다른가"가 섞임
- **추가 control:** ADT-매칭 134개 유전자를 그대로 쓰되 GEX-ADT 세포 쌍 대응관계만 셔플한 조건 추가 — "관련 유전자 집합을 갖고 있음"과 "실제 대응이 맞음"의 효과를 분리

```
quantity 축: ①50 → ②134(무작위) → ③500 → ④2,000 → ⑤전체
quality 축:  ②134(무작위) vs ⑥134(ADT-매칭) vs ⑦134(통계-매칭 랜덤) vs ⑧134(ADT-매칭, 쌍 셔플)
```

### 3-2) 실험 B — Cross-cell-type 미스매칭 (보강)

원본 5조건 표 유지, 아래 통계 절차 추가:

- **Permutation null 확립**: "무작위 쌍" 조건을 단일 샘플이 아니라 라벨을 수백~수천 회 permutation한 null distribution으로 구성. 나머지 조건들이 이 null 대비 유의하게 다른지 검정
- **최소 표본 크기 기준 사전 설정**: rare cell type(naive B, pDC 등)이 포함된 조건은 세포 수 하한선을 정해두고, 미달 시 해당 조건 제외 또는 별도 표기
- **MatchCLOT의 실제 matching probability/OT score도 병행 보고** (embedding cosine similarity뿐 아니라 논문이 실제 쓰는 downstream 지표로도 검증)

### 3-3) 실험 C — 단일 계통 서브셋 (보강)

| 원본 계획 | 보강 내용 |
|---|---|
| 전체 vs 단일 계통 이분법 | **3개 이상 lineage(T cell / B cell / Monocyte 등)로 dose-response 곡선 확인** — heterogeneity 감소에 따라 gap이 단조적으로 줄어드는지 확인 (이분법보다 인과 주장이 훨씬 강해짐) |
| (없음) | **N-matched control 추가**: 단일 계통은 세포 수도 함께 줄어들므로, 전체 조건도 동일 N으로 subsampling한 버전을 대조군으로 비교 — "heterogeneity 감소 효과"와 "샘플 수 감소로 인한 노이즈"를 분리 |

---

## 4. 통합 분석 (보강)

| 원본 계획 | 보강 내용 |
|---|---|
| 정보 비대칭 지표 / gap 지표 / downstream 성능을 산점도·회귀로 결합 | **Mediation analysis(또는 순차 회귀)로 전환**: information asymmetry → gap → downstream performance의 인과 사슬을 gap이 asymmetry의 효과를 얼마나 매개하는지로 정식화. 단순 상관/회귀보다 이론(가설의 인과 구조)과 더 잘 맞음 |
| (없음) | **공변량 통제**: 세포 수, 배치 구성, heterogeneity index를 회귀에 함께 포함하여 information asymmetry의 독립적 기여도를 분리 |
| 4개 지표 × 다수 조건을 개별 비교 | **Multiple comparison 보정** 또는 사전에 지표 위계(1번 항목에서 정한 주지표/보조지표)를 그대로 사용해 검정 수를 통제 |

---

## 5. 실행 순서 (Phase 및 마일스톤)

**Phase 0 — 방법론 확정 (지금, 코드 이전)**
1. Gap 정의를 Δgap(unit-normalized) 방식으로 확정, 지표 위계(주지표/보조지표) 확정
2. Baseline 인코더 3종(pretrained / from-scratch / linear CCA+OT) 확정
3. 각 실험(A/B/C)의 성공·반증 기준을 문장으로 사전 명시 (실험 C 원본의 반증 기준 방식을 A/B에도 동일 적용)

**Phase 1 — Baseline + Confound 정리**
4. Train/test split 정의 → held-out gap 측정 (인코더 3종)
5. Batch variance partitioning + Harmony on/off 비교 + 과교정 sanity check + matched-N 조건

**Phase 2 — 정보량 조작 실험**
6. 실험 A: quantity 축(seed 반복) → quality 축(통계-매칭/쌍-셔플 control 포함)
7. 실험 B: 5조건 + permutation null + 표본크기 기준 적용
8. 실험 C: 3개 이상 lineage dose-response + N-matched control

**Phase 3 — 통합**
9. Mediation/회귀 모델로 information asymmetry → gap → performance 인과 사슬 검증 (공변량 통제 포함)
10. 전체 결과 종합, 이미지-텍스트 도메인 결과와 비교 논의

**Phase 3까지의 의존관계:** Phase 1의 confound 통제가 확실해야 Phase 2 결과의 해석("정보량 조작이 gap을 바꿨다")이 유효하고, Phase 2의 각 실험이 서로 다른 각도(quantity/quality, cross-type, heterogeneity)에서 같은 결론을 가리켜야 Phase 3의 인과 주장이 방어 가능해진다.

---

## 6. 전체를 관통하는 방법론 체크리스트

- [ ] 모든 조건에서 seed 반복(≥3) 및 평균±표준편차 리포트
- [ ] 재학습이 필요한 모든 조건에서 held-out test set 분리
- [ ] 지표 위계 고정 후 사전등록(preregistration) 형태로 문서화 — 실험 순서/가설/통계검정 방법을 미리 못박아 사후 p-hacking 방지
- [ ] 계산자원 확인: 조건 수(HVG 5단계 + quality 3종 + cross-type 5종 + lineage 3+종 등)를 고려하면 MatchCLOT 전체 재학습이 몇 회 필요한지 사전에 견적 — 필요시 linear CCA+OT 같은 경량 baseline으로 먼저 트렌드를 스크리닝하고, 딥러닝 인코더는 유의미한 조건만 골라 재학습하는 것도 고려
