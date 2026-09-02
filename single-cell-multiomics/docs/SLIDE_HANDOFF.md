# Single-Cell Modality Gap 실험 — 슬라이드 제작용 핸드오프 문서

> **이 문서의 용도**: 다른 LLM/에이전트가 이 실험 결과를 보고 발표 슬라이드(SemArt 트랙 H1 검증 슬라이드 형식)를 제작할 수 있도록, 실험 설계·전체 실측 수치·그림 자산의 매핑을 한 파일에 자기완결적으로 정리한 것. 원본 실행 로그는 `docs/HISTORY.md`, 코드-계획 매핑은 `docs/CODE_MAP.md`, 원 계획서는 `docs/PLAN.md`에 있으며 이 문서는 그 세 파일의 내용을 슬라이드 제작 관점으로 재구성한 요약본이다.

---

## 0. 연구 배경과 검증 대상 가설(H1)

**원 가설(Schrödi et al. 2024)**: 이미지-텍스트 대조학습(CLIP류)에서 "모달리티 간 정보 비대칭(information imbalance)이 modality gap과 object bias를 유발한다."

**이 프로젝트의 H1(단일세포 도메인 확장)**: 위 가설이 이미지-텍스트가 아닌 single-cell multi-omics(GEX-ADT, GEX-ATAC) 데이터에서도 성립하는가 — 즉 "두 모달리티 간 정보 비대칭이 클수록 modality gap이 커지고, 그 gap이 downstream 매칭 성능/object-bias 유사 현상에 영향을 준다."

**이 도메인을 쓰는 이유**: 이미지-캡션 데이터는 정답 매칭 자체가 웹 스크래핑 기반이라 noisy하지만, 10x Multiome/CITE-seq은 **동일 세포에서 물리적으로 동시 측정된 진짜 paired ground-truth**를 가지므로, 매칭 품질 노이즈를 배제하고 순수하게 정보 비대칭 효과만 분리해서 볼 수 있다.

**데이터**: GSE194122 (OpenProblems NeurIPS2021 BMMC 대회 배포본)
- cite (GEX+ADT): 90,261 cells, GEX 13,953 genes / ADT 134 proteins, batch 12개(4 site×donor), cell_type 45종
- multiome (GEX+ATAC): 69,249 cells, GEX 13,431 genes / ATAC 116,490 peaks, batch 13개, cell_type 22종
- Train/test: held-out split, test_frac=0.2, batch-stratified

**지표 위계** (`src/metrics/gap_metrics.py`):
- **주지표**: `delta_gap` — unit-normalize된 두 모달리티 embedding centroid 간 L2 거리 (Liang et al. 2022 방식)
- **보조지표**: `alignment`(paired 세포 간 거리 → 코사인 유사도로 환산 가능), `linear_separability`(로지스틱회귀 CV accuracy, 0.5=구분불가/1.0=완전분리), `top5_retrieval_acc`(cosine top-5 내 정답쌍 포함 비율)

---

## 1. 그림(figure) 자산 인벤토리 — 어떤 그림이 어떤 결과에 대응하는가

| 파일 경로 | 상태 | 내용 | 대응하는 결과(§) |
|---|---|---|---|
| `results/figures/phase1_pca_2d_linear_cca_only.png` | **생성완료** | 2×1 패널(각 패널 held-out test 임베딩을 PCA로 2D 축소). 좌: GEX-ADT(cite, 전체 데이터). 우: GEX-ATAC(multiome, **69,249개 중 15,000개 서브샘플** — 데이터셋 전역에 걸쳐 5개 구간으로 나눠 추출, 이 세션 RAM 14GB 제약 때문. 정확한 delta_gap 수치는 §2 표의 전체 데이터 기준 값을 그대로 사용할 것). 파랑=GEX, 주황=ADT/ATAC, X마커=각 모달리티 centroid, 점선=두 centroid를 잇는 선(길이가 delta_gap의 시각적 대응물) | §2 Phase 1 baseline (linear CCA 행) |
| `results/figures/phase1_pca_2d.png` | **미생성** (GPU 필요 — MatchCLOT-arch 재학습 포함 2×2 버전. 이번 세션은 GPU가 없어 스킵) | 생성 시 2×2: 행=pair(cite/multiome), 열=encoder(linear CCA / MatchCLOT-arch) | §2 Phase 1 baseline (MatchCLOT-arch 행까지 포함 시) |
| (없음, 아래 §6 "그래프 스펙"에 x/y만 정의) | 미생성 — 이후 슬라이드 제작 시 각 표의 수치를 그대로 넣어 새로 그리면 됨 | Phase 2~4의 막대/라인 차트 | §3, §4, §5 |

재생성 스크립트: `src/experiments/phase1_pca_plot.py` (`python -m src.experiments.phase1_pca_plot` — GPU 있으면 자동으로 MatchCLOT-arch 패널까지 포함한 2×2를, 없으면 1×2 linear-CCA-only를 생성). multiome은 anndata의 backed 모드가 sparse `layers`를 열자마자 통째로 메모리에 올리는 문제(파일만 열어도 ~6GB)가 있어, 이 스크립트는 multiome에 대해서만 h5py로 필요한 행 구간만 직접 읽는 별도 로더(`_load_multiome_counts_subset`)를 사용함 — 저RAM 환경 재실행 시 참고.

---

## 2. Phase 1 — Baseline Modality Gap 측정

Held-out test set에서 두 인코더(linear CCA = 상관 최대화하는 경량 baseline, MatchCLOT-arch = InfoNCE 대조학습 from-scratch 재현)로 gap 4개 지표 측정.

| 모달리티 쌍 | 인코더 | delta_gap | paired cosine sim* | linear_separability | top5_retrieval_acc |
|---|---|---|---|---|---|
| GEX-ADT (cite) | Linear CCA | 0.0467 | 0.584 | 0.530 | 0.062 |
| GEX-ADT (cite) | MatchCLOT-arch (3-seed 평균) | 0.0838 ± 0.0032 | 0.645 | 0.783 | 0.184 ± 0.002 |
| GEX-ATAC (multiome) | Linear CCA | 0.1559 | 0.619 | 0.679 | 0.123 |
| GEX-ATAC (multiome) | MatchCLOT-arch (3-seed 평균) | 0.0897 ± 0.0020 | 0.693 | 0.845 | 0.249 ± 0.002 |

\* paired cosine sim = 1 − alignment/2 (unit vector 항등식 ‖a-b‖²=2-2cosθ 이용). 원본 alignment: cite-CCA 0.832, cite-MatchCLOT 0.710(3-seed평균), multiome-CCA 0.762, multiome-MatchCLOT 0.613(3-seed평균).

**원본 데이터**: `results/tables/phase1_baseline_linear_cca.csv`, `results/tables/phase1_baseline_matchclot_arch.csv`

**해석 (H1 판정: 반대)**: 두 인코더 모두 GEX-ATAC의 delta_gap이 GEX-ADT와 같거나 큼 — "정보 비대칭이 더 큰 쪽 gap이 클 것"이라는 H1 예측과 반대 방향. 단, 격차 크기는 인코더 의존적(Linear CCA: ATAC이 ADT의 ~3.3배 / MatchCLOT-arch: ~1.07배로 격차 대폭 축소) → 방향은 인코더 선택에 어느 정도 강건, 크기는 인코더(목적함수)에 좌우됨.

**슬라이드 제안**: 이 섹션은 `phase1_pca_2d_linear_cca_only.png`를 그대로 인용 가능. "두 구름이 갈라져 있는 정도(centroid 사이 점선 길이)"가 delta_gap이라는 것을 캡션으로 설명하면 SemArt의 image/text 2-cloud 플롯과 동일한 시각 문법이 된다.

---

## 3. Phase 2 — Batch Effect Confound 분리

배치효과가 gap의 원인인지 분리하기 위해 Harmony(harmonypy) on/off 비교 + variance partitioning(ANOVA-R², 199회 permutation p-value).

| 쌍 | Harmony | delta_gap | r²_batch (p) | r²_batch 감소율 | top5_retrieval_acc | celltype_silhouette |
|---|---|---|---|---|---|---|
| cite | OFF | 0.0467 | 0.1018 (0.005) | — | 0.062 | 0.0898 |
| cite | ON | 0.0477 | 0.0221 (0.005) | −78.3% | 0.028 | 0.0985 |
| multiome | OFF | 0.1559 | 0.0858 (0.005) | — | 0.123 | 0.0539 |
| multiome | ON | 0.1638 | 0.0189 (0.005) | −78.0% | 0.070 | 0.0591 |

(matchedN 조건도 별도 실행됨: cite matchedN n=69,249 결과가 full과 거의 동일 — 세포 수 차이가 gap의 원인이 아님을 재확인. multiome은 애초에 두 데이터셋 중 더 작은 쪽이라 matchedN이 곧 자기 자신의 전체 크기와 같아 실질적 subsampling이 발생하지 않음. 원본: `results/tables/phase1_batch_confound.csv`)

**해석 (H1과 직접 관련 없음, confound 배제 단계)**: Harmony는 배치가 설명하는 분산(r²_batch)을 두 쌍 모두 ~78% 제거하지만, 그 사이 delta_gap은 거의 그대로거나 오히려 소폭 증가함 → **배치효과의 gap에 대한 순기여도 ≈ 0**. Modality gap의 원인은 batch effect가 아니다.

**부수 발견 (슬라이드에 넣을 만한 흥미로운 포인트)**: Harmony 적용 후 top5_retrieval_acc가 절반 이하로 급락(cite 0.062→0.028, multiome 0.123→0.070)했는데 linear_separability는 오히려 소폭 상승. GEX와 ADT/ATAC을 서로 참조 없이 독립적으로 harmonize하면서 세포 단위 미세 대응관계가 훼손된 것으로 추정. **"배치효과 제거(그룹 간 차이 최소화가 목적)"와 "gap/retrieval 측정(개별 세포 단위 정밀도 보존이 전제)"은 서로 다른 목적의 전처리이며 항상 양립하지 않는다**는 트레이드오프.

---

## 4. Phase 3 — 세 가지 개입 실험

### 4-A. HVG gene count sweep (quantity 축) + quality 축

**Quantity 축** (cite 기준, multiome으로 재현 확인): 원본 `results/tables/phase2_expA_dial_swipe.csv`(cite), `results/tables/phase2_expA_multiome_quantity.csv`(multiome 재현)

| n_genes | cite delta_gap (평균±표준편차) | multiome delta_gap | cite top5_retrieval | multiome top5_retrieval |
|---|---|---|---|---|
| 50 | 0.5875 ± 0.0023 | 0.599 | 0.0245 | 0.033 |
| 134 | 0.3313 ± 0.0009 | 0.362 | 0.0576 | 0.069 |
| 500 | 0.1363 ± 0.0024 | 0.134 | 0.1426 | 0.184 |
| 2,000 | 0.0838 ± 0.0032 | 0.090 | 0.1843 | 0.249 |
| 13,953(전체) | 0.0758 ± 0.0066 | 0.063 | 0.1189 | 0.164 |

**Quality 축** (36개 유전자로 개수 고정, 내용만 다르게 비교): 원본 `results/tables/phase2_expA_dial_swipe.csv`

| condition | n_genes | delta_gap (평균±표준편차) |
|---|---|---|
| random_134hvg (참고용 스케일) | 134 | 0.3313 ± 0.0009 |
| adt_matched (ADT와 생물학적으로 대응되는 유전자) | 36 | 0.3634 ± 0.0019 |
| stat_matched_random (발현통계량만 맞춘 무관 유전자) | 36 | 0.7516 ± 0.0102 |
| adt_matched_pair_shuffled (매칭 유전자, 세포쌍만 셔플) | 36 | 0.9831 ± 0.0011 |

**해석 (H1 판정: 혼재)**:
- Quantity 축은 H1과 **정반대**: 유전자 수가 늘수록(정보량↑) delta_gap이 0.588→0.076으로 8배 단조 감소 — GEX-ATAC에서도 거의 동일한 패턴으로 재현됨. "정보 비대칭 클수록 gap 크다"는 소박한 예측과 반대.
- Quality 축은 H1과 **일치**: 같은 36개 유전자 수를 고정하고 내용만 바꾸면, ADT-무관 유전자(gap 0.752)가 ADT-관련 유전자(gap 0.363)보다 gap이 2배 이상 큼 → "정보의 양"이 아니라 "정보의 관련성(질)"이 gap을 좌우한다는 재해석 필요.
- 참고: 세포 대응관계만 셔플하면 gap이 0.983으로 가장 커짐 — 올바른 짝이 올바른 내용보다 더 근본적임을 재확인.
- 추가 관찰: retrieval 성능은 gap과 다른 지점(n=2,000)에서 정점을 찍고 전체 유전자에서는 오히려 하락(0.184→0.119) — gap과 downstream 성능이 항상 같이 움직이지 않음.

**그래프 스펙**: Quantity — x=n_genes(로그축 권장), y=delta_gap 라인(cite/multiome 2개), 보조 패널에 top5_retrieval_acc. Quality — x=condition(4개), y=delta_gap 막대, 에러바=std.

### 4-B. Cross-cell-type mismatching (재학습 없음)

> **지표 정정 (중요)**: 애초 요청된 "R@1/R@5 retrieval accuracy" 형태로는 측정되지 않았음. 실제 구현(`src/experiments/phase2_expB_crosstype.py`)은 MatchCLOT-arch 임베딩에서 세포쌍 간 **cosine 유사도 평균 + permutation null 대비 p-value(2000쌍, null은 무작위 쌍)**를 계통(lineage) 거리별로 측정한 것. 슬라이드에서 "retrieval accuracy"라고 쓰면 부정확하니 "cross-modal cosine similarity by lineage distance"로 표기 권장. R@1/R@5가 꼭 필요하면 재실행 필요(현재 미실행).

원본: `results/tables/phase2_expB_crosstype.csv`

| condition | n_pairs | mean cosine sim | null 대비 p-value |
|---|---|---|---|
| true_pair | 2,000 | 0.647 | 0.002 |
| same_type_diff_object (같은 세포타입, 다른 개체) | 2,000 | 0.412 | 0.002 |
| same_lineage_diff_type (같은 계통, 다른 세포타입) | 2,000 | 0.171 | 0.002 |
| diff_lineage (다른 계통) | 2,000 | −0.003 | **1.000** |
| random_pair (null) | 2,000 | 0.009 | — |

**해석 (H1의 object-bias 아날로그 판정: 지지)**: 계통 거리가 멀어질수록 유사도가 매끄럽게 단조 감쇠(0.647→0.412→0.171→−0.003)하며, diff_lineage는 무작위 쌍과 통계적으로 완전히 구분 안 됨(p=1.000). 인코더가 개별 세포를 암기한 게 아니라 세포타입 수준으로 일반화했다는 뜻으로, "object bias 아날로그" 개념이 예상대로 확인됨.

**그래프 스펙**: x=condition(정렬: true_pair→same_type_diff_object→same_lineage_diff_type→diff_lineage→random_pair), y=mean cosine similarity(에러바=std_sim), null 평균(0.009) 수평 기준선.

### 4-C. Single-lineage subset 재학습 (heterogeneity dose-response)

5개 계통(T_CD4/T_CD8/Myeloid_Mono/B_cell/NK_ILC) 각각 단일 계통으로 좁혀 재학습 vs 동일 N으로 전체 이질성 유지(matchedN) 재학습. 3×3 seed(총 33회 학습)로 재현성 확인까지 완료.

원본: `results/tables/phase2_expC_reproducibility.csv` (최종 확정치, `phase2_expC_lineage.csv`는 단일-seed 초기치)

| lineage | single delta_gap (평균±표준편차) | matchedN delta_gap (평균±표준편차) | single > matchedN 재현율(9 seed쌍) |
|---|---|---|---|
| T_CD4 | 0.0819 ± 0.0056 | 0.0605 ± 0.0090 | 100% |
| T_CD8 | 0.1174 ± 0.0208 | 0.0704 ± 0.0057 | 100% |
| Myeloid_Mono | 0.0853 ± 0.0104 | 0.0629 ± 0.0034 | 100% |
| B_cell | 0.0636 ± 0.0062 | 0.0592 ± 0.0040 | 67% |
| NK_ILC | 0.0771 ± 0.0034 | 0.0668 ± 0.0030 | 100% |

**해석 (H1 판정: 반대, 재현성 검증됨)**: 5개 계통 전부에서 heterogeneity를 낮춘 single-lineage 쪽 delta_gap이 오히려 matchedN(전체 이질성 유지)보다 큼 — "이질성을 줄이면 gap이 준다"는 H1 예측과 정반대. 4/5 계통은 9번의 seed쌍 중 9번 전부(100%) 같은 방향으로 재현되어 seed 우연이 아님. 계획서가 사전에 정한 반증 기준("이 조건에서 gap이 줄지 않으면 인과관계가 성립하지 않는 증거")에 따라 확정 반증으로 처리.

**그래프 스펙**: x=lineage(5개), grouped bar(single vs matchedN) y=delta_gap, 에러바=std(3×3 seed쌍).

---

## 5. Phase 4 — 통합 회귀분석 (Mediation)

실험 A quantity 축 15개 관측치(asymmetry_index = log(n_genes/134)로 정의 가능한 조건만) 기준 Baron & Kenny 순차회귀. 원본: `results/tables/phase3_pooled.csv`

| 경로(Step) | 계수 | p-value | 유의성 |
|---|---|---|---|
| Step 1: asymmetry → gap | −0.0852 | <0.001 | 유의 |
| Step 2: asymmetry → performance (총효과) | +0.0204 | 0.003 | 유의 |
| Step 3: gap → performance (asymmetry 통제) | −0.3475 | <0.001 | 유의 |
| Step 3: asymmetry → performance (직접효과, gap 통제) | −0.0092 | 0.196 | 비유의 (효과 소멸) |

**해석 (H1 판정: 매개구조는 지지 / 부호는 반대)**: gap을 투입하자 asymmetry의 직접효과가 (+0.0204 → −0.0092, 비유의)로 소멸하고 gap 자체는 강하게 유의 — "정보 비대칭 → gap → 성능" 매개 구조 자체는 통계적으로 확인됨. 다만 asymmetry_index가 클수록(유전자 많을수록) gap이 줄어드는 음의 계수라 H1의 소박한 방향과 부호가 반대. gap→성능 부호(음수)는 H1이 예상한 방향과 일치. 표본 15개로 작아 계수 크기는 과신 금지(quality축·expB/expC는 이 척도로 환산 불가해 제외).

**부록 — gap·성능 상관 분해** (follow-up, pooled 44개 관측치: Phase1 baseline + expA + expC. 원본 `results/tables/followup_gap_decomposition_*.csv`): delta_gap을 포함한 5개 지표 모두 top5_retrieval_acc와 강한 음의 상관(Pearson r = −0.72 ~ −0.86, 전부 p<1e-7) — 전체적으로는 "gap 크면 성능 낮다"가 강하게 성립하지만, 실험별로 쪼개면 예외 구간 존재(Harmony 비교, quantity 축 고유전자수 구간에서 방향이 갈림).

---

## 6. Phase 0 — 환경/재현 관련 (슬라이드 뒷단 caveat 슬라이드용)

- GPU A100 40GB / torch 2.8+cu128 환경에서 실행. MatchCLOT(AI4SCR/MatchCLOT) 원 저자 repo의 pretrained weight는 **Box 링크가 만료/404**되어 확보 불가 → 원 논문의 competition SOTA 수치와 이 프로젝트 결과를 **직접 비교하는 것은 불가능**함(방법론적 한계, "미완료"가 아니라 근본적 제약).
- 아키텍처/손실함수(Encoder MLP + symmetric InfoNCE)는 원 repo에서 거의 verbatim으로 vendoring해 구조적 재현은 완료. 단, 학습 루프는 원본의 `catalyst==22.4`+`torch==1.13.1` 의존성이 현재 환경과 충돌해 순수 PyTorch로 새로 작성.
- Epoch 예산은 원 논문 7,000 epoch×9-fold ensemble 대비 150 epoch×1 fold로 대폭 축소(수십 개 실험 조건을 반복해야 하는 이 프로젝트의 트레이드오프). 절대 수치보다 "조건 간 상대 비교/방향성"으로 해석해야 함.

---

## 7. 슬라이드 제작 시 권장 흐름 (요약)

1. **인트로**: H1 정의(§0) + 왜 single-cell이 이미지-텍스트보다 깨끗한 테스트베드인지
2. **Baseline gap** (§2): 2-encoder 비교 표 + `phase1_pca_2d_linear_cca_only.png` 삽입, "방향은 H1과 반대"
3. **Confound 배제** (§3): batch가 원인이 아님을 짧게 (선결 조건 통과 슬라이드)
4. **3개입 실험** (§4-A/B/C): quantity(반대)/quality(지지)/cross-type(지지, 지표명 주의)/single-lineage(반대) — 이 4개를 나란히 놓고 "H1이 부분적으로만, 그리고 조건부로 성립한다"는 메시지로 수렴
5. **통합 회귀** (§5): 매개구조는 있으나 부호가 반대라는 반전 포인트로 마무리
6. **한계/caveat** (§6): pretrained 비교 불가, epoch 축소, 표본 크기(n=15)

**전체를 관통하는 한 문장**: "정보 비대칭이 gap을 유발한다"는 H1은 단일세포 도메인에서 **양이 아니라 질(관련성) 축에서만, 그리고 개체-계통 일반화(object-bias 아날로그) 관점에서만 지지되며, 유전자 개수·이질성 감소 같은 다른 조작 축에서는 재현 가능하게 반증된다.**
