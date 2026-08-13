# 계획서 ↔ 코드 매핑

`docs/PLAN.md`의 각 항목이 실제로 어느 디렉토리/파일에, 어떻게 구현되어 있는지 추적하는 표. 구현이 끝난 항목부터 채워나간다. "상태" 컬럼: 미구현 / 구현중 / 구현완료(미실행) / 실행완료.

---

## Phase 0 — 방법론 확정

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| Gap 정의(Δgap unit-normalized), 지표 위계 | `src/metrics/gap_metrics.py` (`delta_gap`) | 구현완료+테스트 통과 | Unit-normalize 후 centroid L2 거리. 주지표로 확정. `tests/test_gap_metrics.py`에서 shift 단조성 검증 |
| 지표: Alignment-Uniformity | `src/metrics/gap_metrics.py` (`alignment`, `uniformity`) | 구현완료+테스트 통과 | Wang & Isola(2020) 분해 그대로 구현. alignment는 paired 필요 |
| 지표: Linear separability | `src/metrics/gap_metrics.py` (`linear_separability`) | 구현완료+테스트 통과 | 5-fold CV logistic regression, chance=0.5 |
| 지표: Top-5 retrieval | `src/metrics/gap_metrics.py` (`topk_retrieval_accuracy`) | 구현완료+테스트 통과 | cosine sim 기반, paired 필요 |
| 4개 지표 통합 진입점 | `src/metrics/gap_metrics.py` (`gap_report`) | 구현완료+테스트 통과 | `paired=True/False`로 alignment/retrieval 포함 여부 제어 |
| Baseline 인코더 정의 | `src/encoders/linear_baseline.py` (c), `src/encoders/matchclot_arch.py` (b) | 구현완료+테스트 통과 | (a) pretrained MatchCLOT은 가중치 미확보로 **제외** (docs/HISTORY.md 2026-08-13 판단 1). (b) `matchclot_arch.py`: MatchCLOT의 `Encoder`/`Modality_CLIP`/`symmetric_npair_loss`를 vendored(출처 명시, BSD-3-Clause)하고 catalyst 없는 순수 PyTorch 학습 루프를 새로 작성 (판단 2). Epoch을 원 논문의 7000→기본 300으로 축소(다수 실험 조건 반복을 위한 의도적 트레이드오프, 모듈 docstring에 근거 명시). (c) `linear_baseline.py`: sklearn CCA로 공유 임베딩 공간 생성. `tests/test_matchclot_arch.py` 3개 통과(shape, 임의 입력차원 지원, 시드 재현성) |
| MatchCLOT 코드/가중치 확보 | `external/MatchCLOT/` (git 미포함, clone 완료) | 부분완료 | 아키텍처(`models.py`)·전처리(`preprocess.py`)는 재사용(vendored) 확정. Pretrained 가중치(IBM Box 링크)는 접근 불가(404) — docs/HISTORY.md 참고 |

## Phase 1 — Baseline + Confound

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| GSE194122 데이터 확보 | `data/raw/*.h5ad` (GEO 직접 다운로드, git 미포함) | 실행완료 | cite(90,261 cells, GEX 13,953 + ADT 134) / multiome(69,249 cells, GEX 13,431 + ATAC 116,490) 다운로드+압축해제+무결성 확인 완료 |
| 데이터 로딩 / 모달리티 분리 | `src/data/loading.py` (`load_bmmc`, `split_modalities`) | 구현완료 | `var['feature_types']`로 GEX/ADT/ATAC 분리 |
| Train/test split | `src/data/loading.py` (`held_out_split`) | 구현완료 | batch-stratified `train_test_split`, 기본 test_frac=0.2 |
| GEX/ADT/ATAC 전처리 | `src/data/preprocessing.py` (`select_hvgs`, `normalize_gex`, `clr_normalize_adt`, `LSITransformer`) | 구현완료 | GEX: seurat_v3 HVG(개수 조절 가능, 실험 A의 quantity 축과 동일 인터페이스)+normalize_total+log1p. ADT: CLR(margin=2, Seurat 방식). ATAC: TF-IDF+LSI(MatchCLOT 방식 vendored) |
| Baseline gap 측정: encoder (c) linear CCA | `src/experiments/phase1_baseline.py` | **실행완료** | 결과: `results/tables/phase1_baseline_linear_cca.csv`. cite Δgap=0.047, multiome Δgap=0.156 — **계획서 가설과 반대 방향(잠정치)**. CCA는 상관 최대화가 목적함수라 "gap"을 그 자체로 최소화하므로, encoder (b)(contrastive) 결과 없이는 해석 보류 — docs/HISTORY.md 2026-08-13(계속 3) 참고 |
| Baseline gap 측정: encoder (b) MatchCLOT-arch(from-scratch) | `src/experiments/phase1_baseline_matchclot.py` | **실행완료** | 결과: `results/tables/phase1_baseline_matchclot_arch.csv`. cite Δgap=0.084±0.003, multiome Δgap=0.090±0.002 — CCA와 같은 방향(ATAC≥ADT)이지만 격차는 훨씬 작음(3.3배→1.07배). gap과 top5-retrieval이 같은 방향으로 안 움직임(ATAC이 gap도 크고 retrieval도 높음). 세부 해석은 docs/HISTORY.md 2026-08-13(계속 4) |
| Signal quality(SNR) 공변량 | `src/metrics/signal_quality.py` | 미구현 | - |
| Batch variance partitioning | `src/experiments/phase1_batch_confound.py` (`group_r2`+`permutation_test_r2` 사용) | **실행완료** | 결과: `results/tables/phase1_batch_confound.csv`. r2_batch가 두 pair 모두 harmony로 ~78% 감소(p=0.005 유의) — 배치효과가 gap의 주원인이 아님을 확인. docs/HISTORY.md 2026-08-13(계속 5) |
| Harmony on/off + 과교정 sanity check | `src/experiments/phase1_batch_confound.py` (`_harmony_correct`, silhouette) | **실행완료** | delta_gap은 harmony on/off로 거의 안 변함(과교정 없음, silhouette 유지/상승). **예상 밖 발견**: harmony 적용 후 top5_retrieval_acc가 절반 이하로 하락 — batch correction과 gap측정용 전처리가 서로 다른 트레이드오프를 가짐 |
| Matched-N 조건 | `src/experiments/phase1_batch_confound.py::main` | **실행완료** | cite를 69,249로 subsample해도 delta_gap 거의 불변 → 세포 수 차이가 gap 차이의 원인이 아님 확인 |

## Phase 2 — 정보 비대칭 조작 실험

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| 실험 A: quantity 축 | `src/experiments/phase2_expA_dial_swipe.py::run_quantity_axis` | **실행완료** | 결과: `results/tables/phase2_expA_dial_swipe.csv`. **계획서 가설과 반대**: 유전자 50→13,953개로 늘수록 delta_gap이 0.588→0.076로 단조 감소(약 8배). retrieval 성능은 2,000개에서 정점(0.184) 후 하락 — gap과 downstream 성능의 최적점이 다름. 세부 해석: docs/HISTORY.md 2026-08-13(계속 6) |
| 실험 A: quality 축 (통계매칭/쌍셔플 control 포함) | `src/experiments/phase2_expA_dial_swipe.py::run_quality_axis` | **실행완료** | ADT-matched 유전자는 134개가 아니라 실제로는 36개만 확보(대부분 isotype control/gene_id 없음). 36개 기준 공정비교: adt_matched gap=0.363 vs stat_matched_random gap=0.752 — **내용(quality)이 gap에 실제로 영향을 준다는 걸 계획서 가설 방향대로 확인**. pair-shuffled(0.983)가 가장 커서 "올바른 대응관계 > 올바른 내용"도 확인 |
| 실험 B: cross-cell-type 5조건 + permutation null | `src/experiments/phase2_expB_crosstype.py` | **실행완료** | 결과: `results/tables/phase2_expB_crosstype.csv`. **계획서 가설과 방향 일치, 매우 깔끔함**: true_pair(0.647) > same_type_diff_object(0.412) > same_lineage_diff_type(0.171) > diff_lineage(-0.003≈null, p=1.0). docs/HISTORY.md 2026-08-13(계속 7) |
| 실험 C: 단일 lineage dose-response + N-matched | `src/experiments/phase2_expC_lineage.py` | **실행완료** | 결과: `results/tables/phase2_expC_lineage.csv`. **계획서 가설과 반대 — 계획서 자체가 정한 반증 기준에 해당**: 5개 계통 전부에서 single-lineage(이질성↓) 쪽이 matchedN(이질성 유지) 쪽보다 gap이 더 크다. "heterogeneity 감소→gap 감소" 인과관계가 이 데이터/구현에서는 성립하지 않음. docs/HISTORY.md 2026-08-13(계속 7) 참고 |

## Phase 3 — 통합분석

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| Mediation analysis (asymmetry → gap → performance) | `src/experiments/phase3_integration.py` | **실행완료** | 결과: `results/tables/phase3_pooled.csv` + `logs/phase3_integration.log`. exp A quantity 축 15개 관측치 기준 mediation 신호 확인(gap 유의, asymmetry 직접효과 소멸) — 단 방향은 "정보 비대칭↑→gap↓→성능↑"으로 계획서 가설과 반대(exp A 결과와 일관). 표본 작음(n=15) 주의. docs/HISTORY.md 2026-08-13(계속 7) |

---

## 참고 자료 / 외부 의존성

| 항목 | 위치 | 비고 |
|---|---|---|
| MatchCLOT 원본 | `external/MatchCLOT` (git 미포함, clone 스크립트로 재현) | AI4SCR/MatchCLOT, NeurIPS2021 competition 1위 |
| GSE194122 원본 데이터 | `data/raw/` (git 미포함) | OpenProblems NeurIPS2021 Multimodal Single-Cell Integration competition |
