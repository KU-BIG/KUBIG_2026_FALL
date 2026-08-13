# 계획서 ↔ 코드 매핑

`docs/PLAN.md`의 각 항목이 실제로 어느 디렉토리/파일에, 어떻게 구현되어 있는지 추적하는 표. 구현이 끝난 항목부터 채워나간다. "상태" 컬럼: 미구현 / 구현중 / 구현완료(미실행) / 실행완료.

---

## Phase 0 — 방법론 확정

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| Gap 정의(Δgap unit-normalized), 지표 위계 | `src/metrics/gap_metrics.py` | 미구현 | - |
| 지표: Alignment-Uniformity | `src/metrics/gap_metrics.py` | 미구현 | - |
| 지표: Linear separability | `src/metrics/gap_metrics.py` | 미구현 | - |
| 지표: Top-5 retrieval | `src/metrics/gap_metrics.py` | 미구현 | - |
| Baseline 인코더 3종 정의 | `src/encoders/` | 미구현 | - |
| MatchCLOT 코드/가중치 확보 | `external/MatchCLOT/`, `src/encoders/matchclot_wrapper.py` | 미구현 | - |

## Phase 1 — Baseline + Confound

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| GSE194122 데이터 확보 | `src/data/download_bmmc.py` | 미구현 | - |
| Train/test split | `src/data/split.py` | 미구현 | - |
| Baseline gap 측정 (인코더 3종 × 데이터셋 2종) | `src/experiments/phase1_baseline.py` | 미구현 | - |
| Signal quality(SNR) 공변량 | `src/metrics/signal_quality.py` | 미구현 | - |
| Batch variance partitioning | `src/experiments/phase1_batch_confound.py` | 미구현 | - |
| Harmony on/off + 과교정 sanity check | `src/experiments/phase1_batch_confound.py` | 미구현 | - |
| Matched-N 조건 | `src/experiments/phase1_batch_confound.py` | 미구현 | - |

## Phase 2 — 정보 비대칭 조작 실험

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| 실험 A: quantity 축 | `src/experiments/phase2_expA_dial_swipe.py` | 미구현 | - |
| 실험 A: quality 축 (통계매칭/쌍셔플 control 포함) | `src/experiments/phase2_expA_dial_swipe.py` | 미구현 | - |
| 실험 B: cross-cell-type 5조건 + permutation null | `src/experiments/phase2_expB_crosstype.py` | 미구현 | - |
| 실험 C: 단일 lineage dose-response + N-matched | `src/experiments/phase2_expC_lineage.py` | 미구현 | - |

## Phase 3 — 통합분석

| 계획서 항목 | 파일 | 상태 | 구현 방식 요약 |
|---|---|---|---|
| Mediation analysis (asymmetry → gap → performance) | `src/experiments/phase3_integration.py` | 미구현 | - |

---

## 참고 자료 / 외부 의존성

| 항목 | 위치 | 비고 |
|---|---|---|
| MatchCLOT 원본 | `external/MatchCLOT` (git 미포함, clone 스크립트로 재현) | AI4SCR/MatchCLOT, NeurIPS2021 competition 1위 |
| GSE194122 원본 데이터 | `data/raw/` (git 미포함) | OpenProblems NeurIPS2021 Multimodal Single-Cell Integration competition |
