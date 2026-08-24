# English Dense + BGE-M3 Sparse Fusion (Final 400/60 Test40)

## Design

- 기존 JSON에 저장된 Dense/Sparse Test40 Top-20 순서를 재사용했습니다. Retrieval model inference는 다시 실행하지 않았습니다.
- 기존 RRF와 같은 `k=60` 및 `(-score, chunk_id)` tie-break를 사용했습니다.
- Equal RRF는 `1/(60+rank_dense) + 1/(60+rank_sparse)`입니다.
- Weighted RRF는 각 항에 사전 지정한 Dense/Sparse weight를 곱했습니다: 2:1, 3:1.
- Test40 과적합을 피하기 위해 이 세 설정 외 weight는 탐색하지 않았습니다. 결과는 탐색적 sensitivity check입니다.
- 각 signal Top-20 → fused Top-20 → 동일 `BAAI/bge-reranker-v2-m3` → Top-5, 동일 evidence-group metric을 사용했습니다.

## Results

| Pipeline | R@20 | Hit@20 | MRR@20 | R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Dense + Sparse Equal RRF | 0.8708 | 0.9250 | 0.5046 | 0.8000 | 0.8750 | 0.6737 | 0.6487 |
| Dense + Sparse Weighted RRF 2:1 | 0.8958 | 0.9500 | 0.4869 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Dense + Sparse Weighted RRF 3:1 | 0.8958 | 0.9500 | 0.4659 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |

## Query-level changes vs Dense

- **Dense + Sparse Equal RRF**: new hit 1 ['ACC_015']; new miss 2 ['ACC_010', 'FRD_015']; first-gold rank improved/worsened/same = 8/16/16; ACC_015=15, ACC_021=miss
- **Dense + Sparse Weighted RRF 2:1**: new hit 0 []; new miss 0 []; first-gold rank improved/worsened/same = 6/16/18; ACC_015=miss, ACC_021=miss
- **Dense + Sparse Weighted RRF 3:1**: new hit 0 []; new miss 0 []; first-gold rank improved/worsened/same = 5/14/21; ACC_015=miss, ACC_021=miss

## Interpretation

- Equal RRF는 Sparse-only `ACC_015`를 candidate rank 15로 유지했고 reranker가 rank 3으로 올렸습니다. 하지만 Dense hit이던 `ACC_010`, `FRD_015`를 candidate에서 잃어 Hit@20이 0.9500→0.9250으로 하락했습니다.
- Equal RRF reranker는 `ACC_015`를 새로 Top-5 hit했지만 `ACC_010`을 Top-5에서 잃어 Hit@5/Recall@5는 Dense와 같았습니다. MRR@5와 nDCG@5는 각각 0.0125, 0.0057 낮았습니다.
- Dense-heavy 2:1/3:1은 Dense의 38 candidate hit를 전부 유지했지만 `ACC_015`를 잃었습니다. Candidate Recall/Hit은 Dense와 같지만 MRR@20은 각각 0.4869, 0.4659로 낮았습니다.
- 2:1/3:1의 fused Top-20 set은 reranker 관점에서 Dense candidate set을 보존했으므로 Top-5 ranking과 모든 reranker metric이 Dense와 정확히 같았습니다. 성능 이점은 없었습니다.
- `ACC_021`은 세 fusion에서 모두 miss로 남았습니다.
- Test40의 작은 차이와 weight sensitivity를 확정적 hyperparameter 선택으로 해석하지 않습니다.

## Decision

사전 지정한 세 RRF 설정 중 `ACC_015`를 살리면서 Dense candidate/ranking 성능을 동시에 유지한 설정은 없었습니다. 따라서 현재 English architecture baseline은 Dense→Reranker로 유지하고, Dense+Sparse RRF의 우선순위는 낮춥니다. 다음 독립 signal은 translated Nori BM25입니다. 이번 단계에서는 실행하지 않았습니다.
