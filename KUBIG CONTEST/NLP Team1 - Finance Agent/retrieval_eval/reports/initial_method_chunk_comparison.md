# Retrieval 평가 (6.3) — BM25 / Dense / Hybrid / Hybrid+Reranker 비교

> 대상 문서: `다국어_금융_AI_Agent_프로젝트_소개.md` 6.3절
> 평가 코드: `retrieval_eval/eval_retrieval.py`
> 실행 환경: Python 3.14 venv (`retrieval_eval/.venv`), CPU 추론

## 1. 평가 목적

프로젝트 문서 5장에서 채택한 Hybrid Retrieval 설계(BM25 + BGE-M3 Dense + RRF 융합 + bge-reranker-v2-m3 재정렬)가
실제로 단계별로 성능을 개선하는지, 그리고 어떤 조합이 최종 파이프라인에 적합한지 확인하기 위해
문서 6.3절에서 정의한 4가지 방법·3가지 지표 기준으로 비교 실험을 진행했다.

## 2. 데이터 및 전제

| 항목 | 내용 |
|---|---|
| 원본 코퍼스 | `documents/documents.jsonl` (공식문서 21개: PDF 8, Web 13) |
| Chunk 버전 | `chunks/chunk_{300_50, 400_60, 500_80}/chunks.jsonl` (Chunk Size/Overlap, 토큰 단위) |
| 평가 문항 | `rag_evaluation_dataset.jsonl` 중 `split == "test"` 60문항 |
| 질문 언어 | 한국어(`question.ko`) — 코퍼스 21개 문서 중 15개가 한국어라 기본값으로 채택 |
| 임베딩 모델 | `BAAI/bge-m3` (문서 5장에서 최종 채택한 모델) |
| Reranker | `BAAI/bge-reranker-v2-m3` (BGE-M3와 동일 팀 개발 cross-encoder) |
| 융합 방식 | RRF (Reciprocal Rank Fusion), 문서 5장에서 CC 대비 채택된 방식 |

### 2.1 Gold Retrieval Label 생성 방법

평가 데이터셋(`rag_evaluation_dataset.jsonl`)에는 문항별 정답 **chunk_id**가 별도로 존재하지 않고,
`evidence` 배열에 원문 인용(`quote`)과 출처(`source_file` 또는 `source_url`)만 기록되어 있다.
따라서 다음 절차로 정답 chunk 집합을 자동 생성했다.

1. `documents.jsonl`의 각 문서에 대해 `(source_type, source_file 또는 source_url)` → `doc_id` 매핑 테이블을 만든다.
   - PDF 문서는 `source_file`(파일명)로, Web 문서는 `source_url`로 매칭한다.
2. 평가 문항의 각 `evidence` 항목에 대해 위 매핑으로 `doc_id`를 찾는다.
3. 해당 `doc_id`에 속한 chunk들 중, `evidence.quote`를 공백 정규화한 문자열이 **그대로 포함되는 chunk**를 gold로 지정한다.
4. 완전 포함 매칭이 실패하면 quote를 문장 단위로 분할(길이 8자 이상)해 **부분 매칭**을 재시도하고, 매칭된 chunk를 모두 gold 집합에 합친다(청크 경계를 넘는 근거 대응).
5. 그래도 매칭되지 않으면 해당 evidence는 "unmatched"로 기록하고 제외한다. 한 문항의 모든 evidence가 unmatched면 그 문항은 정답이 없는 것으로 보고 Retrieval 지표 계산에서 제외한다.

**doc_id 매핑 자체는 21개 문서 전체에서 100% 성공**했으며, unmatched는 전부 3번 단계의 quote 텍스트 포함 매칭 실패(`quote_not_found`)였다.
주요 원인은 PDF 표 형식이 추출 과정에서 재구성되며 원본 개행/구분자가 chunk 텍스트와 달라지거나, quote가 여러 chunk에 걸쳐 있는 경우로 추정된다.

## 3. Retrieval 방법 구현

| 방법 | 구현 |
|---|---|
| **BM25** | `rank_bm25.BM25Okapi`. 한글은 형태소 분석기 없이 음절 bigram + 어절 단위를 함께 토큰으로 사용해 근사, 영숫자는 단어 단위 토큰화 |
| **Dense Retrieval** | `BAAI/bge-m3`(SentenceTransformer)로 chunk/질의를 임베딩(cosine, 정규화 벡터 내적) |
| **Hybrid Retrieval** | BM25 top-50, Dense top-50을 RRF(k=60)로 융합 |
| **Hybrid + Reranker** | Hybrid 결과 top-20을 `bge-reranker-v2-m3` cross-encoder로 재정렬 후 top-5 산출 |

## 4. 평가 지표 (문서 6.3 기준)

- **Recall@5**: `min(len(gold),5)`가 아닌 `len(gold)`로 정규화한 상위 5개 내 정답 chunk 비율 — `|top5 ∩ gold| / |gold|`
- **MRR@5**: 상위 5개 중 처음 정답이 등장한 순위의 역수(없으면 0)
- **nDCG@5**: 이진 관련도 기준 DCG@5 / IDCG@5 (IDCG는 `min(|gold|,5)`개가 모두 상위에 오는 이상적 랭킹 기준)

## 5. 결과

### 5.1 Chunk 300/50 (기본값)

문항 수 60개 중 **26개(43%)**에서 gold chunk 확보.

| Method | Recall@5 | MRR@5 | nDCG@5 | N |
|---|---|---|---|---|
| BM25 | 0.412 | 0.389 | 0.358 | 26 |
| Dense Retrieval (BGE-M3) | 0.695 | 0.562 | 0.566 | 26 |
| Hybrid Retrieval (RRF) | 0.586 | 0.529 | 0.507 | 26 |
| Hybrid + Reranker (bge-reranker-v2-m3) | 0.656 | **0.668** | **0.627** | 26 |

### 5.2 Chunk 400/60

문항 수 60개 중 **25개(42%)**에서 gold chunk 확보 (chunk 824개).

| Method | Recall@5 | MRR@5 | nDCG@5 | N |
|---|---|---|---|---|
| BM25 | 0.453 | 0.397 | 0.371 | 25 |
| Dense Retrieval (BGE-M3) | 0.743 | 0.595 | 0.603 | 25 |
| Hybrid Retrieval (RRF) | 0.673 | 0.527 | 0.528 | 25 |
| Hybrid + Reranker (bge-reranker-v2-m3) | 0.723 | **0.693** | **0.679** | 25 |

### 5.3 Chunk 500/80

문항 수 60개 중 **26개(43%)**에서 gold chunk 확보 (chunk 674개).

| Method | Recall@5 | MRR@5 | nDCG@5 | N |
|---|---|---|---|---|
| BM25 | 0.465 | 0.389 | 0.387 | 26 |
| Dense Retrieval (BGE-M3) | 0.734 | 0.593 | 0.601 | 26 |
| Hybrid Retrieval (RRF) | 0.676 | 0.478 | 0.506 | 26 |
| Hybrid + Reranker (bge-reranker-v2-m3) | **0.737** | 0.649 | 0.637 | 26 |

### 5.4 청크 크기별 종합 비교

주의: 세 청크 버전은 gold 매칭 성공 문항이 조금씩 달라(N=25~26) 완전히 동일한 문항 집합은 아니지만, 전체 test set(60문항) 대비 비율은 42~43%로 유사하다.

**Recall@5**

| Method | 300/50 | 400/60 | 500/80 |
|---|---|---|---|
| BM25 | 0.412 | 0.453 | 0.465 |
| Dense (BGE-M3) | 0.695 | 0.743 | 0.734 |
| Hybrid (RRF) | 0.586 | 0.673 | 0.676 |
| Hybrid + Reranker | 0.656 | 0.723 | **0.737** |

**MRR@5**

| Method | 300/50 | 400/60 | 500/80 |
|---|---|---|---|
| BM25 | 0.389 | 0.397 | 0.389 |
| Dense (BGE-M3) | 0.562 | 0.595 | 0.593 |
| Hybrid (RRF) | 0.529 | 0.527 | 0.478 |
| Hybrid + Reranker | 0.668 | **0.693** | 0.649 |

**nDCG@5**

| Method | 300/50 | 400/60 | 500/80 |
|---|---|---|---|
| BM25 | 0.358 | 0.371 | 0.387 |
| Dense (BGE-M3) | 0.566 | 0.603 | 0.601 |
| Hybrid (RRF) | 0.507 | 0.528 | 0.506 |
| Hybrid + Reranker | 0.627 | **0.679** | 0.637 |

**청크 크기 관련 관찰**
- 세 청크 버전 모두에서 방법 간 순위는 동일하다: **BM25 < Hybrid(RRF) < Hybrid+Reranker ≈ Dense**, 그리고 랭킹 품질(MRR/nDCG)은 **Hybrid+Reranker가 항상 1위**.
- 청크가 커질수록(300→400→500) 대부분의 지표가 소폭 개선되는 경향이 있다. 청크가 클수록 한 청크가 더 많은 문맥(및 evidence 원문 전체)을 포함할 가능성이 높아져 gold 매칭 자체도 유리해지고, 검색기 입장에서도 관련 정보가 한 청크 안에 온전히 들어있을 확률이 올라가기 때문으로 추정된다.
- 다만 **400/60이 500/80보다 MRR@5·nDCG@5가 더 높다** (Hybrid+Reranker 기준 0.693/0.679 vs 0.649/0.637) — 청크가 지나치게 커지면 한 청크에 여러 주제가 섞여 임베딩·재정렬의 변별력이 떨어질 수 있음을 시사한다. 즉 Recall은 500/80이 근소 우위, 랭킹 품질은 400/60이 우위로, **400/60이 이번 평가에서 가장 균형 잡힌 청크 크기**로 보인다.
- BM25는 청크가 커질수록 꾸준히 개선(0.412→0.453→0.465 Recall@5) — 청크가 클수록 질문의 핵심 토큰이 청크 안에 포함될 확률이 단순히 올라가는 효과로 보인다.

## 6. 해석

- **BM25 < Dense (모든 청크 크기에서 일관)**: 한국어 평가 질문 대부분이 원문 표현을 그대로 쓰지 않고 상황을 풀어 쓴 패러프레이즈 형태라, 어휘 매칭 기반 BM25보다 의미 기반 Dense(BGE-M3)가 전 지표·전 청크 크기에서 크게 앞선다. BM25는 형태소 분석기 없이 음절 bigram으로 근사했다는 한계도 있다(§7-2).
- **Hybrid(RRF) < Dense 단독 (MRR@5·nDCG@5 기준, 3개 청크 버전 모두 동일한 방향)**: RRF 융합이 상대적으로 약한 BM25 랭킹을 섞으면서 Dense 단독 대비 랭킹 품질이 떨어진다. Recall@5만 보면 400/60·500/80에서는 Hybrid가 BM25 대비 크게 개선되지만, Dense 단독을 넘어서지는 못한다. candidate_k(현재 50)나 RRF의 k 파라미터(현재 60)를 튜닝하면 개선 여지가 있다.
- **Hybrid + Reranker가 랭킹 품질(MRR@5, nDCG@5)에서 3개 청크 버전 모두 최고 성능**: Reranker가 Hybrid 후보(top-20) 내에서 질문과의 실제 적합도로 재정렬하면서, 정답이 있는 경우 더 상위에 배치하는 데 일관되게 효과적이었다. Recall@5는 300/50·400/60에서는 Dense 단독보다 낮지만 500/80에서는 근소하게 앞선다 — 이는 Dense 단독보다 후보군이 top-20으로 제한된다는 구조적 한계와, 청크 품질(크기)에 따라 그 한계가 상쇄되는 정도가 달라짐을 보여준다.
- **청크 크기 400/60이 종합적으로 가장 균형 잡힌 선택**(§5.4): Recall은 500/80이 근소 우위이지만 MRR·nDCG는 400/60이 세 버전 중 가장 높다. 문서 5장에서 후보로 제시한 300~500토큰 범위 중, 이번 평가 결과는 400토큰/60오버랩 쪽에 무게를 싣는다.

## 7. 한계

1. **Gold label 커버리지가 낮음(약 43%)**: quote 문자열 완전/부분 포함 매칭에 의존하기 때문에, PDF 표 재구성이나 청크 경계로 인해 원문과 정확히 일치하지 않는 evidence는 정답에서 누락된다. N=26이라는 작은 표본에서 나온 순위(특히 Hybrid vs Dense 역전)는 통계적으로 안정적이라 보기 어렵다.
2. **BM25 한글 토큰화가 근사적**: 실제 형태소 분석기(예: Mecab, Kiwi) 없이 음절 bigram으로 대체했기 때문에, 문서 5장에서 언급한 "영어 질문 → 한국어 번역 후 BM25 검색" 같은 실제 파이프라인 설계는 반영하지 못했다. 본 평가는 한국어 질문만 사용해 이 문제를 우회했다.
3. **단일 언어(한국어)만 평가**: 다국어 성능 비교(문서 8장 피드백 2번 항목)는 이번 평가 범위에 포함하지 않았다.
4. **RRF/Reranker 하이퍼파라미터 미세 튜닝 없음**: candidate_k=50/20, RRF k=60은 초기값을 그대로 사용했다.

## 8. 재현 방법

```bash
cd retrieval_eval
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe eval_retrieval.py --chunk-size 300_50 --lang ko --split test
.venv\Scripts\python.exe eval_retrieval.py --chunk-size 400_60 --lang ko --split test
.venv\Scripts\python.exe eval_retrieval.py --chunk-size 500_80 --lang ko --split test
```

결과는 `retrieval_eval/results/results_{chunk_size}_{lang}_{split}.json`, 매칭 실패 evidence 상세는
`retrieval_eval/results/unmatched_evidence_{chunk_size}.json`에 저장된다.
