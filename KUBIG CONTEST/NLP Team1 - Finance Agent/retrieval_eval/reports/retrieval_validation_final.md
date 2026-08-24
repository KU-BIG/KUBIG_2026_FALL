# Retrieval 데이터 파이프라인 검증 및 보정 — 최종 정리 (400/60)

이 문서는 `kangminhyeok02/KUBIG_FINANCE`(원본) → `reference baseline repository`(재추출) →
이 repo(`KUBIG_FINANCE_final_test`, 추가 검증·보정)로 이어진 작업 전체를 한 문서로
요약합니다. 아래 모든 수치는 이 repo에 실제로 존재하는 artifact 파일을 직접 읽어서
확인한 값입니다(Appendix에 근거 파일 목록).

---

## 1. Background

기존 `kangminhyeok02/KUBIG_FINANCE`의 400/60 retrieval 평가에서 **gold chunk
coverage가 60문항 중 약 25-26개(42~43%)에 그치는 문제**가 있었습니다. 원인 분석
결과 unmatched evidence의 대부분이 **PDF 소스(특히 PDF001)에 집중**되어 있었고,
Web 소스 evidence는 거의 전부 정상 매칭되었습니다. 즉 문제는 retrieval 알고리즘이
아니라 (1) PDF corpus 추출 품질, (2) evidence quote와 chunk를 잇는 gold mapping
로직 양쪽에 있었습니다.

## 2. Original Data Issues

### 2.1 PDF001 Extraction Issue

`kangminhyeok02` 버전의 `documents.jsonl`을 원본 PDF와 직접 대조한 결과, PDF001의
표 3곳에서 **셀 값 자체가 소실**되어 있었습니다(원문 PDF 열람 및 pdfplumber
`extract_tables()`/`find_tables()` 직접 재현으로 확인):

- p.56 주택청약종합저축 개요표 — "Contents/내용" 컬럼 값 전체 소실(월 2만~50만 원,
  금리 조건 등)
- p.62 체크카드·신용카드 비교표 — "신용카드" 값 컬럼(신용한도금액/후불결제일/
  할부가능여부) 소실
- p.162~168 은행 외국인 전용 데스크 목록(7페이지) — "은행명" 컬럼이 rowspan 병합
  셀이었는데 모든 행에서 공백 처리

이는 evidence quote의 문제가 아니라 **corpus 자체의 콘텐츠 손실**이었습니다.

### 2.2 Original Gold Mapping Limitation

`kangminhyeok02`의 `eval_retrieval.py`는 evidence의 `quote`를 정규화(개행→공백,
공백압축)한 뒤 chunk 텍스트에 **완전 substring으로 포함되는지**만 검사했고, 실패
시 문장 단위 부분 매칭만 재시도했습니다. 이 방식은 아래 경우에 구조적으로 취약
했습니다:

- 표의 여러 cell/row를 사람이 하나의 자연어 문장으로 결합해 작성한 quote
- 원문을 그대로 옮기지 않고 일부 절을 생략·압축한 paraphrase quote
- quote가 chunk 경계를 넘어 두 chunk에 걸쳐 있는 경우
- `page` 필드가 gold mapping 로직에서 전혀 사용되지 않음

## 3. Changes in the 606-Chunk Reference Baseline

### 3.1 PDF 재추출

Reference baseline repository의 `retrieval_eval/prepare_retrieval_data.py`는 PDF 8개를
`pdfplumber.page.extract_text()`로 재추출했습니다(표 전용 추출 API는 사용하지
않음). 2.1의 3개 표에 대해 새 `documents.jsonl`을 직접 열어 재확인한 결과, 표
셀 값이 텍스트로 정상 포함되어 있음을 확인했습니다(`extract_tables()`를 쓰지
않아 "빈 셀" 문제 자체가 발생하지 않는 방식). **PASS.**

### 3.2 BGE-M3 Tokenization 및 재청킹 (824 → 606)

Reference baseline repository는 `BAAI/bge-m3` tokenizer 기반 슬라이딩 윈도우로 300/50·400/60·
500/80 3개 chunk 버전을 재생성했습니다. 400/60 기준 chunk 수는 824개→606개로
감소했습니다.

**중요**: 기존 824개 chunk를 생성한 스크립트는 `kangminhyeok02` repo 어디에도
남아있지 않아 **당시 사용된 tokenizer 정체는 확인 불가능합니다**(추정을 사실처럼
서술하지 않습니다). 확인 가능한 사실만 정리하면:

- 문서 콘텐츠가 완전히 동일한 Web 문서 13개 기준으로도 chunk 수가 감소(구 버전
  대비 chunk당 평균 길이가 늘어남) — 즉 재청킹 자체가 chunk 수 감소에 실질적으로
  기여했습니다.
- Coverage 무결성 검사 결과: 21개 문서 전체에서 **uncovered 본문 = 0**(비어있는
  구간은 `[Page N]` 마커 텍스트뿐), **tail loss = 0**, 같은 페이지 내 연속 chunk
  간 **overlap 정상 존재**를 직접 검사로 확인했습니다.

## 4. Additional Final Corrections (이 repo에서 추가로 수행)

### 4.1 PDF002 Page Scope 복원

Reference baseline repository는 PDF002("Guide to Leading a Safe Life in Seoul", 원본 43페이지)를
**43페이지 전체** 재추출하고 있었습니다. 그러나 원래 corpus 설계는 금융사기(피싱)
관련 페이지만 발췌하는 것이었고, 실제로 `kangminhyeok02`의 `documents.jsonl`에는
**17, 18, 19페이지만** 포함되어 있었습니다(`[Page N]` 마커 직접 grep으로 확인).

이 repo에서는 `prepare_retrieval_data.py`에 PDF002 전용 페이지 필터
(`{17, 18, 19}`)를 추가해 원래 범위로 복원했습니다. 재확인 결과:

- 새 `documents.jsonl`의 PDF002 포함 페이지: **[17, 18, 19]**만(직접 확인)
- PDF002 text length: 39,996자 → **3,715자**
- 400/60 chunk 수 606 → 563(**−43**)은 **PDF002의 chunk 수 변화(46→3, −43)로 정확히
  전부 설명됨** — PDF002를 제외한 나머지 20개 문서는 chunk 수·chunk_id·text가
  reference baseline 버전과 **byte 단위로 완전히 동일**함을 직접 hash 비교로 확인했습니다.

### 4.2 Gold Boundary Validation

새 400/60 corpus/chunk 기준으로 167개 evidence 전체의 gold mapping을 **처음부터
재계산**했습니다(reference baseline repository의 fuzzy fallback 로직을 그대로 재사용, 재사용
아님 재실행). 결과:

- **167/167 matched, unmatched = 0**
- 이후 "evidence가 chunk 경계에 걸쳐 있는데 gold가 단일 chunk만 가리키는" 33건을
  전수 재검토 → 30건은 실제로 인접 chunk가 필요한 경계-절단 사례로 확인되어
  multi-gold로 보정, 나머지 3건은 검토 결과 이미 단일 chunk로 충분한 것으로 확인
  (자동 검사 스크립트가 fuzzy 매칭 span이 필요 이상으로 넓게 잡히면서 생긴
  일시적 오탐 — 사람이 직접 재확인 후 원래 단일-gold 상태를 그대로 유지)
- 최종: **single gold 125건 / multi gold 42건**(자연 발생 12건 + 경계보정 30건),
  **PARTIAL = INVALID = CORPUS_ERROR = UNMATCHED = 0**

## 5. Final Dataset

| 항목 | 값 |
|---|---|
| Documents | 21개 (PDF 8, Web 13) |
| Chunk setting | 400 tokens / 60 overlap |
| Tokenizer | BAAI/bge-m3 |
| Chunks (400/60) | **563개** |
| Gold evidence | 167개 (전체 evaluation dataset 120문항 기준) |
| Gold coverage | 167/167 = **100%**(unmatched 0) |
| Gold 구성 | single 125 / multi 42 |
| PDF002 pages | 17, 18, 19 (원본 설계 복원) |

(300/50 chunks 764개, 500/80 chunks 530개는 이번 보정 대상이 아니며 reference baseline 버전
그대로 보존되어 있습니다 — 아래 10번 참고)

## 6. Validation / Test Split

`rag_evaluation_dataset.jsonl`을 직접 읽어 재확인: **validation 80문항, test
40문항, overlap 0**(question_id 집합 기준). 606-chunk baseline 버전과 Final 버전의 split
정의(question_id → split 매핑)도 완전히 동일함을 확인했습니다. Validation은
파이프라인 조정·chunk 크기 비교용, Test는 그 어떤 조정에도 쓰이지 않은 held-out
평가용으로 구분됩니다.

## 7. 606-chunk baseline vs Final 2×2 Comparison (400/60)

| Version | Split | N | Chunks | BM25 R@20 | Dense R@20 | Hybrid R@20 | Hit@20 | Reranker R@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 606-chunk baseline | Validation | 80 | 606 | 0.726 | 0.831 | 0.915 | 0.950 | 0.784 | 0.825 | 0.703 | 0.679 |
| 606-chunk baseline | Test | 40 | 606 | 0.7492 | 0.9333 | 0.9125 | 0.925 | 0.7958 | 0.825 | 0.6279 | 0.6284 |
| Final | Validation | 80 | 563 | 0.7385 | 0.8719 | 0.9427 | 0.975 | 0.8125 | 0.850 | 0.7158 | 0.6970 |
| **Final** | **Test** | **40** | **563** | **0.7492** | **0.9333** | **0.9125** | **0.925** | **0.7958** | **0.825** | **0.6342** | **0.6334** |

**606-chunk baseline/Validation 행 주의**: 이 값은 `test_korean_only.md`(작업 디렉토리 최상위,
어떤 repo git 이력에도 없음)에만 기록된 **reported-only 결과**입니다. 원본 JSON
산출물이 reference baseline repo 어디에도 없어 재실행 검증이 불가능하며, 형식은 실제
`eval_retrieval.py` 출력과 일치하나 100% 재현 검증은 하지 못했습니다.

나머지 3행(606-chunk baseline/Test, Final/Validation, Final/Test)은 전부 이 repo 안의 실제
JSON 파일(`retrieval_eval/reference_baseline/606_chunk_baseline/results/results_400_60_ko_test.json`,
`retrieval_eval/results/results_400_60_ko_validation.json`,
`retrieval_eval/results/results_400_60_ko_test.json`)을 직접 읽어 확인한 값입니다.

## Final Korean Retrieval Architecture

§7까지는 최종 Top-5 생성에 Hybrid(RRF) 후보만 사용했기 때문에, candidate 단계에서
더 높은 성능을 보인 Dense를 동일한 reranker에 직접 연결했을 때의 결과는 알 수
없었습니다. 최종 한국어 architecture를 결정하기 위해 **동일한 Final Test 40문항,
400/60 corpus(563 chunks), gold labels, candidate top-20, final top-5 및
`BAAI/bge-reranker-v2-m3`** 조건에서 아래 두 pipeline을 추가 비교했습니다.

| Pipeline | Recall@20 | Hit@20 | MRR@20 | Recall@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense → Reranker | 0.9333 | 0.9500 | 0.6244 | 0.7792 | 0.8250 | 0.6446 | 0.6376 |
| Hybrid(RRF) → Reranker | 0.9125 | 0.9250 | 0.5415 | 0.7958 | 0.8250 | 0.6342 | 0.6334 |

Query-level candidate hit 비교에서는 Dense-only 2건, Hybrid-only 1건, 둘 다 hit
36건, 둘 다 miss 1건이었습니다. Reranker 이후 first-gold rank는 Dense 우위 2건,
Hybrid 우위 2건, 동일 36건으로 어느 쪽도 일관된 rank 우위를 보이지 않았습니다.

Hybrid는 evidence **Recall@5가 0.0167 높지만 Hit@5는 0.825로 동일**합니다. 반면
Dense는 candidate **Recall@20·Hit@20·MRR@20이 모두 높고**, 최종 **MRR@5와
nDCG@5도 소폭 높습니다**. BM25가 Dense를 보완한 사례가 1건 존재하지만, 반대로
Hybrid에서 Dense의 gold 후보를 잃은 사례가 2건이므로 BM25의 complementary signal은
존재하되 현재 Test 40에서는 일관된 순이익으로 보기 어렵습니다.

따라서 이는 큰 성능 격차를 뜻하는 결론은 아니지만, 현재 지표와 pipeline 단순성을
함께 고려한 최종 한국어 retrieval architecture는 다음과 같습니다.

> **BGE-M3 Dense → Top-20 → bge-reranker-v2-m3 → Top-5**

상세 결과와 question-level 분석은
`retrieval_eval/reports/dense_vs_hybrid_reranker_test.md`, full-precision 결과는
`retrieval_eval/results/results_400_60_dense_vs_hybrid_reranker_test.json`에 기록했습니다.

## 8. Interpretation

### 8.1 606-chunk baseline → Final, 동일 Test(40)에서의 변화

BM25/Dense/Hybrid의 **Recall@20·Hit@20은 소수점까지 완전히 동일**합니다
(0.7492/0.9333/0.9125, Hit@20 0.925 등). Reranker의 Recall@5(0.7958)·Hit@5(0.825)도
동일합니다. 달라진 것은 MRR@20(0.4168→0.3951), Hybrid final MRR@5(0.5438→0.5208)·
nDCG@5(0.5493→0.5252), Reranker MRR@5(0.6279→0.6342)·nDCG@5(0.6284→0.6334) 같은
**랭킹 세부 지표 몇 개뿐**, 그것도 ±0.02 내외입니다.

**이 결과는 "성능이 크게 향상됐다"는 뜻이 아닙니다.** 이번 작업의 목적은 성능
개선이 아니라 **corpus/gold 데이터의 신뢰성 검증 및 보정**이었고, 그 결과
**기존에 이미 확보되어 있던 성능 수준이 정상적으로 유지됨을 확인**한 것으로
해석해야 합니다.

### 8.2 왜 Test 결과가 거의 동일한가

확인된 artifact 기준으로 설명 가능한 근거:

- PDF001 표 손실 복구(§3.1)는 **이미 reference baseline 버전에 반영되어 있었으므로**, reference baseline→
  final 비교에서는 이 개선분이 나타나지 않습니다(reference baseline 자체가 이미 개선된
  상태였음).
- reference baseline→final의 실질적 corpus 변경은 **PDF002 페이지 범위 축소(§4.1)가
  사실상 전부**입니다.
- PDF002는 test/validation 어느 evidence의 gold_chunk_ids에도 등장하지 않습니다
  (167개 evidence의 소스 문서 목록을 확인한 결과 PDF002는 포함되지 않음).
- 나머지 20개 문서의 chunk는 reference baseline과 byte 단위로 동일하므로, BM25/Dense 인덱스가
  이 문서들에 대해 사실상 동일한 입력을 받습니다.

다만 이는 "완전히 무관하다"는 단정이 아니라, **corpus 크기가 줄면 BM25 IDF나
Dense 상대 유사도의 미세한 재정렬이 일어날 수 있다는 점에서 §8.1의 소폭 MRR/nDCG
변동과 정합적**이라는 정도로만 서술합니다. 검색 실행 자체의 오류 가능성을
완전히 배제할 근거는 없으나, 두 실행(reference baseline/test, final/test) 모두 동일한
`eval_retrieval.py`(byte 단위 확인)로 수행되어 코드 차이에 의한 왜곡은 아닙니다.

### 8.3 Validation vs Test 차이 (Final, 동일 파이프라인)

Final 내에서 Validation(80)과 Test(40)를 비교하면 MRR@5(0.716 vs 0.634)·
nDCG@5(0.697 vs 0.633)에서 Test가 뚜렷이 낮습니다. Validation과 Test는 overlap이
없는 서로 다른 문항 집합이므로, 이 차이를 corpus 품질 문제로 해석하지 않습니다.
두 split 간 문항 난이도·유형 분포가 다를 가능성이 있다는 정도로만 서술하며,
추가 원인 분석 없이 인과관계를 단정하지 않습니다.

## 9. Recommended Final Configuration

- **Chunk setting: 400 tokens / 60 overlap (BGE-M3 tokenizer), 563 chunks**
- **최종 한국어 pipeline: BGE-M3 Dense top-20 → bge-reranker-v2-m3 → top-5**
- Final/Test(40, held-out) 기준 Dense → Reranker 성능: Recall@5 **0.7792**,
  Hit@5 **0.825**, MRR@5 **0.6446**, nDCG@5 **0.6376** (출처:
  `retrieval_eval/results/results_400_60_dense_vs_hybrid_reranker_test.json`)
- Hybrid → Reranker의 Recall@5는 **0.7958**로 더 높지만, Hit@5가 동일하고 Dense가
  candidate 지표와 최종 MRR/nDCG에서 우세하며 BM25/RRF가 추가하는 복잡도를 함께
  고려해 Dense → Reranker를 최종 baseline으로 선택합니다.
- Validation(80) 수치는 chunk 크기 비교나 파이프라인 튜닝 참고용으로만 사용하고,
  대표 성능으로 인용하지 않을 것을 권장(§8.3 근거).

## 10. Remaining Work (아직 하지 않은 것)

- 한국어 baseline은 **BGE-M3 Dense → Reranker**로 확정
- 영어(EN) query에서는 아래 retrieval 신호를 별도로 비교·fusion하여 lexical/sparse
  signal이 Dense를 실제로 보완하는지 평가 예정:
  - English → Korean translation → Nori BM25
  - English → BGE-M3 Sparse
  - English → BGE-M3 Dense
- 300/50·500/80 chunk 버전에 대한 이번 단계의 gold boundary 재검증(현재는 reference baseline
  버전 그대로 보존, 재검증되지 않음)
- 다국어(cross-lingual) retrieval 전략 검토
- 팀 대상 최종 발표/보고서 작성

---

## Appendix — Source Artifacts

| 경로 | 역할 |
|---|---|
| `retriever_dataset/documents/documents.jsonl` | 최종 21개 문서(PDF002 페이지 필터 반영) |
| `retriever_dataset/chunks/chunk_400_60/chunks.jsonl` | 최종 563개 400/60 chunk |
| `retriever_dataset/chunks/chunk_300_50/chunks.jsonl` | 300/50 chunk(reference baseline 버전 보존, 이번 단계 미변경) |
| `retriever_dataset/chunks/chunk_500_80/chunks.jsonl` | 500/80 chunk(reference baseline 버전 보존, 이번 단계 미변경) |
| `retriever_dataset/metadata/corpus_statistics.json`, `chunk_statistics_400_60.json` | 최종 corpus/chunk 통계(재생성됨) |
| `retriever_dataset/metadata/chunk_statistics_300_50.json`, `chunk_statistics_500_80.json`, `duplicate_report.json` | reference baseline 버전 보존(미변경) |
| `rag_evaluation_dataset.jsonl` | 최종 evaluation dataset(400/60 gold_chunk_ids 갱신, 300/50·500/80 필드는 reference baseline 값 보존) |
| `retrieval_eval/prepare_retrieval_data.py` | PDF 재추출/청킹/gold 생성 파이프라인(PDF002 필터 추가된 버전) |
| `retrieval_eval/eval_retrieval.py` | BM25/Dense/Hybrid/Reranker 평가 스크립트(reference baseline 버전과 byte 단위 동일, 미변경) |
| `retrieval_eval/regenerate_400_60_only.py` | documents + 400/60 chunk 재생성 드라이버 |
| `retrieval_eval/regenerate_gold_400_60.py` | 400/60 gold mapping 처음부터 재계산 드라이버 |
| `retrieval_eval/apply_boundary_fix_400_60.py` | boundary-aware multi-gold 보정 드라이버 |
| `retrieval_eval/gold/gold_400_60_recomputed.jsonl` | 보정 전(single-gold only) 중간 산출물 |
| `retrieval_eval/gold/gold_400_60_final.jsonl` | 보정 후 최종 gold(= `rag_evaluation_dataset.jsonl`과 동일 내용) |
| `retrieval_eval/gold/gold_quality_risk_audit_400_60.csv` | Risk-based 1차 audit 기록(RISK 130건 전수 + LOW_RISK 20건 표본, 총 150건) |
| `retrieval_eval/results/results_400_60_ko_test.json` | Final/Test40 실행 결과 |
| `retrieval_eval/results/results_400_60_ko_validation.json` | Final/Validation80 실행 결과 |
| `retrieval_eval/results/results_400_60_gold_before_after_comparison.json` | gold 보정 전/후 성능 비교 |
| `retrieval_eval/results/results_400_60_2x2_matrix.json` | 2×2 매트릭스 정리본 |
| `retrieval_eval/reports/dense_vs_hybrid_reranker_test.md` | Final/Test40 Dense→Reranker vs Hybrid→Reranker 상세 비교 및 architecture 결정 |
| `retrieval_eval/results/results_400_60_dense_vs_hybrid_reranker_test.json` | 위 architecture 비교의 full-precision 지표 및 question-level 결과 |
| `retrieval_eval/results/results_300_50_ko_test.json`, `results/results_500_80_ko_test.json`, `results/unmatched_evidence_*.json` | reference baseline 버전 보존(이번 단계 미변경, 42~43% 시절 gold 기준 stale 값 포함 가능) |
| `retrieval_eval/reports/final_data_validation.md` | STEP 1~8 데이터 재생성·검증 상세 기록(이전 단계 산출물) |
| `retrieval_eval/compare_gold_before_after.py` | gold 보정 전/후 비교 스크립트 |
| `retrieval_eval/reference_baseline/606_chunk_baseline/` | 606-chunk baseline documents/606 chunks/gold를 이 repo 안에 복사해 독립적으로 재평가한 결과 일체 |
| `retrieval_eval/FinAgent_Retrieval_Eval_Colab.ipynb` | Colab 평가 노트북(reference baseline 버전 보존) |
| `README.md` | 이 repo 전체에 대한 짧은 개요 |
