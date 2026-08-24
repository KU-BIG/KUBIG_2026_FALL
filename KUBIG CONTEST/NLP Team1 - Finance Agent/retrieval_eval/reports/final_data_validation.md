# 400/60 Final Data Validation (KUBIG_FINANCE_final_test)

작업 범위: 400/60 chunk variant만. 300/50, 500/80은 손대지 않음.
모든 수정/실행은 `KUBIG_FINANCE_final_test` 내부에서만 수행. `KUBIG_FINANCE`,
606-chunk reference baseline repository는 참조만 하고 수정하지 않음.

## STEP 1. PDF002 page filtering 복원

`retrieval_eval/prepare_retrieval_data.py`에 `PDF_PAGE_FILTERS = {"PDF002": {17, 18, 19}}`
추가, `extract_pdf_pages()`가 1-indexed page 번호 기준으로 필터링하도록 수정
(`pdfplumber.pages`는 0-indexed이므로 `pages[page_number - 1]` 대응 확인).
또한 로컬 환경에서 실제로 PDF가 있는 경로(`data/raw/pdf/`)와 기존 `RAW_PDF_DIR`
정의(`data/raw/`)가 어긋나 있어 `RAW_PDF_DIR`를 `data/raw/pdf`로 정정(내용상
PDF002 필터링과 무관한, 이 환경에서 스크립트를 실행 가능하게 만들기 위한
불가피한 보정).

결과: PDF002 포함 페이지 = **[17, 18, 19]**만(직접 확인), text length **3,715자**
(구 kangminhyeok02 버전 3,493자와 근접, pdfplumber 버전/줄바꿈 처리 차이로 추정
되는 미세한 차이는 있으나 페이지 범위는 정확히 일치).

## STEP 2. Documents 재생성

`retrieval_eval/regenerate_400_60_only.py` 실행 → `documents/documents.jsonl` 덮어씀.

- 전체 document 수: 21 (PDF 8, Web 13)
- PDF002 외 20개 document는 reference baseline과 **byte 단위로 완전 동일**(직접 비교 확인)
- PDF002만 39,996자 → 3,715자로 축소

## STEP 3. 400/60 chunks 재생성

- 전체 chunk 수: **563** (reference baseline 606개에서 PDF002 chunk 46→3개로 줄어든 만큼 감소,
  606-46+3=563 정확히 일치)
- PDF001 317개, PDF002 3개(신규), PDF004 6, PDF005 5, PDF006 5, PDF007 5, PDF008 98,
  PDF009 41 / Web 13개 문서 합계 83개 — PDF002를 제외한 나머지는 reference baseline과 **chunk_id,
  text 완전 동일**(직접 diff 확인, PDF001·PDF008·WEB004 각각 0건 차이)
- 평균 토큰 301.09, 최대 400, 최소 2

### Coverage integrity

21개 문서 전체에서 uncovered 콘텐츠 = **`[Page N]` 마커 텍스트뿐**(실 본문 손실 0),
tail loss 전체 0, 같은 페이지 내 연속 chunk overlap 정상.

## STEP 4. PDF001 Regression — **PASS**

| 확인 항목 | 결과 |
|---|---|
| p.56 "월 2만 원 ~ 50만 원의 금액을 자유롭게 납입" | ✅ 존재 |
| p.56 "1개월 이내 무이자" | ✅ 존재 |
| p.62 "신용한도 금액" | ✅ 존재 |
| p.62 "구매 후 매월 정해진 결제일" | ✅ 존재 |
| p.62 "할부구매"(Yes/가능) | ✅ 존재 |
| p.162~168 "Woori Bank"/"우리은행" 등 은행명 | ✅ 7개 페이지 전체 존재 |

## STEP 5. Gold mapping 재계산 (base, from scratch)

`retrieval_eval/regenerate_gold_400_60.py` — 기존 gold_chunk_ids를 재사용하지 않고
`prepare_retrieval_data.py`의 `locate_source_span`/`best_gold_chunks` 로직을 새
chunks에 대해 처음부터 재실행.

- Total evidence: 167 / Matched: 167 / Unmatched: **0**
- Single gold: 155 / Multi gold(자연 발생): 12
- Match method: normalized_exact 74, line_window 93 (reference baseline과 동일 분포 — 코퍼스가
  PDF002 외에는 완전히 동일하므로 예상된 결과)

## STEP 6. Boundary-aware multi-gold 보정

이전 audit(`gold/gold_quality_risk_audit_400_60.csv`)의 PARTIAL_MULTI_GOLD_FIXABLE 33건을
**참고 목록(어떤 evidence를 재검토할지)**으로만 사용, chunk ID는 새로 계산된 값을 사용.
`retrieval_eval/apply_boundary_fix_400_60.py`로 33건 전부 재검토:

- **CASE_B (인접 chunk 추가로 해결)**: 30건
- **CASE_A로 재판정**(자동 스크립트가 처음엔 "해결 안 됨"으로 오판했으나, 원인을
  직접 대조한 결과 실제로는 이미 단일 gold로 충분): **3건**(ACC_024, ACC_027, ACC_028)
  — 원인: `source_quote_raw`(line_window fuzzy fallback)가 필요 이상으로 넓은 구간을
  긁어와서, 그 구간 안의 무관한 문장까지 "빠진 사실"로 오탐지됨. quote 자체를
  기준으로 재확인한 결과 현재 gold chunk 1개로 이미 충분함을 확인(2단계 전 조사에서
  이미 이 3건은 "verbatim은 아니지만 유효한 근거" Case B로 확정된 evidence와 동일).
- **CASE_C(진짜 미해결)**: 0건

## STEP 7. 최종 167건 분류

| Classification | Count |
|---|---:|
| VALID_SINGLE_GOLD | 125 |
| VALID_MULTI_GOLD | 42 |
| PARTIAL | 0 |
| INVALID | 0 |
| CORPUS_ERROR | 0 |
| UNMATCHED | 0 |
| **합계** | **167** |

(단일→다중 42건 = 자연 발생 12건 + boundary 보정 30건)

## STEP 8. 최종 파일

- `retriever_dataset/documents/documents.jsonl` — 최종 documents(PDF002 필터 반영)
- `retriever_dataset/chunks/chunk_400_60/chunks.jsonl` — 최종 400/60 chunks
- `retriever_dataset/metadata/corpus_statistics.json`,
  `metadata/chunk_statistics_400_60.json` — 갱신된 통계
- `rag_evaluation_dataset.jsonl`(KUBIG_FINANCE_final_test 루트) — 400/60 gold_chunk_ids
  최종 반영(300_50/500_80 필드는 기존 reference baseline 값 그대로 보존, 손대지 않음)
- `retrieval_eval/gold/gold_400_60_recomputed.jsonl`, `retrieval_eval/gold/gold_400_60_final.jsonl` —
  STEP5/STEP6 중간 산출물(작업 이력 보존용)
- `retrieval_eval/regenerate_400_60_only.py`, `retrieval_eval/regenerate_gold_400_60.py`,
  `retrieval_eval/apply_boundary_fix_400_60.py` — 이번 작업에 사용한 재현 가능 스크립트
