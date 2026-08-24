# English BGE-M3 Dense → Reranker Baseline (Final 400/60 Test40)

## Dataset validation

- 전체 120문항: Validation 80 / Test 40
- `question` schema: `{ko, en}` 120/120
- KO/EN non-empty: 120/120, empty EN 0
- duplicate question_id 0, duplicate EN question 0, EN field 내 한글 0
- evidence 167건, Final 400/60 gold 누락·unknown ID·aggregate mismatch 0
- KO와 EN은 동일 record 안에서 split/evidence/gold/gold_chunk_ids를 공유

자동 검사와 120개 KO/EN pair 및 한국 특화 용어 표본의 수동 sanity check에서 평가를 무효화할 심각한 오류는 발견되지 않았습니다. 다만 이 검수만으로 모든 번역의 완전한 의미 동일성을 보증하지는 않습니다. FRD_006은 EN의 `stop the transfer`가 KO의 지급정지 취지보다 덜 구체적인 경미한 wording risk로 기록했습니다.

## Experiment configuration

- English Query → BAAI/bge-m3 Dense → Top-20 → BAAI/bge-reranker-v2-m3 → Top-5
- Final 400/60 corpus, 563 chunks; corpus 번역 없음
- Test 40, 현재 Final gold, 기존 evidence-group-aware metric 재사용
- 기존 563-row corpus embedding cache 재사용. KO 40문항의 Top-20 ranking과 candidate metric이 기존 Korean artifact와 full precision으로 일치해 provenance를 검증

## Korean vs English

| Query | Pipeline | Recall@20 | Hit@20 | MRR@20 | Recall@5 | Hit@5 | MRR@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Korean | Dense → Reranker | 0.9333 | 0.9500 | 0.6244 | 0.7792 | 0.8250 | 0.6446 | 0.6376 |
| English | Dense → Reranker | 0.8958 | 0.9500 | 0.5649 | 0.8000 | 0.8750 | 0.6863 | 0.6543 |
| Δ EN−KO | — | -0.0375 | 0.0000 | -0.0595 | 0.0208 | 0.0500 | 0.0417 | 0.0168 |

## Query-level candidate comparison

- KO hit / EN hit: 37
- KO hit / EN miss: 1 — ['ACC_021']
- KO miss / EN hit: 1 — ['FRD_015']
- KO miss / EN miss: 1 — ['ACC_015']

### English-only failures relative to Korean

| ID | KO first gold | EN first gold | Korean question | English question | Gold chunks |
|---|---:|---:|---|---|---|
| ACC_021 | 1 | not in 50 | 신한은행 화상상담센터에서 통장과 체크카드, 온라인뱅킹까지 한 번에 신청하려고 해요. 체류카드가 있어도 여권을 꼭 가져가야 하나요? | I want to apply for an account, debit card, and online banking at Shinhan Bank's video consultation center. Do I still need my passport if I have a residence card? | PDF005_0004 |

## Observations and next experiments

- Candidate 단계에서 EN은 KO보다 Recall@20이 0.0375, MRR@20이 0.0595 낮았지만 Hit@20은 0.9500으로 같았습니다. 즉, 문항 단위 적중 수는 유지했으나 multi-evidence 회수와 최초 gold 순위는 약화됐습니다.
- Reranker 이후 EN은 이번 Test 40에서 KO보다 Recall@5 +0.0208, Hit@5 +0.0500, MRR@5 +0.0417, nDCG@5 +0.0168이었습니다. Candidate ranking 저하가 최종 Top-5 저하로 직결되지는 않았으며, 40문항이라는 작은 표본에서 EN이 KO보다 일반적으로 우수하다고 확장해석하지는 않습니다.
- `ACC_021`은 KO rank 1이지만 EN에서 Top-50 밖입니다. Gold chunk는 한국어 표 형식의 신한은행 디지털 화상상담센터 안내이며 `여권 지참 필수`를 포함합니다. 영어 질의의 `video consultation center`, `passport`, `residence card`와 한국어 표 chunk 사이의 cross-lingual lexical/structure mismatch가 관찰되지만, 단일 사례로 원인을 확정할 수는 없습니다.
- `FRD_015`는 반대로 KO Top-20에는 없고 EN rank 13에만 있었습니다. 영어가 일방적으로 열화된 것이 아니며, 표현별 embedding 순위 변동이 양방향으로 존재함을 보여줍니다. `ACC_015`는 둘 다 Top-20 miss였으며 EN에서는 rank 49였습니다.

### Current judgment

- **Dense-only baseline:** Hit@20 0.9500, reranker Hit@5 0.8750으로 강한 첫 baseline이며, 현재 단계의 영어 retrieval이 실패했다고 볼 근거는 없습니다. 다만 Recall@20과 MRR@20의 KO 대비 하락, `ACC_021`과 같은 cross-lingual miss 때문에 Dense만이 최종적으로 충분하다고 확정하기에는 이릅니다.
- **BGE-M3 Sparse:** 영어 query와 한국어 corpus 사이의 직접 lexical overlap이 제한적이므로 단독 성능을 미리 남지하지는 않습니다. 그럼에도 multilingual learned sparse signal이 Dense의 miss/multi-evidence 누락을 보완하는지 직접 측정할 가치가 있습니다.
- **Translated Nori BM25:** 영어를 한국어로 번역하면 `화상상담센터`, `여권 지참 필수` 같은 corpus lexical cue를 복원할 수 있어 실험 가치가 있습니다. 번역 오류·처리 비용이 추가되므로 Dense Top-20 miss를 실제로 보완하는지를 독립 평가한 뒤 fusion 여부를 결정해야 합니다.
