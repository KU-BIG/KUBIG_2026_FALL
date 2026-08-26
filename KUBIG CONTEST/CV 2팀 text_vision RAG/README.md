# 문서 해상도에 따른 Text RAG vs Vision RAG 비교

MMLongBench-Doc 60문서 266문항에 대해, 144 / 72 / 36 DPI 세 조건에서 검색과 생성 성능을 측정하고, 교차 조건(T→V)을 추가해 성능 격차의 원인을 **검색 방식**과 **입력 형태**로 분해했다.

## 연구 질문

[M3DocRAG](https://arxiv.org/abs/2411.04952) (NeurIPS 2024)은 페이지 이미지를 그대로 검색하는 Vision RAG가 기존 OCR 기반 방식보다 우수하다고 보고했다. 그러나 논문의 실험은 깨끗한 PDF를 전제한다. 실무에서 다루는 문서는 스캔본, 저해상도 이미지인 경우가 많다.

> **문서 품질이 나빠져도 Vision RAG의 우위가 유지되는가?**

## 실험 설계

### 세 조건

| 조건 | 검색기 (Retriever) | 생성 입력 (Generator Input) | 역할 |
|:---:|:---|:---|:---|
| **T→T** | ColBERT v2 (OCR 텍스트 색인) | OCR 텍스트 | 실용 텍스트 RAG |
| **T→V** | ColBERT v2 (OCR 텍스트 색인) | 페이지 이미지 | 교차 조건 (원인 분해용) |
| **V→V** | ColPali v1.2 (페이지 이미지 색인) | 페이지 이미지 | 실용 비전 RAG |

### 통제변인

- **생성기**: Qwen2.5-VL-7B — 세 조건 동일
- **프롬프트**: 세 조건 동일
  ```
  Answer the question based on the given document pages.
  Output only the final answer, with no explanation.

  Question: {q}
  Answer:
  ```
- **top-k**: 5
- **디코딩**: greedy (`do_sample=False`, `max_new_tokens=64`)
- **이미지 해상도 설정**: `MAX_PIXELS` / `MIN_PIXELS` 동일 (T→V와 V→V)

### M3DocRAG와의 차이

M3DocRAG은 텍스트 경로에 Llama 3.1, 비전 경로에 Qwen2-VL을 써서 검색기와 생성기가 동시에 바뀐다. 성능 차이의 원인을 특정할 수 없다. 본 실험은 **생성기를 고정**해 검색 방식과 입력 형태의 효과만 분리했다.

### T→V를 넣은 이유

T→T와 V→V만 비교하면 검색 방식과 입력 형태가 교락(confound)된다. T→V는 T→T와 입력 형태만, V→V와 검색 방식만 다르므로 두 요인을 분해할 수 있다.

- **T→V − T→T** = 입력 형태의 순효과
- **V→V − T→V** = 검색 방식의 순효과

**정합성 검증** (설계가 의도대로 작동했다는 증거):
- T→V의 Recall@5가 T→T와 세 DPI 모두 정확히 일치 (70.7 / 69.9 / 50.0)
- T→V의 입력 토큰이 V→V와 사실상 동일 (11,608 vs 11,616 / 2,966 vs 2,968 / 811 vs 811)

## 데이터

[MMLongBench-Doc](https://arxiv.org/abs/2407.01523) (NeurIPS 2024) — **60문서 / 266문항 / 1,780페이지**

### 문항 필터링

```
1,091 전체 → 517 (선정 문서 소속) → 413 (답변 가능)
→ 411 (페이지 범위 정상, 데이터 오류 2건 제외) → 266 (single-page)
```

**single-page만 사용한 이유**: Recall@5 해석을 명확히 하기 위해서다. multi-hop은 오답 시 검색 실패인지 추론 실패인지 구분되지 않는다.

### 문서 선정

전체 132문서 중 50페이지 이하로 제한. 단 재무보고서(10-K)는 형식상 장문이 불가피해 80페이지까지 허용 (COSTCO 76p, NETFLIX 72p, BESTBUY 75p). 이 예외가 없으면 재무보고서가 2문서 11문항만 남아 관련 가설을 검증할 수 없다. `doc_type` 7종에 비례 배분하고, 근거유형(Chart / Figure / Table / Text / Layout)이 고르게 분포하도록 최적화했다.

**구성**: Research report 67 / Academic paper 46 / Financial report 44 / Guidebook 29 / Administration 29 / Brochure 26 / Tutorial 25

## 결과

### 메인 결과

| Method | DPI | R@5 | R@1 | Acc | Acc\|Ret |
|:---|:---:|:---:|:---:|:---:|:---:|
| **T→T** ColBERT + OCR text | 144 | 70.7 | 49.6 | 37.6 | 47.3 |
| | 72 | 69.9 | 48.9 | 35.0 | 43.0 |
| | 36 | 50.0 | 26.7 | 12.0 | 18.8 |
| **T→V** ColBERT + page image | 144 | 70.7 | 49.6 | 51.1 | 65.4 |
| | 72 | 69.9 | 48.9 | 44.0 | 56.5 |
| | 36 | 50.0 | 26.7 | 21.1 | 29.3 |
| **V→V** ColPali + page image | 144 | 84.6 | 60.5 | 57.9 | 63.1 |
| | 72 | 84.2 | 59.0 | 50.0 | 56.7 |
| | 36 | 84.6 | 61.7 | 23.3 | 25.8 |

> R@5/R@1 = Recall@5/@1, Acc = 답변 정확도, Acc|Ret = 검색 성공 시 정확도

### 요인 분해 (답변 정확도)

| DPI | 입력 형태 (T→V − T→T) | 검색 방식 (V→V − T→V) | 전체 격차 |
|:---:|:---:|:---:|:---:|
| 144 | +13.5%p | +6.8%p | +20.3%p |
| 72 | +9.0%p | +6.0%p | +15.0%p |
| 36 | +9.0%p | +2.3%p | +11.3%p |

### OCR 품질 (MinerU, 36 DPI 붕괴의 직접 원인)

| DPI | 페이지당 글자수 | 영어단어 비율 | 정답페이지 내 정답 잔존 |
|:---:|:---:|:---:|:---:|
| 144 | 2,269 | 81.7% | 39% |
| 72 | 2,117 | 83.2% | 38% |
| 36 | 662 | 78.3% | 16% |

## 재현 방법

중간 산출물은 저장소에 포함되어 있지 않다. 아래 절차로 동일하게 생성할 수 있다.

### 1. 데이터 다운로드

HuggingFace [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc)에서 데이터를 다운로드한다. `data/raw/documents/`에 PDF를, `data/samples.json`에 문항 파일을 배치한다.

### 2. 문항 필터링

```bash
python preprocess/filter_questions.py
```

`data/samples.json`과 `data/docs_60.txt`를 읽어 266문항을 선별한다.
출력: `data/full_samples.json`, `data/full_docs.txt`

### 3. 페이지 렌더링

```bash
python preprocess/render_pages.py          # 144, 72, 36 DPI 전부
python preprocess/render_pages.py 144      # 특정 DPI만
```

출력: `data/pages/{dpi}/{doc_id}/{page:03d}.png` — PNG 약 5,340장

### 4. 텍스트 검색 (MinerU OCR → ColBERT)

```bash
# (a) OCR — MinerU API 서버가 실행 중이어야 한다
bash retrieval/text/run_ocr.sh

# (b) OCR 결과 수집
python retrieval/text/export_ocr_texts.py

# (c) ColBERT용 청크 데이터 생성
python retrieval/text/prepare_colbert_data.py

# (d) ColBERT 검색
python retrieval/text/run_colbert.py
```

출력: `retrieval_text_{dpi}.json`

### 5. 비전 검색 (ColPali)

```bash
# (a) 페이지 임베딩 생성
python retrieval/vision/build_indices.py

# (b) 검색
python retrieval/vision/run_retrieval.py
```

출력: `retrieval_vision_{dpi}.json`

### 6. 답변 생성

```bash
python generation/generate_text.py 144      # T→T
python generation/generate_text.py 72
python generation/generate_text.py 36

python generation/generate_cross.py 144     # T→V
python generation/generate_cross.py 72
python generation/generate_cross.py 36

python generation/generate_vision.py 144    # V→V
python generation/generate_vision.py 72
python generation/generate_vision.py 36
```

출력: `data/answers_{text,text(v),vision}_{dpi}.json`

### 7. 채점

```bash
python eval/score.py text
python eval/score.py text(v)
python eval/score.py vision
```

출력: `data/scored_{pipe}_{dpi}.json`, `data/review_{pipe}.json`

## 입출력 규약

- **페이지 번호는 전부 1-based.** PNG 파일명 `001.png` = 문서의 1페이지. PyMuPDF는 `doc[0]`이 1페이지이므로 결과에 실을 때 +1이 필요하다.

- **검색 결과 JSON 스키마:**
  ```json
  {
    "question_id": 0,
    "doc_id": "example.pdf",
    "question": "...",
    "top_pages": [3, 7, 1, 12, 5],
    "scores": [45.12, 42.30, 39.88, 38.11, 37.05]
  }
  ```
  `top_pages`는 1-based 정수 5개, 점수 내림차순.

- **OCR 결과 JSON 스키마:**
  ```json
  [
    {"doc_id": "example.pdf", "page_number": 1, "text": "..."},
    {"doc_id": "example.pdf", "page_number": 2, "text": "..."}
  ]
  ```
  `page_number`는 1-based.

## 주요 발견

### 1. ColPali 검색은 해상도에 불변하다

Recall@5가 84.6 / 84.2 / 84.6%로 사실상 동일하다. ColPali가 입력을 448x448로 고정 리사이즈하므로 원본 해상도가 임베딩에 거의 반영되지 않는다. 반면 ColBERT는 70.7 → 50.0%로 무너진다.

원인은 OCR이다. MinerU가 36 DPI에서 페이지당 글자수를 2,269 → 662로 떨어뜨린다. 주목할 점은 영어단어 비율이 81.7 → 78.3%로 거의 안 떨어진다는 것이다. 글자가 깨진 게 아니라 아예 인식되지 못하고 누락됐다.

### 2. 그러나 강건성은 검색 단계에 국한된다

검색 성공 시 정확도가 V→V 기준 63.1 → 25.8%로 무너진다. 36 DPI에서 정답 페이지를 84.6% 찾아내고도 읽지 못한다. 입력 토큰이 11,616 → 811로 줄어 판독 정보량이 부족하기 때문이다.

> 검색에 필요한 해상도와 판독에 필요한 해상도가 다르다.

### 3. 20.3%p 격차의 2/3는 검색기가 아니라 입력 형태에서 왔다

T→V를 끼워 넣으면 144 DPI에서 입력 형태 13.5%p, 검색 방식 6.8%p로 분해된다. 36 DPI에서는 입력 형태가 80%를 차지한다.

> "Vision RAG가 낫다"의 실질은 "ColPali가 더 잘 찾는다"가 아니라 "페이지를 이미지 그대로 생성기에 넘기는 설계가 낫다"이다.

## 한계

- 36 DPI 조건은 텍스트 파이프라인의 해상도 민감도가 아니라 MinerU의 동작 하한을 측정한 것에 가깝다. 다른 OCR로 재현이 필요하다.
- Recall@5는 페이지 단위 지표인데 ColBERT는 청크 단위(480토큰 / 50 오버랩)로 동작한다. 오버랩으로 인접 페이지에 정답 문구가 중복 포함되어 "검색 실패인데 정답"인 사례가 존재한다.
- 문서를 50페이지 이하로 제한했으므로(재무보고서만 80p 예외) 초장문 문서에 대한 일반화는 제한적이다.
- 채점은 규칙 기반 채점기를 수동 판정 162건으로 보정(일치율 91.4%)한 뒤 적용했고, 확신도가 낮은 사례는 LLM judge가 개별 검토했다.

## 참고문헌

| | |
|:---|:---|
| M3DocRAG | [arXiv:2411.04952](https://arxiv.org/abs/2411.04952) |
| MMLongBench-Doc | [arXiv:2407.01523](https://arxiv.org/abs/2407.01523) |
| ColPali | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) |
| ColBERTv2 | [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) |
| MinerU | [github.com/opendatalab/MinerU](https://github.com/opendatalab/MinerU) |
| Qwen2.5-VL | [github.com/QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
