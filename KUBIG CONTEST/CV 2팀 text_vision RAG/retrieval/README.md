# Retrieval

두 검색 파이프라인의 역할, 실행 순서, 입출력 형식을 정리한다.

## text/ — MinerU OCR + ColBERT v2

OCR로 추출한 텍스트를 청크 단위로 색인하고, ColBERT late interaction으로 검색한다.

### 실행 순서

| 단계 | 스크립트 | 입력 | 출력 |
|:---:|:---|:---|:---|
| 1 | `run_ocr.sh` | `data/pages/{dpi}/{doc_id}/*.png` | `ocr/raw/{dpi}/{doc_id}/*.md` |
| 2 | `export_ocr_texts.py` | `ocr/raw/` | `results/ocr_text_{dpi}.json` |
| 3 | `prepare_colbert_data.py` | `ocr/raw/` | `colbert_data/{dpi}/{doc_id}/collection.tsv`, `metadata.json` |
| 4 | `run_colbert.py` | `colbert_data/`, `data/full_samples.json` | `results/retrieval_text_{dpi}.json` |

### 세부 사항

- **OCR**: MinerU를 API 모드(`http://127.0.0.1:8000`)로 실행한다. 페이지별로 `.md` 파일을 생성한다.
- **청크**: 480토큰 / 50토큰 오버랩. `colbert-ir/colbertv2.0` 토크나이저 기준. HTML 테이블은 선형화(header + row 반복), 이미지 링크와 마크다운 서식은 제거한다.
- **검색**: Exact MaxSim. 청크 점수를 페이지 단위로 max-aggregation한 뒤 top-5를 반환한다. `doc_maxlen=512`, `query_maxlen=32`.

---

## vision/ — ColPali v1.2

페이지 이미지를 직접 임베딩하고, multi-vector late interaction으로 검색한다.

### 실행 순서

| 단계 | 스크립트 | 입력 | 출력 |
|:---:|:---|:---|:---|
| 1 | `build_indices.py` | `data/pages/{dpi}/{doc_id}/*.png` | `embeddings/{dpi}/{doc_id}.pt` |
| 2 | `run_retrieval.py` | `embeddings/`, `data/full_samples.json` | `output/retrieval_vision_{dpi}.json` |

### 세부 사항

- **모델**: `vidore/colpali-v1.2` (bfloat16, CUDA).
- **이미지 전처리**: SiglipImageProcessor가 448x448로 고정 리사이즈한다. 원본 비율과 무관하게 정사각형으로 변환되며, 이것이 해상도 불변성의 원인이다.
- **임베딩**: 페이지당 (1030, 128) 텐서. 1024 이미지 패치 토큰 + 6 instruction 토큰.
- **검색**: `ColPaliProcessor.score_multi_vector()`로 query-page MaxSim 점수를 계산, top-5 반환.
- **쿼리 임베딩은 DPI 독립**: 텍스트 기반이므로 한 번만 계산해 세 DPI에서 재사용한다.

---

## 출력 형식 (공통)

```json
{
  "question_id": 0,
  "doc_id": "example.pdf",
  "question": "...",
  "top_pages": [3, 7, 1, 12, 5],
  "scores": [45.12, 42.30, 39.88, 38.11, 37.05]
}
```

- `top_pages`: 1-based 페이지 번호 5개, 점수 내림차순.
- `scores`: MaxSim 점수 (float, 소수점 4자리).
