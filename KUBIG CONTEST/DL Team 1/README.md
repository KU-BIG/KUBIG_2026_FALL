# Kr News Rag

네이버 금융 뉴스 기반 한국어 RAG. BGE-M3 dense 검색과 BM25를 RRF로 합치고
Multi-Query·HyDE 질의 확장과 검색 게이트를 붙인 뒤, 50문항 평가셋으로 Dense와
Hybrid를 비교했습니다.

**팀** 24기 김려원 · 24기 김가현 · 24기 박민준 · 23기 신지유

종목 10개 · 2026-07-31 ~ 08-08
**모델** 임베딩 `BAAI/bge-m3` · 답변 Anthropic / OpenAI / OpenRouter 중 선택

<details>
<summary><b>▶ 데모 영상 3편</b> — 단일 질문, 출처 스트리밍, 대화 모드 (펼치기)</summary>

<br>

**단일 질문** — 질문 하나에 답변 하나. 검색 설정을 바꿔가며 비교할 때 쓰는 모드 (16초)

<video src="https://github.com/user-attachments/assets/6c3e5fe4-bf83-49df-ab09-f0f2551b5cd6" controls muted playsinline width="100%"></video>

[원본 내려받기 (오프라인용)](example_videos/rag_test1.mov)

**출처 먼저, 답변은 스트리밍** — 검색이 먼저 끝나므로 출처 카드가 약 0.15초에 뜨고
답변은 쓰이는 대로 흘러나옵니다 (27초)

<video src="https://github.com/user-attachments/assets/be9679df-601d-43ed-894e-a69482eacece" controls muted playsinline width="100%"></video>

[원본 내려받기 (오프라인용)](example_videos/rag_test2.mov)

**대화 모드 + 검색 게이트** — 이력을 기억해 후속 질문이 되고 잡담이나 재질문은
검색을 건너뜁니다 (41초)

<video src="https://github.com/user-attachments/assets/3cd5f04f-5ff8-4dd4-a93b-976d9e1745ae" controls muted playsinline width="100%"></video>

[원본 내려받기 (오프라인용)](example_videos/chat_test.mov)

</details>

> **데이터에 관하여** — 이 저장소에는 **뉴스 기사 본문이 들어 있지 않습니다.** 저작권이
> 각 언론사에 있어 재배포하지 않습니다. 코퍼스는 `preprocessing/crawl.py` → `clean.py`로
> 직접 수집합니다. 평가 파일에는 기사 제목과 관련성 판정 근거 문장(기사당 1~2문장)이
> 남아 있는데, 이게 없으면 평가셋을 읽을 수 없기 때문입니다.

---

## 실험 결과

### 설정

동결된 source-seeded 질문 **50개**로 Dense와 Hybrid를 **같은 질문에서 짝지어** 비교했습니다.

- 질문은 날짜 4개 층으로 층화 추출한 서로 다른 기사 50건에서 만들고 유형을
  exact_token 13 / abstract 13 / multi_aspect 12 / factoid 12로 배분
- 질문 제작에만 50건을 쓰고 **검색은 항상 전체 432기사·1,377청크**에서 수행
- 두 시스템의 상위 20기사를 합쳐 **1,299개 고유 query–candidate 쌍**을 만들고 시스템
  이름과 원래 순위를 가린 채 관련성 판정
- 판정은 2패스(1차 전량 → 관련·불확실 전량 + 비관련 10% 표본 재검토). 최종 `relevant`만
  gold에 추가하고 `uncertain`은 인정하지 않음
- 사람이 지정한 seed 기사 50건은 LLM 판정에서도 전부 `relevant` (일치율 1.00)

### 지표

| Metric | Dense | Hybrid | 차이 |
| --- | ---: | ---: | ---: |
| Hit@1 | 0.780 | **0.900** | +0.120 |
| Hit@3 | 0.920 | **0.960** | +0.040 |
| Hit@5 | 0.920 | **0.960** | +0.040 |
| MRR@5 | 0.833 | **0.923** | +0.090 |

Hybrid가 네 지표 모두에서 앞섰고 개선폭이 Hit@1과 MRR@5에 몰려 있습니다. **Hybrid의
이점은 새 문서를 찾아내는 것보다 이미 찾은 문서를 위로 올리는 데 있다**는 뜻입니다
(Hit@5는 +0.04에 그침).

### 이 차이는 유의한가

50문항 짝지은 비교라 승패가 갈린 질문 수 자체가 적습니다.

| Metric | Hybrid 승 | Dense 승 | 무승부 | p (양측) |
| --- | ---: | ---: | ---: | ---: |
| Hit@1 | 7 | 1 | 42 | 0.070 |
| Hit@3 | 2 | 0 | 48 | 0.500 |
| Hit@5 | 2 | 0 | 48 | 0.500 |
| MRR@5 | 8 | 1 | 41 | 0.039 |

**+12%p라는 Hit@1 격차는 α=0.05를 넘지 못합니다**(p=0.070). MRR@5만 p=0.039로 통과하는데,
지표 4개를 검정했으므로 다중비교를 보정하면(Bonferroni α=0.0125) 이것도 유의하지
않습니다.

방향은 네 지표에서 일관되게 Hybrid 쪽이지만 **크기는 이 표본으로 확정되지 않습니다.**
지금 관찰된 7:1 불일치 비율이 그대로 유지된다고 가정해도 검정력 0.80을 얻으려면 약
**85문항**이 필요하고 실제 승패가 조금만 더 팽팽해지면(6:2) **194문항**으로 뜁니다.
50문항은 애초에 이 크기의 차이를 확정할 수 있는 규모가 아니었습니다.

> 이 검정은 **사후 분석**입니다. 동결된 `paired_outcomes`로 계산한 정확 이항검정이며
> 평가 설계 시점에 신뢰구간이나 검정을 사전 지정하지 않았습니다(manifest의
> `rules.statistical_test = null`).

### 어디서 갈렸나

| 질문 유형 | Dense Hit@1 | Hybrid Hit@1 | 차이 |
| --- | ---: | ---: | ---: |
| multi_aspect (12) | 0.750 | **1.000** | +0.250 |
| exact_token (13) | 0.615 | **0.769** | +0.154 |
| abstract (13) | 0.769 | **0.846** | +0.077 |
| factoid (12) | 1.000 | 1.000 | 0.000 |

- **multi_aspect에서 가장 크게 벌어집니다.** 여러 측면을 한 기사가 다 받쳐야 하는
  질문이라, 키워드 신호가 붙는 Hybrid가 유리합니다.
- **exact_token에서 Dense Hit@1이 0.615로 가장 낮습니다.** 임베딩이 정확한 토큰을
  놓친다는 가설과 맞고 BM25를 섞는 근거가 됩니다. 다만 Hit@3은 양쪽 다 1.000이라
  Dense도 3위 안에는 넣습니다 — **순위 문제이지 회수 문제가 아닙니다.**
- **factoid는 양쪽 다 1.000으로 포화**라 이 유형으로는 아무것도 구분되지 않습니다.
- 날짜층으로는 2026-08-07(18문항)에서 Hit@1 0.722 → 1.000으로 가장 크게 벌어집니다.
  다만 층별 표본이 작아 탐색적으로만 봅니다.

**갈린 8문항이 전부 시장 전체를 묻는 질문입니다.**

Hit@1이 어긋난 8건을 전부 펼치면 이렇습니다. 개별 종목만 묻는 질문은 **한 건도**
갈리지 않았습니다.

| 질문 | 유형 | Dense | Hybrid | |
| --- | --- | ---: | ---: | --- |
| `K009` 8월 6일 코스피·코스닥 종가와 수급·업종 요인 | multi_aspect | 실패 | **1위** | Hybrid |
| `R015` 8월 7일 코스피 상승 출발 후 하락 마감 배경 | multi_aspect | 실패 | **1위** | Hybrid |
| `K015` 7일 국내 증시 반등에 필요한 수급 조건 | multi_aspect | 3위 | **1위** | Hybrid |
| `R013` 8월 7일 코스피·코스닥 하락 전환의 수급 요인 | abstract | 3위 | **1위** | Hybrid |
| `K011` 8월 7일 코스피 6300선 보합, 삼전·하닉 흐름 | exact_token | 3위 | **1위** | Hybrid |
| `R001` 7월 31일 코스피 외국인 순매수 금액 | exact_token | 2위 | **1위** | Hybrid |
| `R011` 8월 7일 코스피 반등 외국인 순매수 규모 | exact_token | 2위 | **1위** | Hybrid |
| `R007` 8월 6일 코스피 급락, 투자자별 순매수·순매도 | exact_token | **1위** | 3위 | Dense |

Dense도 대부분 2~3위에는 넣습니다. 못 찾는 게 아니라 **1위로 못 올리는 것**이고
Hybrid가 하는 일이 정확히 그 재정렬입니다.

시황·수급 기사는 같은 날 여러 매체가 거의 같은 내용을 씁니다. 후보들이 서로 비슷해
임베딩만으로는 우열이 안 서고 `6300선`·`외국인 순매수` 같은 정확한 토큰을 잡는 BM25가
순위를 가릅니다. 반대로 종목이 특정되면 Dense 단독으로 충분합니다.

**양쪽 다 실패한 2건**(`K003`, `R003`)도 같은 부류입니다 — 둘 다 8월 4일 코스피·코스닥
장중 흐름 질문입니다. 시황 기사가 여러 건이라 seed 하나만 정답으로 두면 내용상 맞는
기사를 찾아도 miss로 집계됩니다. blind pooling으로 추가 gold를 넣는 설계가 필요했던
이유이고 그러고도 남은 실패입니다.

> 요약하면 **Hybrid의 이득은 코퍼스의 좁은 한 구석에 몰려 있습니다.** 전체 평균 +12%p는
> 시장 전체 질문 8건에서 나온 것이고 나머지 42문항은 Hit@1이 동일합니다. 이 코퍼스가
> 일일 시황 기사를 많이 담고 있어서 그 구석이 작지 않을 뿐입니다.

### 한계

- 질문 50개는 소규모입니다. 위에서 본 대로 대부분의 지표에서 유의성을 못 냅니다.
- 질문 생성과 관련성 판정 모두 **AI 보조**이며 작성자 self-check는 있었지만 **독립적인
  사람 교차검수는 없습니다.**
- 1차 판정과 재검토를 **같은 모델**이 했습니다. 재검토에서 22건이 바뀌었지만
  (`relevant` → `not_relevant` 5, `uncertain` → `relevant` 7 등) 이건 사람 검증이 아닙니다.
- 최종 `uncertain` 24건은 gold에서 제외했습니다. 이들이 실제로 관련 문서라면 두 시스템
  모두 과소평가된 것입니다.
- source-seeded 방식이라 질문이 특정 기사에서 출발합니다. 실제 사용자 질문 분포와 다를
  수 있습니다.

### 재현

판정과 mapping이 동결돼 저장소에 있으므로 **지표 재계산은 API 없이 됩니다.** 판정이
붙어 있던 후보 기사 본문은 저장소에 없지만, 어디서 왔는지는
`blind_pool_50_sources.json`에 남아 있습니다 — 기사 371건의 URL·제목·날짜와, 동결 당시
본문의 SHA-256.

```bash
uv run python -m evaluation.rebuild_pool        # URL에서 기사를 다시 받아 pool 복원 (6~7분)
uv run python -m evaluation.final_evaluation    # 지표 재계산
```

이미 전체 수집을 해뒀다면 `--from-corpus`로 요청 없이 즉시 복원됩니다.

`rebuild_pool`은 **기사별로** 기록된 해시와 대조하고 어느 기사가 달라졌는지 이름을
댑니다. 파일 전체를 하나의 통과·실패로 보고하지 않는 이유는, 기사 한 건만 수정돼도
전체 해시가 어긋나서 진단이 안 되기 때문입니다.

> **시간이 지나면 일부는 되살아나지 않습니다.** 동결 이튿날 실측으로 371건 중 **365건이
> 바이트 일치, 6건이 드리프트**(기사 수정), 삭제 0건이었습니다. 이 비율은 시간이 갈수록
> 나빠집니다. 다만 **지표는 pool을 읽지 않습니다** — `final_evaluation`은 동결된 질문·
> mapping·판정만으로 계산하고, 셋 다 기사 본문이 없어 저장소에 그대로 있습니다. 복원한
> pool은 "어떤 기사를 두고 내린 판정인지" 읽어보기 위한 것입니다.

> ⚠️ **현재 재계산 명령은 그 앞의 무결성 검사에서 멈춥니다.** 두 manifest가 기록한
> `freeze_sha256`(`fee480d1…`)이 커밋된 `retrieval_eval_50.jsonl`의 실제 해시
> (`bc5e5480…`)와 다릅니다. 원인은 CRLF입니다 — 커밋본을 줄바꿈만 `\r\n`으로 바꿔
> 해싱하면 기록된 값이 정확히 나옵니다. pool을 만든 환경의 파일이 Windows 줄바꿈이었고,
> 이후 LF로 정규화된 버전이 커밋됐습니다. 평가 내용은 온전합니다(질문 문구·유형·seed
> 기사가 blind pool과 전부 일치, `combine` 재실행 시 커밋본 그대로 재생성). manifest의
> 해시를 실제 값으로 갱신하면 이 명령도 다시 돕니다.

- [최종 aggregate 및 subgroup metrics](evaluation/results/retrieval_eval_50_metrics.json)
- [질문별 짝지은 비교](evaluation/results/retrieval_eval_50_query_comparison.csv)
- [평가 설정·판정 진단·무결성 manifest](evaluation/results/retrieval_eval_50_manifest.json)
- [평가셋 제작 절차](evaluation/README.md)

---

## 환경 설정

### 1. uv

의존성과 Python 버전은 [uv](https://docs.astral.sh/uv/)로 관리합니다. `uv.lock`에
버전이 고정돼 있어 어느 머신에서든 같은 환경이 재현됩니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows(PowerShell): powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync          # 실행 + 개발(주피터) 의존성
uv sync --no-dev # 실행 의존성만
```

`.python-version`에 고정된 Python 3.13을 uv가 자동으로 내려받고 `.venv/`를 만듭니다.
별도의 `python -m venv`나 `pip install`은 필요 없습니다.

### 2. LLM 프로바이더

답변 생성·질의 확장·검색 게이트가 LLM을 씁니다. `LLM_PROVIDER`로 고르며 설정
스크립트가 `.env`까지 만들어 줍니다.

```bash
uv run python setup_api.py            # 대화형: 프로바이더 → 키 → 모델
uv run python setup_api.py --check    # 현재 설정 진단 (아무것도 고치지 않음)
```

키는 화면에 보이지 않게 입력받고 기록 전에 프로바이더의 models 엔드포인트로
검증합니다. **이 검증은 토큰을 쓰지 않으며** 오타난 키나 존재하지 않는 모델 ID를 첫
질문 전에 잡아냅니다. 기존 `.env`는 덮어쓰지 않고 줄 단위로 병합하므로 다른
프로바이더 설정과 주석이 그대로 남습니다.

| `LLM_PROVIDER` | 키 | 모델 | 비고 |
| --- | --- | --- | --- |
| `anthropic` (기본) | `ANTHROPIC_API_KEY` | `CLAUDE_MODEL` | 기본값 `claude-sonnet-5` |
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` | `OPENAI_BASE_URL`로 호환 게이트웨이 지정 |
| `openrouter` | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` | base URL 고정 |

`anthropic` 외에는 모델 기본값이 없습니다. 틀린 모델 ID를 기본값으로 박아두면 첫
요청에서야 실패하므로, 설정하지 않으면 시작할 때 바로 알려줍니다.

**키가 없어도 검색 기능은 전부 동작합니다.** CLI는 `--retrieval-only`, UI는 검색 전용
모드로 떨어집니다.

> blind adjudication은 `OPENAI_API_KEY`를 같이 쓰지만 모델은 `evaluation/cli.py`가
> 직접 지정합니다. `OPENAI_MODEL`을 바꿔도 동결된 평가 결과는 영향받지 않습니다.

### 3. 인덱스 만들기

기사 본문은 저장소에 없으므로 **수집부터 시작합니다.**

```bash
uv run python preprocessing/crawl.py    # 네이버 금융에서 수집 (20~30분)
uv run python preprocessing/clean.py    # 정제 → data/processed/news_data_clean.json
uv run python indexing/chunk.py
uv run python indexing/build_index.py --rebuild --batch-size 8 --device mps
```

맛보기로 먼저 돌려보려면 `crawl.py`에 `--stocks 005930 --pages 2 --per-stock 5`를
붙이세요. 수집이 중간에 끊겨도 `data/raw/crawl_cache/`에 종목별로 저장돼 이어받습니다
(처음부터 받으려면 그 폴더를 지우세요). 같은 로직을 노트북
(`preprocessing/crawler.ipynb`)에서도 쓸 수 있습니다 — 노트북은 이 스크립트를 불러
쓰기만 하므로 코드가 두 벌이 아닙니다.

`--device`는 `cpu` / `cuda`(NVIDIA) / `mps`(Apple Silicon). **CPU로는 1,377청크에 30분
이상 걸리니 GPU가 있으면 반드시 지정하세요.** BGE-M3 최초 다운로드도 큽니다.

인덱싱이 중단되면 `--rebuild` 없이 같은 명령을 다시 실행하면 됩니다. 청크 ID가
결정적이고 Chroma `upsert`를 쓰므로 중복은 안 생깁니다(다만 전체를 다시 임베딩합니다).

> ⚠️ **`chunk.py`와 `build_index.py`는 항상 함께 다시 실행하세요.** `chunk_id`가 본문
> 해시라서 정제·청킹 규칙을 바꾸면 값이 전부 달라지고 Chroma 인덱스와 청크 파일이
> 어긋나 RRF 조인이 조용히 실패합니다.

`data/processed/news_chunks.jsonl`과 `data/chroma/`는 재생성 가능한 대용량 산출물이라
Git에서 제외합니다.

종목 10개 × 목록 25페이지 × 최대 60건, 요청 간격 1초입니다.

> ⚠️ **지금 수집하면 2026년 8월 코퍼스와 같아지지 않습니다.** 기사가 수정·삭제되고 목록
> 순서도 바뀝니다. 아래 「실험 결과」의 숫자는 그때의 432기사·1,377청크에 대한 것이고,
> 새로 수집한 코퍼스로는 같은 값이 나오지 않습니다.

---

## 주요 커맨드

| 명령 | 하는 일 |
| --- | --- |
| `uv run streamlit run app.py` | 데모 UI |
| `uv run python cli.py "질문"` | 터미널에서 질의 |
| `uv run python cli.py --chat --gate` | 대화 모드 + 검색 게이트 |
| `uv run python cli.py "질문" --compare dense,hybrid` | 기법별 결과 나란히 비교 |
| `uv run python indexing/search.py "질문" --top-k 5` | Dense 검색만 |
| `uv run python preprocessing/crawl.py` | 뉴스 수집 (캐시 이어받기) |
| `uv run python -m evaluation.rebuild_pool` | 기사 URL에서 blind pool 복원 |
| `uv run python -m evaluation.final_evaluation` | 평가 지표 재계산 (API 호출 없음) |
| `uv run pytest` | 테스트 400개 (API 호출 없음) |

### CLI

Streamlit 사이드바 옵션이 그대로 플래그로 대응합니다.

```bash
uv run python cli.py "삼성전자 HBM 실적 전망은?"            # 기본: Hybrid
uv run python cli.py "005930 주가" --mode bm25              # BM25만
uv run python cli.py "HBM 수요" --mode dense --expand hyde  # Dense + HyDE
uv run python cli.py "코스피 시황" --retrieval-only         # 검색만, API 비용 0
uv run python cli.py "환율 전망" --json                     # 기계 판독용
```

| 플래그 | 값 (기본값) | 뜻 |
| --- | --- | --- |
| `--compare` | 조합 목록 | 비교할 조합을 직접 지정 (`--mode`/`--expand`보다 우선) |
| `--mode` | `dense` / `bm25` / **`hybrid`** | 검색 방식, 쉼표로 여러 개 |
| `--expand` | **`none`** / `multi_query` / `hyde` | 질의 확장, 쉼표로 여러 개 |
| `--gate` | 플래그 (꺼짐) | 검색이 필요한 질문인지 LLM이 먼저 판단 |
| `--chat` | 플래그 (꺼짐) | 이력을 기억하는 대화 모드 |
| `--retrieval-only` | 플래그 (꺼짐) | LLM 호출 안 함 — **API 키 없이 동작** |
| `--top-k` | 정수 (**5**) | 조합마다 가져올 청크 수 |
| `--candidate-k` | 정수 (**20**) | RRF 융합 전 각 검색기가 뽑는 후보 수 |
| `--rrf-k` | 정수 (**60**) | RRF 상수 `1/(k+rank)` — 키우면 순위 차이가 완만해짐 |
| `--device` | **`auto`** / `cpu` / `cuda` / `mps` | `auto`는 cuda → mps → cpu 순 탐지 |
| `--json` | 플래그 (꺼짐) | 조합별 `runs` 배열로 출력 |

출력은 `출처 → 구분선 → 답변` 순입니다. 검색이 먼저 끝나므로 **출처가 답변보다 먼저**
뜨고 답변은 스트리밍됩니다.

```
출처 2건
  [1] [칩톡]"살 빼야 출하된다"…AI 공룡들, 메모리 기근에 강제 'HBM...
      2026.08.08 08:00 | 삼성전자 · SK하이닉스
      RRF 0.0325 · dense #1 · bm25 #2 · 유사도 0.7140 · BM25 9.88 · 인용 [뉴스1], [뉴스2]
      https://finance.naver.com/item/news_read.naver?article_id=0005800438&...
────────────────────────────────────────────────────────────
삼성전자의 HBM 관련 실적 전망은 …[뉴스1][뉴스2].
```

`RRF 0.0325 · dense #1 · bm25 #2`는 어느 검색기가 몇 위로 올렸는지, `인용 [뉴스1]`은
답변 속 번호가 이 기사를 가리킨다는 뜻입니다.

거절되는 조합 둘: `--chat --retrieval-only`(답변이 없으면 남길 이력도 없음),
`--chat`에 조합 여러 개(어느 답을 이력에 남길지 정할 수 없음).

### 기법 비교

```bash
uv run python cli.py "HBM 수요 전망" \
  --compare dense,hybrid,dense+hyde,dense+multi_query --retrieval-only
```

```
변형별 순위 (· = 그 변형은 못 찾음)
dense  hybrid  dense+hyde  dense+multi_query  기사
    1       2           1                  2  [칩톡]"살 빼야 출하된다"…AI 공룡들…
    2       1           3                  1  삼전·하닉은 월가 'ATM'이었다?...
    ·       4           5                  5  엔비디아, HBM 공급부족 대응…
    ·       5           ·                  4  메모리 품귀에…엔비디아 HBM4E 용량 축소 검토
```

모든 조합이 찾은 기사가 위로 올라오므로, **아래쪽 `·`가 섞인 줄이 기법 간 차이가 나는
지점**입니다. `--mode dense,bm25,hybrid`처럼 쉼표 목록을 주면 모든 짝을 전수 비교합니다.
무거운 모델은 조합 사이에 공유되므로 BGE-M3는 몇 개를 비교하든 한 번만 올라옵니다.

> 확장(`hyde`, `multi_query`)은 조합마다 LLM을 한 번씩 더 부릅니다. `--retrieval-only`
> 로도 확장은 LLM을 쓰므로, 확장을 비교하려면 API 키가 필요합니다.

### 검색 동작 요약

**RRF** — 각 검색기에서 후보를 `candidate_k`(20)개씩 뽑고 청크마다
`sum(1 / (rrf_k + 순위))`를 더해 정렬합니다. 점수가 아니라 **순위**로 합치는 이유는
코사인 유사도(~0.6)와 BM25 점수(~12)가 비교할 수 없는 척도이기 때문입니다. 후보 풀은
`top_k`보다 넓어야 융합될 여지가 생깁니다.

**BM25 토크나이저** — 공백으로 자르면 질문의 `삼성전자`가 본문의 `삼성전자의`와 안
붙습니다. `kiwipiepy`로 조사를 떼고 명사·외국어(`SL`)·숫자(`SN`)·용언 어간을 남깁니다.
`SL`이 특히 중요합니다 — NAVER, SK, HBM, AI가 전부 여기 걸립니다. konlpy·mecab은 Java나
C 라이브러리가 필요해 `uv.lock` 재현성이 깨지므로 순수 wheel인 kiwipiepy를 씁니다.

**질의 확장** — Multi-Query는 질문을 뉴스 어휘로 3가지쯤 바꿔 쓰고, HyDE는 답이 될 법한
가상 기사 본문을 지어내 그것으로 검색합니다. **둘 다 원본 질문을 항상 첫 검색어로
넣고** LLM 호출이 실패하면 예외 없이 원본만으로 검색합니다 — 확장은 최적화이지 필수
경로가 아닙니다. HyDE가 지어낸 본문은 **검색에만** 쓰이고 답변 근거로는 안 들어갑니다.

**검색 게이트** — 턴마다 LLM이 `RETRIEVE` / `CHAT`을 판단하고 애매하면 검색 쪽으로
기웁니다. 유사도 임계값을 먼저 재봤지만 이 코퍼스에서는 두 부류가 안 갈립니다.

| 질의 | 유사도 |
| --- | --- |
| `오늘 서울 날씨 어때` (잡담) | **0.558** |
| `삼성바이오로직스 위탁생산` (정상) | 0.552 |
| `005930` (정상) | **0.428** |

잡담이 정상 질문보다 높고 종목코드 질의가 전체 최저입니다. BM25를 같이 봐도 최선
조합에서 잡담 6/14가 통과합니다. 스칼라 컷으로는 못 가릅니다.

> ⚠️ **후속 질문 한계** — 종목명이 있으면 잘 됩니다(`그럼 SK하이닉스는?` 0.629). 대명사만
> 있으면 실패합니다: `거기는 어때?` 0.481, `그 회사 실적은?` 0.585. 후속 질문을 독립형
> 검색어로 재작성하는 expander를 추가하면 해결됩니다.

---

## 참고

검색 기법 비교라는 문제 설정과 평가 축은 아래 논문을 참고했습니다. 다만 그쪽은
정답 라벨이 이미 있는 **분류** 과제라 정답이 공짜인 반면, 이 프로젝트는 개방형 QA라
관련성 판정을 직접 만들어야 했습니다. 그 차이가 평가셋 설계의 대부분을 결정했습니다.

> 김혜윤, 노연수, 박종혁, 박민정, 양병욱, 정윤서. (2025).
> 임베딩 모델 및 Advanced RAG 기법을 활용한 한국어 텍스트 분류.
> *응용통계연구*, 38(4), 571-588.

## 라이선스

코드는 [MIT](LICENSE). 뉴스 기사 본문은 저작권이 각 언론사에 있어 이 저장소에 포함하지
않습니다 — `preprocessing/crawl.py`로 각자 수집해 쓰세요.
