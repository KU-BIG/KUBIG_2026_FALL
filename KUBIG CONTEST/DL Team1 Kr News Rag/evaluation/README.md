# Dense vs Hybrid retrieval 파일럿 평가

이 평가는 50개의 source_seeded 질문으로 기존 Dense Retrieval과 Hybrid Retrieval을 비교합니다.
질문 제작용 source article은 50개지만 검색 corpus는 축소하지 않습니다. 두 시스템 모두 전체
432개 기사와 1,377개 청크에서 검색하며 기존 candidate_k, rrf_k, Chroma 설정을 그대로 사용합니다.

## 표본 선정과 배정

assign 명령은 retrieval 결과나 예상 성능을 보지 않고 고정 seed로 날짜 층화표본을 만듭니다.
작업자는 kahyun과 ryeowon이며 각 날짜층을 절반씩 맡아 서로 다른 기사 25개씩 작성합니다.

| 날짜 stratum | corpus 기사 수 | 표본 | kahyun | ryeowon |
|---|---:|---:|---:|---:|
| 2026-07-31~2026-08-05 | 65 | 10 | 5 | 5 |
| 2026-08-06 | 67 | 10 | 5 | 5 |
| 2026-08-07 | 195 | 18 | 9 | 9 |
| 2026-08-08 | 105 | 12 | 6 | 6 |

    uv run python -m evaluation.cli assign --seed 42 --output evaluation/pilot_assignment.json

각 stratum의 표본 뒤에는 같은 seed로 섞인 나머지 기사 전체가 reserve 순서로 저장됩니다.
교체는 incomplete_article, question_not_possible, near_duplicate_event 중 하나의 사유만 허용하며,
동일 stratum의 다음 reserve를 사용하고 사유를 기록해야 합니다. 교체 판단에 Dense나 Hybrid 결과를
사용하지 않습니다.

질문 유형 총량은 exact_token 13, abstract 13, multi_aspect 12, factoid 12입니다.
날짜별 배분은 각각 3/3/2/2, 3/2/3/2, 4/5/4/5, 3/3/3/3입니다. 기사 내용상 배정 유형이
불가능하면 같은 날짜 stratum 안에서 다른 기사와 유형을 교환해 전체 quota를 유지합니다. 유형은
검색 조건을 다양화하기 위한 층화 변수이며 특정 시스템에 유리한 질문을 만들기 위한 기준이 아닙니다.

## 질문 작성과 self-check

annotations/kahyun_25.jsonl과 annotations/ryeowon_25.jsonl은 각각 kahyun과 ryeowon만 수정하는
실제 작성 파일입니다. 두 파일은 질문·placeholder·예시 없이 빈 상태로 시작합니다. query_id는
각각 K001–K025와 R001–R025를 쓰고 author는 파일 소유자와 일치해야 합니다.
pilot_retrieval_eval.example.jsonl은 명백한 placeholder 한 줄만 가진 작성 예시이고 CLI가 자동으로
읽지 않습니다. 기존 query-first 및 pooling 기능은 호환을 위해 보존하지만 이번 50문항 제작에는
query-first를 사용하지 않습니다.

AI 초안을 받을 수 있지만 작성자가 source 기사와 대조해 질문을 확정해야 합니다.

    uv run python -m evaluation.cli generate-question 123 --category abstract --query-id pilot_001 --author annotator_a

모든 레코드는 construction_method를 source_seeded, review_mode를 ai_assisted_self_check로 사용하고
seed_article_id를 initial gold로 gold_article_ids에 포함합니다. reviewer는 null이어도 괜찮으며
가짜 교차검수자를 적지 않습니다. approved가 되려면 작성자가 다음 self_check를 모두 true로 확인합니다.

- answer_supported_by_source
- natural_question
- not_title_copy
- not_duplicate
- source_article_id_verified

AI-assisted generation과 single-annotator self-check에는 질문·gold 편향을 놓칠 수 있다는 한계가
있습니다. 교차검수를 수행한 것처럼 해석해서는 안 됩니다.

## blind pooling과 추가 gold

질문과 initial gold를 확정한 뒤에는 질문 문구를 바꾸지 않습니다. 다음 명령은 전체 corpus에서
Dense와 Hybrid를 실행하고 article_id로 합친 후보를 출력합니다. 출력에는 어느 시스템이
반환했는지와 원래 순위가 포함되지 않습니다.

Freeze된 50문항은 `pool-batch`로 일괄 처리합니다. 이 명령은 AI 판정용 blind packet, 비공개
mapping, 실행 manifest를 분리해 원자적으로 기록합니다. 다음 AI 관련성 판정 단계에서는 blind
packet만 사용하고 mapping 파일은 열지 않습니다. Mapping은 판정 완료 후 candidate key를 실제
article ID와 시스템별 최초 기사 순위에 연결할 때만 사용합니다.

    uv run python -m evaluation.cli pool-batch evaluation/retrieval_eval_50.jsonl \
        --expected-freeze-sha256 <frozen-file-sha256>

    uv run python -m evaluation.cli pool "확정한 질문"
    uv run python -m evaluation.cli judge "확정한 질문"

Freeze된 blind packet의 AI 관련성 판정은 OpenAI Responses API 기반 `judge-blind`로 실행합니다.
`.env`에 `OPENAI_API_KEY=발급받은_전체_키` 한 줄을 넣으면 기본 `gpt-5.6-luna` 모델을
사용합니다. 모델을 바꿔야 할 때만 `OPENAI_MODEL`을 설정합니다. 이 명령은 mapping
경로를 받지 않으며 blind packet의 질문·category·불투명 candidate key·기사 표시 필드만
OpenAI에 전달합니다. 기사 본문은 신뢰할 수 없는 데이터 구분자로 감싸며 내부 명령은 따르지
않습니다. temperature 0은 완전 결정론을 보장하지 않으므로 prompt hash, 실행 시각, model ID와
출력을 manifest에 보존합니다.

    uv run python -m evaluation.cli judge-blind evaluation/pools/blind_pool_50.jsonl \
        --expected-sha256 <blind-packet-sha256>

각 API 요청은 질문 하나와 후보 기사 하나만 포함하며 `candidate_key`는 모델에 보내지 않고
로컬에서 결과에 결합합니다. 요청 모델은 정확히 `gpt-5.6-luna`, reasoning effort는 `low`,
최대 동시 호출 수는 4입니다. pass1과 review가 공유하는 총 예상 비용 상한은 USD 1.20이며,
상한을 넘길 수 있는 새 요청은 호출 전에 중단합니다. pass1과 review checkpoint는 별도 원자적
파일로 저장되어 재개 시 이미 성공한 후보를 다시 호출하지 않습니다.

중단 시 `evaluation/.checkpoints/blind_adjudication`에서 성공한 판정만 재개합니다. 이 경로는
Git에서 제외되며 최종 judgment 산출물이 아닙니다. 독립 재검토가 끝날 때까지 mapping 파일을
열지 않으며, 관련성 판정 단계에서는 `evaluation/judgments`의 결과만 사용합니다.

AI 판정은 relevant, not_relevant, uncertain과 근거 문장을 반환합니다. relevant 기사는 작성자가
확인한 뒤 gold_article_ids와 evidence에 추가할 수 있고, not_relevant는 추가하지 않습니다.
uncertain은 작성자가 해결하기 전에는 final validation과 평가 실행이 거부됩니다. AI 출력은 평가
JSONL을 자동 수정하지 않습니다.

이 방식은 source 기사 하나만 정답으로 강제하는 known-item retrieval과 다릅니다. Source는 initial
gold일 뿐이며, pooled 후보 중 질문에 충분히 답하는 다른 기사도 gold가 되는 article-level relevance
평가입니다. 다른 gold를 찾기 위해 432개 전체를 사람이 읽지는 않습니다.

## 검증과 실행

작성 중 draft는 명시적으로 허용해 검증합니다.

    uv run python -m evaluation.cli validate evaluation/pilot_retrieval_eval.jsonl --allow-draft

최종 실행 전에는 모든 문항이 approved이고 uncertain이 없어야 합니다.

    uv run python -m evaluation.cli validate evaluation/pilot_retrieval_eval.jsonl
    uv run python -m evaluation.cli run evaluation/pilot_retrieval_eval.jsonl --json-output evaluation/results/pilot.json --csv-output evaluation/results/pilot.csv

실제 평가 파일이 비어 있으면 retriever를 로드하거나 결과 파일을 만들지 않습니다. 결과에는 실행 시각,
Git commit, 질문 파일 SHA-256, 전체 corpus 크기, source article 수, retrieval 설정, raw chunk 순위와
중복 제거한 article 순위가 기록됩니다. API key나 로컬 캐시 경로는 저장하지 않습니다.

지표는 기사 단위 Hit@1/3/5와 MRR@5이며 전체, 날짜 stratum별, 질문 유형별로 출력됩니다. 날짜별
요약에는 해당 stratum의 corpus 기사 수와 표본 수도 포함됩니다. 날짜별 표본이 작으므로 날짜 차이는
탐색적으로만 해석하며 통계적 우월성을 주장하지 않습니다.

## 실제 문항 작성 순서

1. 두 작업자 모두 작업 시작 전에 현재 프레임워크 브랜치를 pull합니다.
2. assign으로 고정 배정표와 reserve를 생성합니다. 기사 선정과 담당자 확정은 질문 작성과 별도입니다.
3. kahyun은 kahyun_25.jsonl만, ryeowon은 ryeowon_25.jsonl만 수정합니다.
4. 일부 문항만 작성한 상태에서도 각 파일을 --allow-draft로 검증하고 commit·push할 수 있습니다.
5. 상대방이 먼저 push했다면 자신의 push 전에 최신 변경을 반영합니다.
6. 작성자는 배정된 source를 읽고 AI 초안을 참고해 질문과 initial gold를 확정한 뒤 self-check합니다.
7. 질문을 동결하고 blind pool을 판정해 relevant 후보를 추가 gold로 반영하고 uncertain을 해결합니다.
8. 두 파일이 각각 25개로 완성되면 아래 한 줄 명령으로 최종 파일을 만듭니다.

    uv run python -m evaluation.cli combine evaluation/annotations/kahyun_25.jsonl evaluation/annotations/ryeowon_25.jsonl --output evaluation/retrieval_eval_50.jsonl

combine은 입력 파일을 수정하지 않고 25/25개, author, query_id 범위, 중복 query/source/question,
approved 상태와 self-check를 모두 검사합니다. 실패 시 기존 출력 파일을 덮어쓰지 않으며 성공 시
K001–K025, R001–R025 순서로 기록합니다.

9. 최종 파일을 validate한 후 전체 432개 기사 corpus에서 Dense/Hybrid 평가를 실행합니다.

개발 중 개별 검증 예:

    uv run python -m evaluation.cli validate evaluation/annotations/kahyun_25.jsonl --allow-draft
