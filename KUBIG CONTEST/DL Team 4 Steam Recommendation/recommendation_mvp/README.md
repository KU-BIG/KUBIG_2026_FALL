# Steam 멀티모달 추천 MVP

팀의 Text, Image, Tabular 결과를 실제 사용자 interaction과 결합해 Steam 게임 Top-K를
추천하는 Streamlit 서비스입니다. 이 폴더만 처음 보는 팀원도 실행과 결과 해석을 할 수 있도록
현재 배포 기준을 정리했습니다.

## 1분 요약

| 항목 | 현재 상태 |
|---|---|
| 게임 카탈로그 | Kaggle 기준 50,872개, key는 `app_id` |
| 기존 사용자 | Positive train history가 있는 49,742명 |
| 최종 known-user 모델 | **MF 40% + Multimodal 60%** |
| Multimodal 구성 | MiniLM Text + CLIP Image + SVD Tabular → 64D |
| 신규 사용자 | 좋아하는 게임은 Multimodal, 태그는 Text, 무입력은 Popularity |
| Top-K 후처리 | MMR 다양성 reranking, UI 기본 ON |
| 화면 | Steam 이미지가 포함된 2열 카드, 표, CSV 다운로드 |
| 배포 진입점 | `recommendation_mvp/app.py` |

## 추천 흐름

```text
기존 사용자 ID
  ├─ MF-BPR 협업 점수 ───────────────┐
  └─ Multimodal-BPR 콘텐츠 점수 ─────┤─ 0.4 : 0.6 결합
                                      └─ 관측 게임 제외 → MMR → Top-K

신규 사용자
  ├─ 좋아하는 게임 있음 → 게임들의 Multimodal 64D 평균
  ├─ 선호 태그 있음      → MiniLM Text 태그 프로필
  └─ 입력 없음           → Train positive popularity
                                      └─ MMR → Top-K
```

Text·Image·Tabular encoder는 추천 시 다시 학습하지 않습니다. 미리 만들어진 game embedding은
고정하고, 실제 train interaction으로 학습한 user embedding과 내적해 점수를 계산합니다.

## 모델을 이해하기 위한 기본 개념

### MF: Matrix Factorization

MF는 **Matrix Factorization(행렬 분해)**의 약자입니다. 사용자–게임 interaction 행렬을
직접 저장하는 대신 사용자와 게임을 각각 작은 잠재 벡터로 표현합니다. 이 프로젝트에서는
각각 64차원이며, 두 벡터의 내적이 클수록 해당 사용자가 게임을 선호한다고 봅니다.

```text
user embedding (64D) · game embedding (64D) = 추천 점수
```

MF는 게임 설명이나 이미지를 읽지 않고 “이 게임을 좋아한 사용자들이 다른 어떤 게임을 함께
좋아했는가?”라는 협업 패턴을 학습합니다. 희소한 Steam interaction에서 강한 baseline이고
전체 카탈로그 점수를 빠르게 계산할 수 있어 사용했습니다. 반면 interaction이 없는 신규 사용자와
신규 게임에는 약하고 인기 게임 쪽으로 편향될 수 있습니다.

### BPR: Bayesian Personalized Ranking

BPR은 **Bayesian Personalized Ranking(베이지안 개인화 순위 학습)**의 약자입니다. 정확한
평점이나 좋아할 확률을 예측하는 대신 사용자가 긍정적으로 평가한 게임을 관측하지 않은 게임보다
위에 놓도록 학습합니다.

```text
사용자 u가 좋아한 게임 i, 관측하지 않은 게임 j
학습 목표: score(u, i) > score(u, j)
```

개념적인 loss는 다음과 같습니다.

```python
loss = -log(sigmoid(score_positive - score_negative))
```

우리의 목적은 별점 예측이 아니라 Top-K 추천 순서를 잘 만드는 것이므로 Recall@K, NDCG@K와
목표가 잘 맞는 BPR을 사용했습니다.

### MF와 BPR의 관계

둘은 서로 경쟁하는 모델명이 아니라 역할이 다릅니다.

```text
MF  = 사용자와 게임을 어떤 벡터로 표현할 것인가
BPR = positive 게임이 negative 게임보다 위에 오도록 어떻게 학습할 것인가
```

따라서 `mf_bpr`은 사용자 ID와 게임 ID에서 각각 학습 가능한 embedding을 만들고, 그 embedding을
BPR loss로 학습한 모델입니다.

### MF-BPR과 Multimodal-BPR의 차이

| 구분 | MF-BPR | Multimodal-BPR |
|---|---|---|
| 사용자 표현 | 학습 가능한 64D user embedding | 학습 가능한 64D user embedding |
| 게임 표현 | 게임 ID별 학습 가능한 embedding | Text + Image + Tabular로 미리 만든 고정 64D embedding |
| 강점 | 사용자 행동·협업 패턴 | 게임 자체 특성·long-tail 탐색 |
| 약점 | 신규 게임, 인기 편향 | 협업 신호를 직접 표현하지 못함 |

`mf_multimodal_hybrid`은 두 모델의 점수를 결합해 협업 패턴과 게임 자체 정보를 함께 사용합니다.
현재 validation에서는 MF 40%, Multimodal 60%가 선택됐습니다.

### Negative sampling 해석 주의

BPR 학습에서는 사용자가 긍정적으로 평가한 게임을 positive로 두고, 그 사용자의 train 이력에서
관측되지 않은 게임을 negative 후보로 표본 추출합니다. 여기서 negative는 “싫어한 게임”이 아니라
**아직 관측되지 않은 게임**입니다. 실제로는 사용자가 좋아하지만 아직 플레이하지 않은 게임도
negative로 뽑힐 수 있으므로 추천 점수를 절대적인 선호 확률로 해석하면 안 됩니다.

한 문장으로 정리하면 다음과 같습니다.

```text
MF = 사람들이 무엇을 함께 좋아했는가
BPR = 좋아한 게임을 추천 목록 위쪽에 놓는 학습 방법
Multimodal = 게임 내용·이미지·정형 특성이 얼마나 잘 맞는가
Hybrid = 행동 정보와 게임 자체 정보를 함께 사용
```

## 가장 빠른 실행

repository 루트에서 실행합니다. `recommendation_mvp` 폴더 안으로 이동할 필요가 없습니다.

```bash
pip install -r recommendation_mvp/requirements.txt
streamlit run recommendation_mvp/app.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시된 `http://localhost:8501`로 접속합니다.

### Streamlit Cloud

| 설정 | 값 |
|---|---|
| Repository | `Hong-Junki/26_2_Contest` |
| Branch | `main` |
| Main file path | `recommendation_mvp/app.py` |
| Python | **3.12 권장** |

GitHub 변경이 화면에 반영되지 않거나 새 함수의 인자를 인식하지 못하는 `TypeError`가 발생하면
**Manage app → Reboot app**을 실행합니다. 이는 이전 Python module이 메모리에 남아 있을 때
발생할 수 있습니다.

## UI 사용법

### 기존 사용자

1. 사이드바에서 `기존 사용자`를 선택합니다.
2. 학습 데이터에 존재하는 사용자 ID를 입력합니다.
3. 기본 모델인 `MF + Multimodal Hybrid (recommended)`를 사용합니다.
4. 필요하면 다른 모델도 함께 선택해 모델별 탭에서 결과를 비교합니다.

### 신규 사용자

1. 선호 태그와 좋아하는 게임을 입력합니다. 둘 중 하나만 입력해도 됩니다.
2. 좋아하는 게임은 최대 5개까지 선택할 수 있습니다.
3. 아무것도 입력하지 않으면 Train 데이터의 인기 게임을 추천합니다.

추천 결과는 2열 카드로 표시됩니다. 카드에는 이미지, 순위, 게임명, Steam 평가, 긍정 비율,
가격과 추천 이유가 있습니다. 게임명을 누르면 Steam Store가 새 탭에서 열립니다.

- 최신 Steam 이미지 URL 실패 → legacy CDN URL 재시도
- 두 이미지 URL 모두 실패 → placeholder 표시
- `표 형태로 보기` → 기존 테이블 확인
- `CSV 다운로드` → 전체 추천 결과 저장

## 제공 모델

| UI/코드 이름 | 구성 | 용도 |
|---|---|---|
| `mf_multimodal_hybrid` | MF 40% + Multimodal 60% | **기본 추천 모델** |
| `multimodal_bpr` | Text + Image + Tabular 고정 game bank | 콘텐츠·long-tail 탐색 |
| `balanced_hybrid` | MF + Text | 이전 baseline 비교 |
| `mf_bpr` | 사용자 ID + 게임 ID 협업 필터링 | 인기·협업 신호 비교 |
| `text_bpr` | MiniLM 기반 Text 콘텐츠 | Text-only 비교 |

서로 다른 모델의 원점수 범위가 다르기 때문에 사용자별 추천 가능 카탈로그 안에서 z-score로
표준화한 후 hybrid weight를 적용합니다. 최종 `0.4 / 0.6` 가중치는 test가 아니라 validation
데이터에서 선택했습니다.

## 오프라인 성능

동일한 9,984개 test query, query당 positive 1개 + negative 99개, seed 42/7/2026의
평균 ± 표본표준편차입니다.

| 모델 | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| **MF + Multimodal** | **0.8165 ± 0.0014** | **0.5609 ± 0.0006** | **0.4898 ± 0.0012** | 0.0341 ± 0.0041 | 0.1133 ± 0.0002 |
| Multimodal only | 0.8039 ± 0.0013 | 0.5185 ± 0.0008 | 0.4381 ± 0.0008 | **0.2744 ± 0.0031** | **0.1588 ± 0.0007** |
| 기존 MF + Text | 0.7906 ± 0.0086 | 0.5404 ± 0.0032 | 0.4713 ± 0.0034 | 0.0279 ± 0.0198 | 0.1020 ± 0.0025 |
| MF only | 0.7960 ± 0.0007 | 0.5372 ± 0.0014 | 0.4656 ± 0.0016 | 0.0000 ± 0.0000 | 0.0974 ± 0.0003 |

MF + Multimodal은 기존 MF + Text보다 Recall@10 `+0.0259`, NDCG@10 `+0.0205`였습니다.
상세 평가표는 [`../game_fusion/downstream_evaluation/`](../game_fusion/downstream_evaluation/)에
있습니다.

## 어떤 fusion 파일을 사용하나?

최종 서비스는 다음 파일을 사용합니다.

```text
game_fusion/emb_game_concat_64.npy
game_fusion/emb_game_concat_64.csv
```

`emb_game_finetuned_64`와 `emb_game_partial_fusion_tuned_64`는 구조 실행을 확인하기 위해
synthetic interaction으로 만든 smoke-test 산출물입니다. 실제 interaction 평가에서 frozen
concat보다 성능이 크게 낮았으므로 **서비스에 사용하지 않습니다.**

## 핵심 파일

| 경로 | 역할 |
|---|---|
| `app.py` | Streamlit 화면과 이미지 카드 |
| `config.json` | 기본 모델·artifact 설정 |
| `deploy_data/` | UI용 catalog, 관측 이력, Train popularity |
| `model_artifacts/` | 실제 interaction으로 학습한 Multimodal user tower |
| `../mvp_recommendation/inference.py` | 기존 사용자 full-catalog scoring |
| `../mvp_recommendation/cold_start.py` | 신규 사용자 프로필과 추천 |
| `../mvp_recommendation/reranking.py` | MMR 다양성 reranking |
| `../scripts/evaluate_multimodal_game_bpr.py` | 실제 interaction downstream 평가 |
| `../scripts/validate_multimodal_pipeline.py` | 배포 artifact와 추천 경로 검증 |

대용량 `.pt`와 일부 `.npy`는 Git LFS로 관리합니다. clone 후 파일 내용이 LFS pointer로만
보이면 `git lfs pull`을 실행합니다.

## CLI 사용

기존 사용자:

```bash
python scripts/recommend_users.py \
  --user-ids 13 \
  --models mf_multimodal_hybrid multimodal_bpr \
  --top-k 10 \
  --output outputs/user_13_multimodal.csv
```

신규 사용자:

```bash
python scripts/recommend_new_user.py \
  --profile-name witcher_fan \
  --preferred-tags RPG "Open World" \
  --liked-app-ids 292030 \
  --top-k 10 \
  --output outputs/witcher_fan.csv
```

각 실행은 CSV와 함께 실행 조건을 담은 `.manifest.json`을 생성합니다.

## 주요 출력 컬럼

| 컬럼 | 설명 |
|---|---|
| `app_id`, `title` | Steam 게임 ID와 게임명 |
| `rank`, `score` | 해당 사용자·모델 내 순위와 점수 |
| `model` | 추천에 사용한 모델 |
| `mf_score_z` | 표준화된 협업 점수 |
| `multimodal_score_z` | 표준화된 멀티모달 점수 |
| `recommendation_reason` | 사용한 추천 신호를 요약한 설명 |
| `alpha_mf`, `alpha_multimodal` | hybrid 결합 가중치 |
| `excluded_history_scope` | 추천에서 제외한 interaction 범위 |

`recommendation_reason`은 모델 신호를 읽기 쉽게 요약한 휴리스틱이며 인과적 XAI 설명은
아닙니다.

## 검증 명령

```bash
python scripts/validate_multimodal_pipeline.py
python scripts/validate_recommendation_pipeline.py
python scripts/validate_cold_start_pipeline.py
python scripts/validate_diversity_reranking.py
```

모든 검증이 `*_OK`로 끝나야 합니다.

## 해석 시 주의사항과 다음 단계

- 현재 평가는 sampled candidate 기반이며 full-catalog offline 평가는 아직 남아 있습니다.
- 실제 만족도는 클릭·찜·플레이 로그 또는 사용자 설문으로 검증해야 합니다.
- 신규 사용자 태그-only 경로에는 이미지 정보가 없으므로 Text 중심으로 동작합니다.
- 이미지가 없는 소수 게임은 fusion 제작 단계의 대체 벡터 정책을 따릅니다.
- interaction 또는 game bank를 다시 만들면 validation에서 hybrid weight도 다시 선택해야 합니다.
- `.npy`와 대응 `.csv`의 `app_id` 행 순서는 절대 따로 변경하지 않습니다.
