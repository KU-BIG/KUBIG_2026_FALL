# 사용자 맞춤 Steam 게임 추천 시스템

> **리뷰가 거의 없는 게임도, 사용자의 취향 공간 안에서는 후보가 될 수 있을까?**

**DL 4팀** · 23기 이서현, 23기 홍준기, 24기 고서연, 24기 이정용  
프로젝트 결과물은 Steam 게임 카탈로그, 유저-게임 상호작용, 텍스트·이미지·정형 피처를 결합해 기존 사용자와 신규 사용자 모두에게 Top-K 게임을 추천하는 end-to-end 파이프라인입니다.

[github.com/Hong-Junki/26_2_Contest](https://github.com/Hong-Junki/26_2_Contest)

---

## 프로젝트 요약

Steam 추천 문제를 단순 `is_recommended` 이진 분류가 아니라 **50,872개 전체 카탈로그에서 사용자가 볼 만한 게임을 고르는 Top-K retrieval 문제**로 재정의했다. 텍스트, 이미지, 정형 정보를 하나의 64차원 **Multimodal Game Embedding**으로 만들고, 기존 사용자는 플레이 이력으로, 신규 사용자는 관심 게임과 장르로 같은 64차원 공간의 user vector를 만든다. 최종 추천은 `dot product` 기반 콘텐츠 점수에 Popularity 또는 MF-BPR baseline을 z-score로 혼합해 산출한다.

| | 값 |
|---|---|
| 데이터 | Kaggle Game Recommendations on Steam: 게임 50,872개, 유저-게임 상호작용 41,154,794건, 유저 13,781,059명 |
| 문제 설정 | 이진 분류가 아닌 Top-K retrieval, 사용자별 chronological split |
| 핵심 모델 | MiniLM Text + CLIP Image + SVD Tabular → 64D Multimodal Game Tower |
| 사용자 표현 | Warm: positive 이력의 log1p(hours) 가중 평균, Cold: 관심 게임 + 장르 prototype pooling |
| 최종 retrieval | `score(u, i) = u^T v_i`, 이후 hybrid scoring과 MMR diversity reranking |
| 대표 평가 | MF+Multimodal balanced: Recall@10 0.8165, NDCG@10 0.5609, MRR 0.4898 |
| 데모 | Steam Picks UI: `steam-picks-kubig-demo.vercel.app` |

---

## 1. 문제 정의

### 왜 단순 정확도 문제가 아닌가

| 관찰 | 내용 | 추천 문제에서의 의미 |
|---|---|---|
| 유저 이력 희소성 | 유저 상호작용 수 중앙값이 1 | 많은 유저는 ID 임베딩 기반 협업필터링을 학습할 신호가 거의 없다 |
| 게임 노출 편중 | 게임 13,262개, 약 26%는 리뷰 0건 | 순수 CF에서는 리뷰가 없는 게임이 추천 후보로 올라오기 어렵다 |
| Positive 비율 편향 | `is_recommended=True` 비율이 85.8% | 이진 분류 정확도/AUC만으로는 "무엇을 보여줄지"에 답하기 어렵다 |
| 카탈로그 long-tail | 카탈로그의 55%가 상호작용 50건 이하 | 인기 게임 위주 추천은 쉽지만 발견성과 다양성이 사라진다 |
| 시간 변화 | 연도별 positive 비율 변동 | 시간 기반 split에서 단순 분류 모델은 기저율 변화에 취약하다 |

### 이 프로젝트의 질문

> "리뷰와 상호작용이 부족한 Steam 카탈로그에서도, 게임의 콘텐츠 정보와 사용자의 이력/의도를 같은 공간에 놓으면 개인화된 Top-K 추천을 만들 수 있는가?"

따라서 목표는 "좋아요/싫어요를 맞히는 분류기"가 아니라 **전체 Steam 카탈로그에서 사용자가 볼 만한 후보를 순위화하는 retrieval 시스템**이다.

---

## 2. 전체 파이프라인

```text
Steam Catalog
  games.csv · games_metadata.json · Steam header images
        |
        v
      Game Tower
Text Tower      Image Tower       Tabular Tower
MiniLM          CLIP ViT-B/32     SVD safe tabular
64D             64D               64D
        \          |             /
         \         |            /
          Concat 192D
              |
          Fusion MLP
              |
      Game Embedding 64D
              |
              v
      50,872 x 64 Game Bank
              ^
              |
          User Tower
  +-----------+----------------+
  |                            |
Warm input                  Cold input
Train positive history      liked games + genre prototypes
log1p(hours) weighting      intent pooling
  |                            |
Warm User Vector 64D        Cold User Vector 64D
  +-----------+----------------+
              |
        Dot Product Retrieval
              |
      Hybrid Scoring + MMR
              |
        Top-K Recommendation
```

이 구조에서 `Game Tower`는 모든 게임을 동일한 64차원 콘텐츠 공간에 올리고, `User Tower`는 기존 사용자와 신규 사용자의 서로 다른 입력을 같은 64차원 user vector로 변환한다. 즉 warm/cold는 별도 추천 모델이 아니라 **User Tower 입력 방식이 다른 동일 retrieval 경로**다.

### 랭킹 학습 관점

초기 접근은 `is_recommended` 예측이었다. 하지만 positive 비율이 85.8%인 데이터에서 이진 분류는 추천 후보를 고르는 문제와 어긋난다. 최종적으로는 아래와 같이 랭킹 문제로 고정했다.

```text
positive item: is_recommended=True
confidence: log1p(hours)
hard negative: is_recommended=False
random negative: 사용자가 관측하지 않은 게임
metric: Recall@K, NDCG@K, MRR, Coverage, Cold Recall@K
```

---

## 3. 핵심 방법론

### Game Tower: 3개 모달리티를 하나의 게임 벡터로

| 모달리티 | 입력 | 표현 방식 | 산출물 |
|---|---|---|---|
| Text | 게임 설명, 태그, 제목 | MiniLM 기반 텍스트 임베딩 | `text_data/emb_text_minilm.npy` |
| Image | Steam header image 50,864장 | CLIP ViT-B/32, squash resize | `Data_process/emb_clip_squash.csv` |
| Tabular | 정형 피처 19종 및 safe feature bank | SVD 64D | `tabular_embedding/emb_tabular_svd64.npy` |
| Fusion | Text/Image/Tabular 64D concat | Fusion MLP | `game_fusion/emb_game_concat_64.npy` |

최종 채택된 기본 game bank는 `game_fusion/emb_game_concat_64.npy`이다. shape은 `(50872, 64)`이며 L2 정규화된 64차원 게임 임베딩이다.

### Step 1, 3, 4의 구분

| 구분 | 설명 | 현재 해석 |
|---|---|---|
| Step 1 | frozen text/image/tabular feature를 projection 후 fusion | 실추천 평가에서 최종 채택된 frozen concat bank |
| Step 3 | projection/fusion 경로를 BPR-style pairwise objective로 조정 | synthetic interaction smoke test 산출물 |
| Step 4 | frozen encoder 출력 위에 residual adapter 추가 | synthetic interaction smoke test 산출물 |

Step 3/4 산출물은 코드 경로와 artifact는 존재하지만, 저장된 embedding은 실제 user-item interaction이 아닌 synthetic smoke-test interaction으로 생성되었다. 실제 추천 평가에서는 frozen concat bank가 선택되었고, Step 3/4는 같은 split과 같은 candidate set에서 재학습·재평가해야 한다.

### User Tower: ID보다 이력과 의도를 사용

| 사용자 유형 | 입력 | user vector 생성 |
|---|---|---|
| Warm user | Train 기간 positive 이력 | 이력 게임의 Game Embedding을 `log1p(hours)`로 가중 평균 |
| Cold user | 관심 게임, 선호 장르 | 관심 게임 embedding과 장르 prototype을 pooling |

두 경로 모두 같은 64차원 공간의 벡터를 만들기 때문에 retrieval은 동일하다.

```text
score(user, game) = user_vector dot game_vector
```

### Hybrid Scoring

Raw 점수는 scale이 다르기 때문에 전체 카탈로그 기준 z-score로 표준화한 뒤 혼합한다.

```text
final_score = w * z(content_score) + (1 - w) * z(baseline_score)
```

| 경로 | baseline | 의미 |
|---|---|---|
| Cold | train popularity | 신규 사용자의 입력이 얕을 때 안전망 |
| Warm | MF-BPR score | 기존 사용자의 협업 필터링 신호 |

UI에서는 `interest_weight`를 "내 취향"과 "새로운 발견" 사이의 스펙트럼 슬라이더로 노출한다.

---

## 4. 평가

### 평가 설정

| 항목 | 설정 |
|---|---|
| split | 사용자별 chronological 80/10/10 |
| cohort | 50,000명 |
| train interactions | 451,741 |
| train positives | 385,030 |
| test query | 약 9,984~9,992개 |
| chronology audit | train→validation, validation→test 순서 위반 0건 |
| 후보군 | sampled 1 positive + 99 negatives, full catalog 50,872개를 분리 평가 |

Warm과 Cold는 반드시 분리해서 본다. 하나의 평균 지표로 합치면 콘텐츠 표현이 cold item 노출에 기여했는지 확인할 수 없다.

### Game Bank + MF Hybrid 3-seed 결과

| 모델 | Recall@10 | NDCG@10 | MRR | Long-tail Recall@10 | Catalog Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Multimodal only | 0.8039 | 0.5185 | 0.4381 | **0.2744** | **0.1588** |
| MF only | 0.7960 | 0.5372 | 0.4656 | 0.0000 | 0.0974 |
| MF + Multimodal balanced | **0.8165** | **0.5609** | **0.4898** | 0.0341 | 0.1133 |

해석은 명확하다. MF는 평균 랭킹 성능이 강하지만 long-tail cold 노출이 구조적으로 약하다. Multimodal only는 평균 NDCG는 낮아도 long-tail recall과 coverage가 높다. Balanced hybrid는 평균 성능과 콘텐츠 기반 발견성을 절충한다.

### User Tower 평가

| User 표현 | 평가 | Recall@10 | NDCG@10 | 비고 |
|---|---|---:|---:|---|
| Simple mean | sampled 1+99 | 0.7680 | 0.4870 | 단순 이력 평균 |
| log1p(hours) weighted mean | sampled 1+99 | 0.7809 | 0.5036 | 현재 배포 기본값 |
| History MLP + BPR | sampled 1+99 | **0.8175** | **0.5359** | sampled 후보에서는 최고 |
| log1p(hours) weighted mean | full catalog | **0.0222** | **0.0110** | 전체 카탈로그에서는 MLP보다 안정적 |
| History MLP + BPR | full catalog | 0.0165 | 0.0074 | sampled 지표 대비 역전 |

sampled 평가만 보면 MLP가 좋아 보이지만, 전체 50,872개 카탈로그에서는 `log1p(hours)` weighted mean이 더 안정적이었다. 그래서 최종 warm content profile은 복잡한 MLP가 아니라 이력 가중 평균을 기본값으로 사용한다.

---

## 5. Steam Picks 데모

최종 추천 흐름은 `recommendation_mvp/app.py`와 배포 UI에서 확인할 수 있다.

```bash
streamlit run recommendation_mvp/app.py
```

PPT 기준 데모 URL:

```text
steam-picks-kubig-demo.vercel.app
```

UI 기능은 다음 흐름으로 구성된다.

| 단계 | 내용 |
|---|---|
| 신규/기존 분기 | 기존 user id가 있으면 warm 추천, 없으면 cold onboarding |
| 관심 게임 선택 | 장르별 인기 게임과 전체 카탈로그 검색으로 seed game 선택 |
| 장르 선택 | 장르 prototype을 cold user vector에 반영 |
| 스펙트럼 조절 | `interest_weight`와 MMR 다양성 강도 조절 |
| 추천 카드 | 추천 이유, 게임 정보, 좋아요/싫어요/담기 인터랙션 |

기본 MMR 설정은 relevance 0.65, diversity 0.35이며, 샘플 프로필에서 relevance 99.74%를 유지하면서 제목 중복 DLC pair를 제거했다.

---



## 6. 한계

| 한계 | 내용 |
|---|---|
| 관측 편향 | 리뷰를 남길 만큼 강한 감정이 있는 사용자만 관측된다 |
| 실제 Steam ID 아님 | Kaggle 내부 익명 user id이므로 실서비스에서는 계정 연동이 필요하다 |
| Cold Recall의 어려움 | full-catalog 기준 User Tower cold recall은 아직 0.0000으로 남아 있다 |
| 콘텐츠-인기 교란 | 이미지 품질, 제작 규모, 인기의 상관을 완전히 분리하지 못했다 |
| 모달리티 중복 | 헤더 이미지 안의 제목 텍스트가 text modality와 중복될 수 있다 |
| Step 3/4 artifact 주의 | 현재 저장된 Step 3/4 embedding은 synthetic smoke test 기반이므로 배포용으로 쓰지 않는다 |

설계 단계에서 `positive_ratio`, `user_reviews`, 후보 게임의 `hours`, 후보 게임의 `is_recommended`처럼 추론 시점에 알 수 없거나 미래 정보가 섞일 수 있는 피처는 제외했다. `hours`는 후보 게임 피처가 아니라 사용자의 과거 이력 가중치로만 사용한다.

---

## 7. 의의

1. **추천은 분류 정확도보다 순위 문제에 가깝다.** Positive 비율이 높은 데이터에서는 AUC보다 Recall@K, NDCG@K, Coverage가 더 직접적이다.
2. **콘텐츠 신호의 가치는 평균 정확도만으로 보이지 않는다.** Text/Image/Tabular는 특히 cold item, long-tail, catalog coverage에서 의미가 드러난다.
3. **sampled 평가는 과대평가될 수 있다.** 1 positive + 99 negatives에서 좋던 MLP가 full catalog에서 역전된 것이 대표적이다.
4. **복잡한 모델보다 맞는 inductive bias가 중요하다.** 이력 중앙값이 1인 데이터에서는 sequence 모델보다 이력 pooling이 더 안정적인 선택이었다.
5. **하이브리드는 목적에 따라 달라진다.** Top-10 적중만 원하면 MF가 강하고, 발견성과 coverage까지 보면 Multimodal 또는 balanced hybrid가 필요하다.

---

## 8. 차별점과 추후 과제

### 차별점

| 구분 | 내용 |
|---|---|
| 문제 재정의 | `is_recommended` 분류가 아니라 전체 Steam 카탈로그 Top-K retrieval로 접근 |
| Warm/Cold 통합 | 기존 사용자와 신규 사용자를 같은 64D multimodal space에서 처리 |
| 멀티모달 Game Tower | 텍스트, 이미지, 정형 피처를 결합해 리뷰가 부족한 게임도 후보화 |


### 추후 과제

- 관측 편향과 인기 편향을 줄이기 위해 IPS 또는 debiasing 평가를 도입한다.
- Steam 계정 연동을 통해 Kaggle 익명 ID가 아닌 실제 사용자 이력을 입력받는다.
- 추천 카드의 설명, 출시연도, 장르, 한국어/영어 소개를 보강해 "왜 추천됐는가"와 "무슨 게임인가"를 함께 보여준다.

---

## 폴더 구조

```text
DL 4팀/
  Data_process/                 # Steam 메타데이터 정제, 이미지 tower, CLIP image embedding
  text_data/                    # 텍스트 전처리, MiniLM embedding, text-only 추천 실험
  tabular_embedding/            # 정형 피처 schema, SVD tabular embedding
  game_fusion/                  # Text/Image/Tabular fusion Game Tower와 64D game bank
    downstream_evaluation/      # Game bank ablation, MF+multimodal hybrid 평가
  history_user_tower/           # warm/cold user profile, genre prototype, User Tower 평가
  mvp_recommendation/           # 추천 모델 공통 모듈, BPR, metric, reranking
  recommendation_mvp/           # Streamlit UI, 배포 데이터, 샘플 추천 결과
  final/                        # UI 연동용 최종 추천 API wrapper
  scripts/                      # 학습, 평가, 검증, 요약 스크립트
  outputs/                      # 중간/최종 실험 산출물
```

최종 사용 관점에서는 `final/engine.py`가 warm/cold 추천 API를 제공하고, `recommendation_mvp/app.py`가 데모 UI를 담당한다. 모델 관점에서는 `game_fusion/emb_game_concat_64.npy`가 현재 추천에 쓰이는 기본 Game Tower artifact이다.
