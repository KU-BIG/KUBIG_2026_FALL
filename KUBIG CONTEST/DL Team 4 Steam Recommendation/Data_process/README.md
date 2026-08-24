# Steam 게임 추천용 통합 데이터

Kaggle Steam Games 데이터를 기준으로 게임 목록과 기존 컬럼을 그대로 유지하면서, Kaggle metadata의 게임 설명과 Hugging Face FronkonGames의 설명·장르·언어·평가·플레이타임 정보를 `app_id` 기준으로 보완한 데이터입니다.

## 빠른 요약

| 항목 | 결과 |
|---|---:|
| 최종 데이터 크기 | 50,872행 × 42열 |
| 기준 데이터 | Kaggle `games.csv` |
| Kaggle 원본 컬럼 | 13개, 순서와 값 보존 |
| Hugging Face ID 매칭 | 39,436개 (77.52%) |
| 원래 description 결측 | 10,374개 (20.39%) |
| HF 설명으로 보완 | 839개 |
| 최종 description 결측 | 9,535개 (18.74%) |

## 파일 안내

| 파일 | 용도 |
|---|---|
| `games_metadata_enriched.parquet` | **분석 권장 파일.** nullable 타입과 list 구조를 비교적 잘 보존하고 로딩이 빠름 |
| `games_metadata_enriched.csv` | 범용 공유·확인용 최종 데이터. Git LFS로 관리 |
| `games_description_still_missing.csv` | 보완 후에도 최종 설명이 없는 게임만 추출 |
| `steam_metadata_enrichment.py` | 전체 전처리 재실행용 Python 스크립트 |
| `steam_metadata_enrichment.ipynb` | 셀 단위 실행용 Jupyter Notebook |

Git LFS가 설치된 환경에서 clone 후 다음 명령으로 대용량 데이터를 받습니다.

```bash
git lfs install
git lfs pull
```

Python에서는 Parquet 사용을 권장합니다.

```python
import pandas as pd

df = pd.read_parquet("Data_process/games_metadata_enriched.parquet")
print(df.shape)  # (50872, 42)
```

## 왜 Hugging Face 컬럼을 추가했나?

Kaggle `games.csv`는 게임 목록, 가격, 플랫폼, 평가 요약에는 강하지만 추천에 필요한 콘텐츠·이용 행태 정보는 제한적입니다.

- 설명, 태그, 장르, 카테고리: 콘텐츠 기반 추천과 게임 간 의미적 유사도 계산
- 개발사, 퍼블리셔: 제작사 선호 및 게임군 분석
- 지원 언어: 사용자 언어 및 지역화 조건을 반영한 추천
- 긍정·부정 평가: 선호도와 대중 반응을 보여주는 보조 신호
- 플레이타임, 소유자 범위, Peak CCU: 실제 참여도와 대중성 분석
- Metacritic: 희소하지만 외부 품질 신호로 선택적 활용

Kaggle과 HF 값은 수집 시점과 정의가 다를 수 있으므로 기존 값을 덮어쓰지 않고 `_hf` 접미사를 붙여 별도로 보존했습니다.

## 왜 이 구조를 사용했나?

1. **Kaggle 기준 left join**: 추천 대상 게임이 바뀌지 않도록 Kaggle의 50,872개 `app_id`만 유지했습니다. HF에만 있는 게임은 추가하지 않았습니다.
2. **원본 구조 보존**: Kaggle `games.csv`의 13개 컬럼을 최종 데이터 앞쪽에 같은 순서와 값으로 유지해 기존 분석 코드와 호환됩니다.
3. **원문과 모델 입력 분리**: `description_original`, `description_final`, `description_clean`을 나누어 원문 감사와 모델 활용을 모두 지원합니다.
4. **0과 결측 구분**: `hf_matched`와 `has_*` 플래그를 두어 실제 값 0과 HF에서 알 수 없는 값을 구분합니다.
5. **두 가지 저장 형식**: CSV는 범용 공유용, Parquet은 타입·용량·속도를 고려한 분석용입니다.

## description 생성 규칙

`description_final`은 다음 순서로 처음 존재하는 값을 선택합니다.

1. Kaggle `games_metadata.json`의 `description`
2. HF `short_description`
3. HF `About the game`을 매핑한 `description_hf_detailed`
4. 모두 없으면 결측

현재 HF 원본에는 `short_description` 컬럼이 없으므로 `description_hf_short`는 전부 결측입니다. 기존 Kaggle 설명은 HF 설명으로 덮어쓰지 않았습니다.

`description_clean`에는 HTML entity decode, HTML tag 제거, 줄바꿈 및 반복 공백 정리만 적용했습니다. 소문자화, 숫자 삭제, stemming, lemmatization은 하지 않았습니다.

## 컬럼 사전

### Kaggle 원본 컬럼

| 컬럼 | 설명 |
|---|---|
| `app_id` | Steam 애플리케이션 고유 ID이자 병합 키 |
| `title` | 게임명 |
| `date_release` | 출시일 (`YYYY-MM-DD`) |
| `win`, `mac`, `linux` | 각 운영체제 지원 여부 |
| `rating` | Steam 사용자 평가 범주 |
| `positive_ratio` | 긍정 평가 비율, 0~100 |
| `user_reviews` | 사용자 리뷰 수 |
| `price_final` | 할인 적용 후 가격 |
| `price_original` | 할인 전 가격 |
| `discount` | 할인율 |
| `steam_deck` | Steam Deck 지원/호환 여부 |

### 설명 및 콘텐츠 metadata

| 컬럼 | 설명 |
|---|---|
| `description_original` | Kaggle metadata의 기존 설명 |
| `description_hf_short` | HF 짧은 설명 후보. 현재 원본에는 대응 컬럼이 없어 전부 결측 |
| `description_hf_detailed` | HF `About the game` 원문 |
| `description_final` | Kaggle → HF short → HF detailed 우선순위로 선택한 최종 원문 |
| `description_clean` | HTML과 반복 공백만 보수적으로 정리한 모델 입력용 텍스트 |
| `description_source` | `kaggle`, `huggingface_short`, `huggingface_detailed`, `missing` 중 하나 |
| `tags_kaggle` | Kaggle 태그 목록 원본 |
| `tags_huggingface` | HF 태그 원본 문자열 |
| `genres` | HF 장르 문자열 |
| `categories` | HF Steam 기능·카테고리 문자열 |

### HF 분석 컬럼

| 컬럼 | 설명 |
|---|---|
| `hf_matched` | 해당 `app_id`가 HF에 존재하는지 여부 |
| `developers_hf`, `publishers_hf` | 개발사와 퍼블리셔 |
| `supported_languages_hf` | 지원 텍스트 언어. 빈 목록은 결측 처리 |
| `full_audio_languages_hf` | 전체 음성 지원 언어. 빈 목록은 결측 처리 |
| `positive_hf`, `negative_hf` | HF 수집 시점의 긍정·부정 평가 수 |
| `achievements_hf` | 도전 과제 수 |
| `recommendations_hf` | HF의 추천/리뷰 관련 집계 수 |
| `average_playtime_forever_hf` | 누적 평균 플레이타임, 단위는 분 |
| `median_playtime_forever_hf` | 누적 플레이타임 중앙값, 단위는 분 |
| `metacritic_score_hf` | Metacritic 점수. 원본의 미수집 표기 0은 결측 처리 |
| `estimated_owners_hf` | 추정 소유자 수 범위 문자열(예: `0 - 20000`) |
| `peak_ccu_hf` | 최고 동시 접속자 수 |
| `required_age_hf` | 요구 연령. 0은 제한 없음과 미기재가 섞일 수 있음 |
| `dlc_count_hf` | DLC 수 |
| `has_language_info_hf` | 지원 언어 정보 존재 여부 |
| `has_playtime_hf` | 평균 누적 플레이타임이 0보다 큰지 여부 |
| `has_metacritic_hf` | 유효한 Metacritic 점수 존재 여부 |

## HF 컬럼 결측 현황

전체 결측률에는 HF 미매칭 11,436개(22.48%)가 포함됩니다. **매칭 내부 결측률**은 HF와 ID가 매칭된 39,436개 게임 중에서 해당 정보가 실제로 없는 비율입니다.

| 컬럼 | 전체 결측률 | HF 매칭 내부 결측률 |
|---|---:|---:|
| `description_hf_short` | 100.00% | 100.00% |
| `description_hf_detailed` | 22.61% | 0.16% |
| `tags_huggingface` | 26.83% | 5.62% |
| `genres` | 22.62% | 0.17% |
| `categories` | 23.07% | 0.76% |
| `developers_hf` | 22.67% | 0.24% |
| `publishers_hf` | 23.17% | 0.88% |
| `supported_languages_hf` | 22.50% | 0.03% |
| `full_audio_languages_hf` | 63.28% | 52.63% |
| `positive_hf` | 22.48% | 0.00% |
| `negative_hf` | 22.48% | 0.00% |
| `achievements_hf` | 22.48% | 0.00% |
| `recommendations_hf` | 22.48% | 0.00% |
| `average_playtime_forever_hf` | 22.48% | 0.00% |
| `median_playtime_forever_hf` | 22.48% | 0.00% |
| `metacritic_score_hf` | 92.65% | 90.51% |
| `estimated_owners_hf` | 22.48% | 0.00% |
| `peak_ccu_hf` | 22.48% | 0.00% |
| `required_age_hf` | 22.48% | 0.00% |
| `dlc_count_hf` | 22.48% | 0.00% |

## 결측값 사용 시 주의사항

- HF 미매칭은 값이 0이라는 뜻이 아니라 **HF에서 알 수 없음**을 뜻합니다.
- 긍정·부정 평가, 도전 과제, 추천 수, 플레이타임의 0은 실제 값일 수 있으므로 그대로 보존했습니다.
- `metacritic_score_hf=0`은 HF의 미수집 표기로 판단해 결측으로 변환했습니다.
- `supported_languages_hf`, `full_audio_languages_hf`의 빈 목록 `[]`은 결측으로 변환했습니다.
- CSV의 빈 셀은 `pandas.read_csv()`에서 일반적으로 `NaN`이 됩니다. nullable 정수와 null을 정확히 유지하려면 Parquet을 사용하세요.

## 분석 권장사항

- 텍스트 모델에는 `description_clean` 사용을 권장합니다.
- 리뷰 비율은 `positive_hf + negative_hf > 0`인 행에서만 계산하세요.

```python
review_total = df["positive_hf"] + df["negative_hf"]
df["positive_ratio_hf"] = df["positive_hf"].div(review_total).where(review_total.gt(0))
```

- 플레이타임은 분 단위이며 오른쪽 꼬리가 길 수 있으므로 `log1p` 변환을 검토하세요.
- `estimated_owners_hf`는 범위 문자열입니다. 필요하면 하한과 상한을 별도 파생 컬럼으로 만드세요.
- 결측 대체 시 원본 컬럼을 덮어쓰지 말고 별도 파생 컬럼과 `hf_matched`, `has_*` 플래그를 함께 사용하세요.
- Metacritic은 유효 정보가 전체의 7.35%뿐이므로 핵심 변수보다 선택적 보조 변수로 사용하는 것이 안전합니다.

## 전처리 재실행

원천 데이터는 라이선스와 파일 크기 때문에 이 폴더에 포함하지 않았습니다. 다음 파일을 스크립트 또는 Notebook과 같은 디렉터리에 준비해야 합니다.

```text
games.csv                  # Kaggle
games_metadata.json        # Kaggle, JSON Lines
games_hugging.csv          # Hugging Face FronkonGames
```

필요한 패키지:

```bash
pip install pandas numpy pyarrow beautifulsoup4 nbformat
```

실행:

```bash
python steam_metadata_enrichment.py
```

스크립트는 HF 원본 CSV의 `DiscountDLC count` 결합 헤더를 감지하면 `Discount`와 `DLC count`로 자동 복구합니다.
