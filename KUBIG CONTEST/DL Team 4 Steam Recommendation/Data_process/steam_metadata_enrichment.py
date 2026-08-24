# %% [markdown]
# # Steam metadata description enrichment
#
# Kaggle metadata를 기준(base)으로 유지하고, 동일한 `app_id`를 가진
# Hugging Face FronkonGames 레코드의 설명만 보완합니다.
# 각 `# %%` 구간은 VS Code/Jupyter에서 개별 셀처럼 실행할 수 있습니다.

# %% 1. Imports
from __future__ import annotations

import ast
import csv
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# Windows의 기본 CP949 콘솔에서도 게임명/설명의 모든 유니코드를 안전하게 기록합니다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# BeautifulSoup이 없다면 다음 명령으로 설치하세요.
# pip install beautifulsoup4

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
GAMES_PATH = BASE_DIR / "games.csv"
METADATA_PATH = BASE_DIR / "games_metadata.json"
HF_PATH = BASE_DIR / "games_hugging.csv"

OUTPUT_CSV = BASE_DIR / "games_metadata_enriched.csv"
OUTPUT_PARQUET = BASE_DIR / "games_metadata_enriched.parquet"
MISSING_CSV = BASE_DIR / "games_description_still_missing.csv"
SUMMARY_PATH = BASE_DIR / "preprocessing_summary.txt"

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 160)


def section(title: str) -> None:
    print(f"\n{'=' * 24} {title} {'=' * 24}")


def resolve_column(
    columns: Iterable[str], candidates: Iterable[str], *, required: bool = False
) -> str | None:
    """대소문자/공백/밑줄 차이를 무시해 의미가 같은 컬럼을 찾습니다."""
    def key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).casefold())

    lookup = {key(column): column for column in columns}
    for candidate in candidates:
        if key(candidate) in lookup:
            return lookup[key(candidate)]
    if required:
        raise KeyError(
            f"필수 컬럼을 찾지 못했습니다. 후보={list(candidates)}, 실제={list(columns)}"
        )
    return None


def clean_missing_string(series: pd.Series) -> pd.Series:
    """설명 문자열의 공백과 문자열형 결측 표기를 pd.NA로 통일합니다."""
    missing_tokens = {"", "nan", "none", "null"}

    def normalize(value: Any) -> Any:
        if value is None or value is pd.NA:
            return pd.NA
        if isinstance(value, float) and np.isnan(value):
            return pd.NA
        if isinstance(value, str):
            stripped = value.strip()
            return pd.NA if stripped.casefold() in missing_tokens else stripped
        return value

    return series.map(normalize)


def inspect_frame(name: str, frame: pd.DataFrame, id_candidates: list[str]) -> None:
    """요청된 기본 구조 정보를 일정한 형식으로 출력합니다."""
    section(f"Inspect: {name}")
    print("shape:", frame.shape)
    print("column names:", frame.columns.tolist())
    print("head():")
    print(frame.head().to_string(index=False))
    id_column = resolve_column(frame.columns, id_candidates)
    print("주요 ID dtype:", {id_column: str(frame[id_column].dtype)} if id_column else "ID 컬럼 없음")
    print("결측치 개수:")
    print(frame.isna().sum().to_string())


def load_json_defensively(path: Path) -> pd.DataFrame:
    """우선 JSON Lines로 읽고, 실패하면 일반 JSON을 시도합니다."""
    try:
        result = pd.read_json(path, lines=True)
        print(f"{path.name}: JSON Lines(lines=True) 로드 성공")
        return result
    except ValueError as lines_error:
        print(f"{path.name}: JSON Lines 로드 실패({lines_error}); 일반 JSON 재시도")
        try:
            return pd.read_json(path)
        except ValueError as regular_error:
            raise ValueError(
                f"{path.name}을 JSON Lines와 일반 JSON 어느 형식으로도 읽지 못했습니다. "
                f"lines 오류={lines_error}; 일반 JSON 오류={regular_error}"
            ) from regular_error


def repaired_hf_header(path: Path) -> tuple[list[str], bool]:
    """HF CSV의 실제 필드 수를 확인하고 알려진 결합 헤더를 안전하게 복구합니다."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first_row = next(reader)

    if len(header) == len(first_row):
        return header, False

    # 현재 FronkonGames 파일에서 'Discount,DLC count'의 쉼표가 누락되어
    # 'DiscountDLC count' 한 컬럼처럼 기록된 경우를 명시적으로 처리합니다.
    combined = resolve_column(header, ["DiscountDLC count"])
    if len(first_row) == len(header) + 1 and combined is not None:
        index = header.index(combined)
        fixed = header[:index] + ["Discount", "DLC count"] + header[index + 1 :]
        assert len(fixed) == len(first_row)
        print(
            "Hugging Face CSV 헤더 복구: 'DiscountDLC count'를 "
            "'Discount'와 'DLC count'로 분리했습니다."
        )
        return fixed, True

    raise ValueError(
        "Hugging Face CSV의 헤더/데이터 필드 수가 예상과 다릅니다. "
        f"header={len(header)}, first_data_row={len(first_row)}. "
        "자동 복구 가능한 알려진 형식이 아니므로 원본 파일을 확인하세요."
    )


def load_and_inspect_hf(path: Path, chunksize: int = 10_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """큰 HF CSV를 chunk 단위로 점검하며 필요한 컬럼만 메모리에 유지합니다."""
    header, repaired = repaired_hf_header(path)
    app_col = resolve_column(header, ["appID", "app_id"], required=True)
    name_col = resolve_column(header, ["name", "title"])
    short_col = resolve_column(
        header, ["short_description", "short description", "short description english"]
    )
    detailed_col = resolve_column(
        header,
        ["detailed_description", "detailed description", "about the game", "description"],
    )
    tags_col = resolve_column(header, ["tags"])
    genres_col = resolve_column(header, ["genres"])
    categories_col = resolve_column(header, ["categories"])
    analysis_cols = {
        "developers": resolve_column(header, ["developers"]),
        "publishers": resolve_column(header, ["publishers"]),
        "supported_languages": resolve_column(header, ["supported languages"]),
        "full_audio_languages": resolve_column(header, ["full audio languages"]),
        "positive": resolve_column(header, ["positive"]),
        "negative": resolve_column(header, ["negative"]),
        "achievements": resolve_column(header, ["achievements"]),
        "recommendations": resolve_column(header, ["recommendations"]),
        "average_playtime_forever": resolve_column(header, ["average playtime forever"]),
        "median_playtime_forever": resolve_column(header, ["median playtime forever"]),
        "metacritic_score": resolve_column(header, ["metacritic score"]),
        "estimated_owners": resolve_column(header, ["estimated owners"]),
        "peak_ccu": resolve_column(header, ["peak ccu"]),
        "required_age": resolve_column(header, ["required age"]),
        "dlc_count": resolve_column(header, ["dlc count"]),
    }

    selected = [
        column
        for column in [
            app_col,
            name_col,
            short_col,
            detailed_col,
            tags_col,
            genres_col,
            categories_col,
            *analysis_cols.values(),
        ]
        if column is not None
    ]
    selected = list(dict.fromkeys(selected))

    # names=header를 지정하면 복구된 40개 헤더를 실제 40개 필드에 정확히 대응시킵니다.
    reader = pd.read_csv(
        path,
        header=0,
        names=header,
        chunksize=chunksize,
        low_memory=False,
    )
    parts: list[pd.DataFrame] = []
    missing_counts = pd.Series(0, index=header, dtype="int64")
    row_count = 0
    head = None
    first_dtypes = None

    for chunk_number, chunk in enumerate(reader, start=1):
        if head is None:
            head = chunk.head().copy()
            first_dtypes = chunk.dtypes.astype(str)
        row_count += len(chunk)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")
        parts.append(chunk[selected].copy())
        if chunk_number % 10 == 0:
            print(f"HF CSV 로딩 진행: {row_count:,}행")

    relevant = pd.concat(parts, ignore_index=True)
    assert len(relevant) == row_count, "HF chunk 결합 과정에서 행 수가 달라졌습니다."
    inspection = {
        "shape": (row_count, len(header)),
        "columns": header,
        "head": head,
        "dtypes": first_dtypes,
        "missing_counts": missing_counts,
        "id_column": app_col,
        "header_repaired": repaired,
        "resolved": {
            "app_id": app_col,
            "name": name_col,
            "short_description": short_col,
            "detailed_description": detailed_col,
            "tags": tags_col,
            "genres": genres_col,
            "categories": categories_col,
            **analysis_cols,
        },
    }
    return relevant, inspection


# %% 2. Load data
section("2. Load data")
for required_path in [GAMES_PATH, METADATA_PATH, HF_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {required_path}")

games_raw = pd.read_csv(GAMES_PATH, low_memory=False)
metadata_raw = load_json_defensively(METADATA_PATH)
hf_raw, hf_inspection = load_and_inspect_hf(HF_PATH)


# %% 3. Inspect schema
inspect_frame("Kaggle games.csv", games_raw, ["app_id", "appID"])
inspect_frame("Kaggle games_metadata.json", metadata_raw, ["app_id", "appID"])

section("Inspect: Hugging Face games_hugging.csv")
print("shape:", hf_inspection["shape"])
print("column names:", hf_inspection["columns"])
print("head():")
print(hf_inspection["head"].to_string(index=False))
hf_id_col = hf_inspection["id_column"]
print("주요 ID dtype:", {hf_id_col: hf_inspection["dtypes"].loc[hf_id_col]})
print("결측치 개수:")
print(hf_inspection["missing_counts"].to_string())
print("의미 기반 컬럼 매핑:", json.dumps(hf_inspection["resolved"], ensure_ascii=False, indent=2))


# %% 4. Clean IDs and missing values
section("4. Clean IDs and missing values")
meta_app_col = resolve_column(metadata_raw.columns, ["app_id", "appID"], required=True)
meta_desc_col = resolve_column(
    metadata_raw.columns, ["description", "detailed_description", "about the game"], required=True
)
meta_tags_col = resolve_column(metadata_raw.columns, ["tags"])

games_app_col = resolve_column(games_raw.columns, ["app_id", "appID"], required=True)
games_title_col = resolve_column(games_raw.columns, ["title", "name"])

hf_map = hf_inspection["resolved"]

metadata = pd.DataFrame(
    {
        "app_id": metadata_raw[meta_app_col],
        "description_original": metadata_raw[meta_desc_col],
        "tags_kaggle": metadata_raw[meta_tags_col] if meta_tags_col else pd.NA,
    }
)
games_lookup = pd.DataFrame(
    {
        "app_id": games_raw[games_app_col],
        "title_kaggle": games_raw[games_title_col] if games_title_col else pd.NA,
    }
)
hf = pd.DataFrame(
    {
        "app_id": hf_raw[hf_map["app_id"]],
        "name_hf": hf_raw[hf_map["name"]] if hf_map["name"] else pd.NA,
        "description_hf_short": (
            hf_raw[hf_map["short_description"]] if hf_map["short_description"] else pd.NA
        ),
        "description_hf_detailed": (
            hf_raw[hf_map["detailed_description"]]
            if hf_map["detailed_description"]
            else pd.NA
        ),
        "tags_huggingface": hf_raw[hf_map["tags"]] if hf_map["tags"] else pd.NA,
        "genres": hf_raw[hf_map["genres"]] if hf_map["genres"] else pd.NA,
        "categories": hf_raw[hf_map["categories"]] if hf_map["categories"] else pd.NA,
        "developers_hf": hf_raw[hf_map["developers"]] if hf_map["developers"] else pd.NA,
        "publishers_hf": hf_raw[hf_map["publishers"]] if hf_map["publishers"] else pd.NA,
        "supported_languages_hf": (
            hf_raw[hf_map["supported_languages"]] if hf_map["supported_languages"] else pd.NA
        ),
        "full_audio_languages_hf": (
            hf_raw[hf_map["full_audio_languages"]]
            if hf_map["full_audio_languages"]
            else pd.NA
        ),
        "positive_hf": hf_raw[hf_map["positive"]] if hf_map["positive"] else pd.NA,
        "negative_hf": hf_raw[hf_map["negative"]] if hf_map["negative"] else pd.NA,
        "achievements_hf": (
            hf_raw[hf_map["achievements"]] if hf_map["achievements"] else pd.NA
        ),
        "recommendations_hf": (
            hf_raw[hf_map["recommendations"]] if hf_map["recommendations"] else pd.NA
        ),
        "average_playtime_forever_hf": (
            hf_raw[hf_map["average_playtime_forever"]]
            if hf_map["average_playtime_forever"]
            else pd.NA
        ),
        "median_playtime_forever_hf": (
            hf_raw[hf_map["median_playtime_forever"]]
            if hf_map["median_playtime_forever"]
            else pd.NA
        ),
        "metacritic_score_hf": (
            hf_raw[hf_map["metacritic_score"]] if hf_map["metacritic_score"] else pd.NA
        ),
        "estimated_owners_hf": (
            hf_raw[hf_map["estimated_owners"]] if hf_map["estimated_owners"] else pd.NA
        ),
        "peak_ccu_hf": hf_raw[hf_map["peak_ccu"]] if hf_map["peak_ccu"] else pd.NA,
        "required_age_hf": (
            hf_raw[hf_map["required_age"]] if hf_map["required_age"] else pd.NA
        ),
        "dlc_count_hf": hf_raw[hf_map["dlc_count"]] if hf_map["dlc_count"] else pd.NA,
    }
)

for frame_name, frame in [
    ("Kaggle metadata", metadata),
    ("Kaggle games", games_lookup),
    ("Hugging Face", hf),
]:
    frame["app_id"] = pd.to_numeric(frame["app_id"], errors="coerce").astype("Int64")
    invalid_count = int(frame["app_id"].isna().sum())
    print(f"{frame_name}: app_id dtype={frame['app_id'].dtype}, 변환 불가/결측={invalid_count:,}")

for column in ["description_original"]:
    metadata[column] = clean_missing_string(metadata[column])
for column in ["description_hf_short", "description_hf_detailed"]:
    hf[column] = clean_missing_string(hf[column])

# HF 문자열형 분석 컬럼의 결측 표기를 통일합니다.
for column in [
    "developers_hf",
    "publishers_hf",
    "supported_languages_hf",
    "full_audio_languages_hf",
    "estimated_owners_hf",
]:
    hf[column] = clean_missing_string(hf[column])

# 언어 목록의 빈 리스트 문자열은 '언어 정보 없음'으로 해석합니다.
for column in ["supported_languages_hf", "full_audio_languages_hf"]:
    empty_list_mask = hf[column].astype("string").str.strip().eq("[]").fillna(False)
    hf.loc[empty_list_mask, column] = pd.NA

# 수치형 HF 컬럼은 nullable integer로 유지합니다.
# 0이 실제 값일 수 있는 평가/도전과제/플레이타임은 임의로 결측 처리하지 않습니다.
hf_numeric_columns = [
    "positive_hf",
    "negative_hf",
    "achievements_hf",
    "recommendations_hf",
    "average_playtime_forever_hf",
    "median_playtime_forever_hf",
    "metacritic_score_hf",
    "peak_ccu_hf",
    "required_age_hf",
    "dlc_count_hf",
]
for column in hf_numeric_columns:
    hf[column] = pd.to_numeric(hf[column], errors="coerce").astype("Int64")

# HF의 Metacritic 0은 실제 0점이 아니라 미수집 값이므로 명시적 결측으로 바꿉니다.
hf["metacritic_score_hf"] = hf["metacritic_score_hf"].mask(
    hf["metacritic_score_hf"].eq(0)
)


def report_duplicates(name: str, frame: pd.DataFrame) -> int:
    duplicate_mask = frame["app_id"].notna() & frame["app_id"].duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_ids = int(frame.loc[duplicate_mask, "app_id"].nunique())
    print(f"{name}: 중복 행={duplicate_rows:,}, 중복 app_id={duplicate_ids:,}")
    if duplicate_rows:
        print("중복 샘플(삭제 전):")
        print(frame.loc[duplicate_mask].sort_values("app_id").head(20).to_string(index=False))
    return duplicate_rows


metadata_duplicate_rows = report_duplicates("Kaggle metadata", metadata)
games_duplicate_rows = report_duplicates("Kaggle games", games_lookup)
hf_duplicate_rows = report_duplicates("Hugging Face", hf)

# HF 중복은 요구사항대로 샘플을 먼저 출력한 후 첫 행을 유지합니다.
hf_before_dedup = len(hf)
hf = hf.drop_duplicates(subset="app_id", keep="first").copy()
print(f"Hugging Face 중복 제거: {hf_before_dedup - len(hf):,}행 제거(첫 번째 행 유지)")

# games.csv는 title lookup용이므로 병합 행 폭증을 막기 위해 첫 행만 사용합니다.
games_lookup = games_lookup.drop_duplicates(subset="app_id", keep="first").copy()


# %% 5. Merge datasets
section("5. Merge datasets")
hf["_hf_matched"] = True
merged = metadata.merge(hf, on="app_id", how="left", validate="m:1", sort=False)
merged = merged.merge(games_lookup, on="app_id", how="left", validate="m:1", sort=False)
merged["hf_matched"] = merged["_hf_matched"].eq(True).astype(bool)
merged["has_language_info_hf"] = merged["supported_languages_hf"].notna()
merged["has_playtime_hf"] = merged["average_playtime_forever_hf"].fillna(0).gt(0)
merged["has_metacritic_hf"] = merged["metacritic_score_hf"].notna()

assert len(merged) == len(metadata), "left join 후 행 수가 변했습니다. 중복 키를 확인하세요."
assert merged["app_id"].reset_index(drop=True).equals(
    metadata["app_id"].reset_index(drop=True)
), "left join 후 Kaggle metadata의 app_id 순서/구성이 변했습니다."

base_count = len(metadata)
matched_count = int(merged["_hf_matched"].eq(True).sum())
unmatched_count = base_count - matched_count
match_rate = matched_count / base_count if base_count else np.nan
original_missing = int(merged["description_original"].isna().sum())
original_missing_rate = original_missing / base_count if base_count else np.nan

print(f"Kaggle metadata 전체 게임 수: {base_count:,}")
print(f"Hugging Face와 app_id가 매칭된 게임 수: {matched_count:,}")
print(f"매칭되지 않은 게임 수: {unmatched_count:,}")
print(f"매칭률: {match_rate:.2%}")
print(f"원래 Kaggle description 결측: {original_missing:,} ({original_missing_rate:.2%})")
print("병합 성공: Kaggle metadata 행 수와 app_id 구성이 그대로 유지되었습니다.")


def normalized_title(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def title_ratio(left: Any, right: Any) -> float:
    left_norm, right_norm = normalized_title(left), normalized_title(right)
    if not left_norm or not right_norm:
        return np.nan
    try:
        from rapidfuzz.fuzz import ratio

        return ratio(left_norm, right_norm) / 100.0
    except ImportError:
        return SequenceMatcher(None, left_norm, right_norm).ratio()


title_check = merged.loc[
    merged["title_kaggle"].notna() & merged["name_hf"].notna(),
    ["app_id", "title_kaggle", "name_hf"],
].copy()
title_check["title_similarity"] = [
    title_ratio(left, right)
    for left, right in zip(title_check["title_kaggle"], title_check["name_hf"])
]
title_check["normalized_equal"] = [
    normalized_title(left) == normalized_title(right)
    for left, right in zip(title_check["title_kaggle"], title_check["name_hf"])
]
print("\napp_id는 같지만 제목 유사도가 가장 낮은 검증 샘플 20개:")
print(
    title_check.loc[~title_check["normalized_equal"]]
    .sort_values("title_similarity")
    .head(20)
    .to_string(index=False)
)


# %% 6. Build description_final
section("6. Build description_final")
merged["description_final"] = (
    merged["description_original"]
    .combine_first(merged["description_hf_short"])
    .combine_first(merged["description_hf_detailed"])
)
merged["description_source"] = np.select(
    [
        merged["description_original"].notna(),
        merged["description_hf_short"].notna(),
        merged["description_hf_detailed"].notna(),
    ],
    ["kaggle", "huggingface_short", "huggingface_detailed"],
    default="missing",
)

assert merged.loc[merged["description_original"].notna(), "description_final"].equals(
    merged.loc[merged["description_original"].notna(), "description_original"]
), "기존 Kaggle description이 덮어써졌습니다."
assert set(merged["description_source"].unique()) <= {
    "kaggle", "huggingface_short", "huggingface_detailed", "missing"
}


# %% 7. Analyze missingness
section("7. Analyze missingness")
filled_short = int((merged["description_source"] == "huggingface_short").sum())
filled_detailed = int((merged["description_source"] == "huggingface_detailed").sum())
filled_total = filled_short + filled_detailed
final_missing = int(merged["description_final"].isna().sum())
final_missing_rate = final_missing / base_count if base_count else np.nan
resolved_pct = filled_total / original_missing if original_missing else np.nan

print(f"원래 description 결측 개수: {original_missing:,}")
print(f"원래 description 결측률: {original_missing_rate:.2%}")
print(f"HF short_description으로 새로 채운 개수: {filled_short:,}")
print(f"HF detailed_description으로 새로 채운 개수: {filled_detailed:,}")
print(f"최종 남은 결측 개수: {final_missing:,}")
print(f"최종 결측률: {final_missing_rate:.2%}")
print(f"원래 결측 중 Hugging Face로 해소한 비율: {resolved_pct:.2%}")
print("description_source value_counts:")
print(merged["description_source"].value_counts(dropna=False).to_string())


def print_text_samples(frame: pd.DataFrame, mask: pd.Series, label: str, n: int = 5) -> None:
    subset = frame.loc[mask, ["app_id", "title_kaggle", "name_hf", "description_final"]].head(n).copy()
    print(f"\n{label}: {int(mask.sum()):,}개 / 샘플 {min(n, len(subset))}개")
    if subset.empty:
        print("(샘플 없음)")
    else:
        print(subset.to_string(index=False))


section("7-b. Text quality")
nonmissing_text = merged["description_final"].dropna().astype(str)
char_length = merged["description_final"].astype("string").str.len()
word_count = merged["description_final"].astype("string").str.findall(r"\S+").str.len()
print("문자 수 describe():")
print(char_length.describe().to_string())
print("단어 수 describe():")
print(word_count.describe().to_string())

for threshold in [10, 30, 50]:
    mask = merged["description_final"].notna() & char_length.lt(threshold).fillna(False)
    print_text_samples(merged, mask, f"{threshold}자 미만")
for threshold in [2_000, 5_000]:
    mask = merged["description_final"].notna() & char_length.gt(threshold).fillna(False)
    print_text_samples(merged, mask, f"{threshold:,}자 초과")

html_tag_pattern = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
html_mask = merged["description_final"].astype("string").str.contains(
    html_tag_pattern, na=False, regex=True
)
print_text_samples(merged, html_mask, "HTML 태그 포함 설명")


# %% 8. Clean text
section("8. Clean text")
def clean_description(value: Any) -> Any:
    """원문을 보존하면서 모델 입력용 텍스트만 보수적으로 정제합니다."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    text = html.unescape(str(value))
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return pd.NA if text.casefold() in {"", "nan", "none", "null"} else text


merged["description_clean"] = merged["description_final"].map(clean_description)
clean_html_count = int(
    merged["description_clean"].astype("string").str.contains(html_tag_pattern, na=False).sum()
)
print(f"정제 전 HTML 태그 포함: {int(html_mask.sum()):,}개")
print(f"정제 후 HTML 태그 포함: {clean_html_count:,}개")
print("description_clean 샘플:")
print(merged[["app_id", "description_final", "description_clean"]].head().to_string(index=False))


# %% 9. Process metadata columns
section("9. Process metadata columns")
def audit_literal_lists(series: pd.Series, column_name: str, sample_n: int = 10) -> dict[str, int]:
    """list처럼 보이는 문자열만 literal_eval로 시험하고 원본 series는 변경하지 않습니다."""
    stats = Counter(already_list=0, candidates=0, parsed=0, failed=0)
    failures: list[tuple[Any, str]] = []
    for index, value in series.items():
        if isinstance(value, (list, tuple, set, dict)):
            stats["already_list"] += 1
            continue
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            continue
        stats["candidates"] += 1
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple, set)):
                stats["parsed"] += 1
            else:
                stats["failed"] += 1
                failures.append((index, stripped[:300]))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            stats["failed"] += 1
            failures.append((index, stripped[:300]))

    print(f"{column_name}: {dict(stats)}")
    if failures:
        print(f"{column_name} 파싱 실패 샘플(원본 유지):")
        for index, value in failures[:sample_n]:
            print(f"  index={index}: {value}")
    return dict(stats)


metadata_parse_audit = {}
for metadata_column in ["tags_kaggle", "tags_huggingface", "genres", "categories"]:
    metadata_parse_audit[metadata_column] = audit_literal_lists(
        merged[metadata_column], metadata_column
    )
print("eval()은 사용하지 않았으며, 파싱 감사 후 모든 metadata 원본 값을 그대로 유지합니다.")


# %% 10. Build final dataset
section("10. Build final dataset")
merged["title"] = clean_missing_string(merged["title_kaggle"]).combine_first(
    clean_missing_string(merged["name_hf"])
)

# 팀 분석용 최종본은 games.csv 자체를 base로 사용합니다.
# 원본 13개 컬럼과 순서를 그대로 앞에 두고, metadata 보완 컬럼만 오른쪽에 추가합니다.
games_base = games_raw.copy()
games_base[games_app_col] = pd.to_numeric(
    games_base[games_app_col], errors="coerce"
).astype("Int64")
if games_app_col != "app_id":
    games_base = games_base.rename(columns={games_app_col: "app_id"})

original_games_columns = [
    "app_id" if column == games_app_col else column for column in games_raw.columns
]
added_columns = [
    "description_original",
    "description_hf_short",
    "description_hf_detailed",
    "description_final",
    "description_clean",
    "description_source",
    "tags_kaggle",
    "tags_huggingface",
    "genres",
    "categories",
    "hf_matched",
    "developers_hf",
    "publishers_hf",
    "supported_languages_hf",
    "full_audio_languages_hf",
    "positive_hf",
    "negative_hf",
    "achievements_hf",
    "recommendations_hf",
    "average_playtime_forever_hf",
    "median_playtime_forever_hf",
    "metacritic_score_hf",
    "estimated_owners_hf",
    "peak_ccu_hf",
    "required_age_hf",
    "dlc_count_hf",
    "has_language_info_hf",
    "has_playtime_hf",
    "has_metacritic_hf",
]
final_columns = original_games_columns + added_columns

overlapping_columns = set(original_games_columns) & set(added_columns)
assert not overlapping_columns, f"원본/추가 컬럼명이 충돌합니다: {sorted(overlapping_columns)}"

enrichment_lookup = merged[["app_id"] + added_columns].copy()
assert not enrichment_lookup["app_id"].duplicated().any(), "metadata enrichment app_id가 중복입니다."

final_df = games_base.merge(
    enrichment_lookup,
    on="app_id",
    how="left",
    validate="1:1",
    sort=False,
)

assert final_df.shape[0] == games_base.shape[0]
assert final_df.columns.tolist() == final_columns
assert final_df["app_id"].reset_index(drop=True).equals(
    games_base["app_id"].reset_index(drop=True)
)
assert final_df[original_games_columns].equals(
    games_base[original_games_columns]
), "games.csv 원본 컬럼의 값 또는 행 순서가 변경되었습니다."
assert set(final_df["app_id"].dropna()) == set(metadata["app_id"].dropna()), (
    "games.csv와 games_metadata.json의 app_id 집합이 다릅니다."
)
assert int(final_df["hf_matched"].sum()) == matched_count
assert final_df.loc[~final_df["hf_matched"], "positive_hf"].isna().all(), (
    "HF 미매칭 행에 HF 평가 정보가 들어갔습니다."
)
assert final_df["metacritic_score_hf"].dropna().gt(0).all(), (
    "Metacritic의 0점 미수집 표기가 결측으로 변환되지 않았습니다."
)
assert final_df["has_playtime_hf"].equals(
    final_df["average_playtime_forever_hf"].fillna(0).gt(0)
)
print("final shape:", final_df.shape)
print("final columns:", final_df.columns.tolist())
print(f"원본 games.csv 컬럼 보존: {len(original_games_columns)}개")
print(f"오른쪽에 추가된 metadata 컬럼: {len(added_columns)}개")
print("HF 분석 컬럼 정보 보유 현황:")
print(f"- hf_matched: {int(final_df['hf_matched'].sum()):,}개")
print(f"- has_language_info_hf: {int(final_df['has_language_info_hf'].sum()):,}개")
print(f"- has_playtime_hf (> 0분): {int(final_df['has_playtime_hf'].sum()):,}개")
print(f"- has_metacritic_hf (> 0점): {int(final_df['has_metacritic_hf'].sum()):,}개")
print(final_df.head().to_string(index=False))


# %% 11. Save files
section("11. Save files")
final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
final_df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
still_missing_df = final_df.loc[final_df["description_final"].isna()].copy()
still_missing_df.to_csv(MISSING_CSV, index=False, encoding="utf-8-sig")

generated_files = [
    OUTPUT_CSV.name,
    OUTPUT_PARQUET.name,
    MISSING_CSV.name,
    SUMMARY_PATH.name,
]

COLUMN_DICTIONARY = {
    # Kaggle games.csv 원본 컬럼
    "app_id": "[Kaggle] Steam 애플리케이션 고유 ID. 모든 병합의 기준 키.",
    "title": "[Kaggle] Steam 게임명.",
    "date_release": "[Kaggle] 출시일. 원본 표기는 YYYY-MM-DD 문자열.",
    "win": "[Kaggle] Windows 지원 여부(bool).",
    "mac": "[Kaggle] macOS 지원 여부(bool).",
    "linux": "[Kaggle] Linux 지원 여부(bool).",
    "rating": "[Kaggle] Steam 사용자 평가의 범주형 등급(예: Very Positive, Mixed).",
    "positive_ratio": "[Kaggle] 사용자 리뷰 중 긍정 평가 비율. 단위는 퍼센트(0~100).",
    "user_reviews": "[Kaggle] 수집 시점의 사용자 리뷰 수.",
    "price_final": "[Kaggle] 할인 적용 후 최종 가격. 원본 데이터의 통화 기준을 따름.",
    "price_original": "[Kaggle] 할인 전 원래 가격. 원본 데이터의 통화 기준을 따름.",
    "discount": "[Kaggle] 할인율. 원본 데이터의 퍼센트 단위 값을 유지.",
    "steam_deck": "[Kaggle] Steam Deck 지원/호환 여부(bool, Kaggle 원본 정의).",
    # 설명 및 태그 보완 컬럼
    "description_original": "[Kaggle metadata] 원래 description. 문자열형 결측 표기만 통일했으며 본문은 보존.",
    "description_hf_short": "[HF] short description 후보. 현재 HF 파일에는 대응 컬럼이 없어 전부 결측.",
    "description_hf_detailed": "[HF] About the game을 상세 설명으로 매핑한 원문.",
    "description_final": "모델/분석용 최종 원문. Kaggle description > HF short > HF detailed 순으로 선택.",
    "description_clean": "description_final에서 HTML entity/tag와 반복 공백만 보수적으로 정리한 텍스트.",
    "description_source": "최종 설명 출처. kaggle, huggingface_short, huggingface_detailed, missing 중 하나.",
    "tags_kaggle": "[Kaggle metadata] 태그 목록 원본. Parquet에서는 list, CSV에서는 문자열처럼 보일 수 있음.",
    "tags_huggingface": "[HF] 쉼표로 구분된 Steam 태그 원본 문자열.",
    "genres": "[HF] 쉼표로 구분된 장르 원본 문자열.",
    "categories": "[HF] 싱글플레이, 협동, 컨트롤러 지원 등 Steam 기능/카테고리 문자열.",
    # HF 분석용 컬럼
    "hf_matched": "Kaggle app_id가 HF에 존재하는지 나타내는 bool. False이면 HF 파생 값은 원칙적으로 결측.",
    "developers_hf": "[HF] 개발사명. 복수 값은 원본 구분 형식을 유지.",
    "publishers_hf": "[HF] 퍼블리셔명. 복수 값은 원본 구분 형식을 유지.",
    "supported_languages_hf": "[HF] 지원 텍스트 언어 목록. 빈 목록 []은 결측 처리.",
    "full_audio_languages_hf": "[HF] 전체 음성을 지원하는 언어 목록. 빈 목록 []은 결측 처리.",
    "positive_hf": "[HF] 긍정 평가 수. 0은 실제 값일 수 있어 보존하며 HF 미매칭만 결측.",
    "negative_hf": "[HF] 부정 평가 수. 0은 실제 값일 수 있어 보존하며 HF 미매칭만 결측.",
    "achievements_hf": "[HF] Steam 도전 과제 수. 0은 실제 값일 수 있어 보존.",
    "recommendations_hf": "[HF] Steam 추천/리뷰 관련 집계 수. 수집 시점 차이에 유의.",
    "average_playtime_forever_hf": "[HF] 전체 기간 평균 플레이타임. 단위는 분(minutes); 0은 보존.",
    "median_playtime_forever_hf": "[HF] 전체 기간 플레이타임 중앙값. 단위는 분(minutes); 0은 보존.",
    "metacritic_score_hf": "[HF] Metacritic 점수(0~100). 원본의 미수집 표기 0은 결측으로 변환.",
    "estimated_owners_hf": "[HF] 추정 소유자 수 범위 문자열(예: 0 - 20000). 단일 숫자가 아님.",
    "peak_ccu_hf": "[HF] 수집 데이터 기준 최고 동시 접속자 수(Peak CCU).",
    "required_age_hf": "[HF] 요구 연령. 0은 제한 없음 또는 미기재가 섞일 수 있어 해석에 유의.",
    "dlc_count_hf": "[HF] 연결된 DLC 개수.",
    # 정보 유무 플래그
    "has_language_info_hf": "supported_languages_hf가 결측이 아닌지 나타내는 bool.",
    "has_playtime_hf": "average_playtime_forever_hf가 0보다 큰지 나타내는 bool.",
    "has_metacritic_hf": "유효한 metacritic_score_hf가 존재하는지 나타내는 bool.",
}
assert list(COLUMN_DICTIONARY) == final_columns, (
    "컬럼 사전의 순서/구성이 최종 데이터 컬럼과 다릅니다."
)

MISSING_VALUE_GUIDE = [
    "CSV의 빈 셀은 pandas로 다시 읽을 때 일반적으로 NaN이 되며, Parquet은 native null을 유지합니다.",
    "HF 미매칭(hf_matched=False)은 '값이 0'이 아니라 'HF에서 알 수 없음'이므로 HF 컬럼을 결측으로 유지합니다.",
    "positive/negative/achievements/playtime의 0은 실제 0일 수 있어 임의로 결측 또는 평균값으로 바꾸지 않았습니다.",
    "metacritic_score_hf의 원본 0은 미수집 표기로 판단해 결측으로 변환했습니다.",
    "supported_languages_hf와 full_audio_languages_hf의 빈 목록 문자열 []은 결측으로 변환했습니다.",
    "description_hf_short는 현재 HF 원본에 대응 컬럼이 없으므로 전부 결측이며 오류가 아닙니다.",
]

USAGE_GUIDE = [
    "추천 모델 텍스트 입력에는 원문 description_final보다 HTML/공백을 정리한 description_clean 사용을 권장합니다.",
    "Kaggle과 HF의 리뷰/가격 정보는 수집 시점이 다를 수 있으므로 서로 덮어쓰지 말고 별도 특징으로 사용하세요.",
    "HF 리뷰 비율은 positive_hf + negative_hf > 0인 행에서만 positive_hf / (positive_hf + negative_hf)로 계산하세요.",
    "플레이타임은 오른쪽 꼬리가 길 수 있으므로 모델링 시 log1p 변환을 검토하세요. 원본 분 단위 컬럼은 보존하세요.",
    "estimated_owners_hf는 범위 문자열이므로 필요하면 하한/상한을 별도 파싱한 파생 컬럼을 만드세요.",
    "결측 대체가 필요하면 원본 컬럼을 덮어쓰지 말고 별도 파생 컬럼과 정보 유무 플래그를 함께 사용하세요.",
]

WHY_HF_GUIDE = [
    "Kaggle games.csv는 게임 목록, 가격, 플랫폼, 평가 요약에는 강하지만 콘텐츠와 이용 행태 정보가 제한적입니다.",
    "HF의 설명/태그/장르/카테고리는 콘텐츠 기반 추천에서 게임 간 의미적 유사도를 만드는 데 사용하기 위해 추가했습니다.",
    "개발사, 퍼블리셔, 지원 언어는 취향·지역화·제작사 선호 같은 분석 축을 제공하기 위해 추가했습니다.",
    "긍정/부정 평가와 추천 수는 Kaggle 지표의 대체값이 아니라 수집 시점이 다른 보조 신호로 사용하기 위해 분리했습니다.",
    "플레이타임, 소유자 범위, Peak CCU는 단순 구매/평가 외에 실제 참여도와 대중성을 살펴보기 위해 추가했습니다.",
    "Metacritic은 희소하지만 외부 품질 신호로 활용할 수 있어 선택적 보조 변수로 포함했습니다.",
]

WHY_STRUCTURE_GUIDE = [
    "추천 대상 게임의 범위가 바뀌지 않도록 Kaggle games.csv의 50,872개 app_id를 기준으로 left join했습니다.",
    "Kaggle 원본 13개 컬럼을 앞쪽에 그대로 보존해 기존 팀 분석 코드와의 호환성을 유지했습니다.",
    "HF 값은 _hf 접미사를 사용해 출처와 수집 시점이 다른 값을 Kaggle 값과 혼동하거나 덮어쓰지 않도록 했습니다.",
    "description_original/final/clean을 분리해 원문 보존, 보완 과정 감사, 모델 입력이라는 목적을 각각 충족했습니다.",
    "hf_matched와 has_* 플래그를 두어 '실제 값 0'과 'HF에서 알 수 없음'을 구분할 수 있게 했습니다.",
    "CSV는 공유·확인용, Parquet은 nullable 타입과 저장 효율을 보존하는 분석용으로 함께 제공합니다.",
]

HF_MISSINGNESS_COLUMNS = [
    "description_hf_short",
    "description_hf_detailed",
    "tags_huggingface",
    "genres",
    "categories",
    "developers_hf",
    "publishers_hf",
    "supported_languages_hf",
    "full_audio_languages_hf",
    "positive_hf",
    "negative_hf",
    "achievements_hf",
    "recommendations_hf",
    "average_playtime_forever_hf",
    "median_playtime_forever_hf",
    "metacritic_score_hf",
    "estimated_owners_hf",
    "peak_ccu_hf",
    "required_age_hf",
    "dlc_count_hf",
]
matched_mask_for_report = final_df["hf_matched"]
hf_missingness_lines = []
for column in HF_MISSINGNESS_COLUMNS:
    overall_missing_count = int(final_df[column].isna().sum())
    overall_missing_rate = overall_missing_count / len(final_df) if len(final_df) else np.nan
    matched_missing_count = int(final_df.loc[matched_mask_for_report, column].isna().sum())
    matched_missing_rate = (
        matched_missing_count / int(matched_mask_for_report.sum())
        if int(matched_mask_for_report.sum())
        else np.nan
    )
    hf_missingness_lines.append(
        f"- {column}: 전체 결측 {overall_missing_count:,}/{len(final_df):,} "
        f"({overall_missing_rate:.2%}); HF 매칭 행 내부 결측 "
        f"{matched_missing_count:,}/{int(matched_mask_for_report.sum()):,} "
        f"({matched_missing_rate:.2%})"
    )

summary_lines = [
    "Steam game metadata preprocessing summary",
    "=" * 43,
    f"원본 게임 수: {base_count:,}",
    f"원본 description 결측률: {original_missing_rate:.2%} ({original_missing:,}/{base_count:,})",
    f"Hugging Face 매칭률: {match_rate:.2%} ({matched_count:,}/{base_count:,})",
    f"short_description으로 보완된 수: {filled_short:,}",
    f"detailed_description으로 보완된 수: {filled_detailed:,}",
    f"최종 결측률: {final_missing_rate:.2%} ({final_missing:,}/{base_count:,})",
    f"최종 남은 결측 게임 수: {final_missing:,}",
    f"최종 데이터 shape: {final_df.shape}",
    f"games.csv 원본 컬럼 보존 수: {len(original_games_columns)}",
    "추가된 컬럼: " + ", ".join(added_columns),
    f"HF 매칭 플래그 True: {int(final_df['hf_matched'].sum()):,}",
    f"지원 언어 정보 보유: {int(final_df['has_language_info_hf'].sum()):,}",
    f"누적 평균 플레이타임 > 0: {int(final_df['has_playtime_hf'].sum()):,}",
    f"유효 Metacritic 점수 보유: {int(final_df['has_metacritic_hf'].sum()):,}",
    "Hugging Face description 컬럼 매핑: "
    + (
        f"short={hf_map['short_description']!r}, detailed={hf_map['detailed_description']!r}"
    ),
    "",
    "왜 Hugging Face 컬럼을 추가했나",
    "=" * 43,
    *[f"- {item}" for item in WHY_HF_GUIDE],
    "",
    "왜 이런 데이터 구조를 만들었나",
    "=" * 43,
    *[f"- {item}" for item in WHY_STRUCTURE_GUIDE],
    "",
    "HF 컬럼별 결측 현황",
    "=" * 43,
    "설명: 전체 결측률에는 HF 미매칭 11,436개가 포함됩니다. 'HF 매칭 행 내부 결측률'은 원본 HF 자체의 정보 부족을 보여줍니다.",
    f"hf_matched=False: {int((~final_df['hf_matched']).sum()):,}/{len(final_df):,} ({(~final_df['hf_matched']).mean():.2%})",
    *hf_missingness_lines,
    "정보 유무 플래그(hf_matched, has_language_info_hf, has_playtime_hf, has_metacritic_hf)는 결측 없이 True/False로 제공됩니다.",
    "",
    "컬럼 사전 (총 42개)",
    "=" * 43,
    *[f"- {column}: {description}" for column, description in COLUMN_DICTIONARY.items()],
    "",
    "결측값 해석 규칙",
    "=" * 43,
    *[f"- {item}" for item in MISSING_VALUE_GUIDE],
    "",
    "팀 분석 권장사항",
    "=" * 43,
    *[f"- {item}" for item in USAGE_GUIDE],
    "",
    "생성된 파일 목록:",
    *[f"- {name}" for name in generated_files],
]
SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

for output_path in [OUTPUT_CSV, OUTPUT_PARQUET, MISSING_CSV, SUMMARY_PATH]:
    assert output_path.exists() and output_path.stat().st_size > 0, f"출력 실패: {output_path}"
    print(f"저장 완료: {output_path.name} ({output_path.stat().st_size:,} bytes)")


# %% 12. Print final summary and verify outputs
section("12. Print final summary")
# 대용량 CSV는 전체를 재로딩하지 않고 행 수/헤더를 스트리밍 검증합니다.
with OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
    csv_reader = csv.reader(handle)
    saved_header = next(csv_reader)
    saved_csv_rows = sum(1 for _ in csv_reader)

parquet_check = pd.read_parquet(OUTPUT_PARQUET, columns=["app_id", "description_source"])
missing_check = pd.read_csv(MISSING_CSV, usecols=["app_id", "description_source"])

assert saved_header == final_columns, "저장된 CSV 헤더가 최종 컬럼과 다릅니다."
assert saved_csv_rows == len(final_df), "저장된 CSV 행 수가 최종 DataFrame과 다릅니다."
assert len(parquet_check) == len(final_df), "저장된 Parquet 행 수가 다릅니다."
assert len(missing_check) == final_missing, "결측 전용 CSV의 행 수가 통계와 다릅니다."
assert (parquet_check["description_source"] == "missing").sum() == final_missing

print("병합 성공 여부: 성공 (Kaggle metadata 기준 left join 및 행 수 보존 검증 완료)")
print(f"최종 데이터 shape: {final_df.shape}")
print(f"description 보완 전후 결측률: {original_missing_rate:.2%} -> {final_missing_rate:.2%}")
print(f"Hugging Face로 보완된 수: {filled_total:,}")
print("최종 생성 파일 목록:")
for name in generated_files:
    print(f"- {name}")
print("\n" + "\n".join(summary_lines))
