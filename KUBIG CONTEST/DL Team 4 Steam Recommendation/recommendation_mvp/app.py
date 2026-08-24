"""Streamlit UI for the Steam recommendation MVP."""

from __future__ import annotations

import io
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mvp_recommendation.cold_start import ColdStartRecommendationPipeline  # noqa: E402
from mvp_recommendation.inference import KnownUserRecommendationPipeline  # noqa: E402
from mvp_recommendation.reranking import mmr_rerank  # noqa: E402


CHECKPOINT_DIR = REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "checkpoints"
HYBRID_SUMMARY = REPO_ROOT / "outputs" / "mvp_50k" / "repro_seed_42" / "hybrid" / "hybrid_summary.json"
DATA_DIR = REPO_ROOT / "outputs" / "mvp_50k" / "data_seed_42"
DEPLOY_DATA_DIR = REPO_ROOT / "recommendation_mvp" / "deploy_data"
CATALOG_PATH = DEPLOY_DATA_DIR / "catalog_ui.parquet"
HISTORY_PATH = DEPLOY_DATA_DIR / "seen_history_all.parquet"
POPULARITY_PATH = DEPLOY_DATA_DIR / "train_positive_counts.csv"
TEXT_PREFIX = REPO_ROOT / "text_data" / "emb_text_minilm"
MULTIMODAL_PREFIX = REPO_ROOT / "game_fusion" / "emb_game_concat_64"
MULTIMODAL_CHECKPOINT = (
    REPO_ROOT / "recommendation_mvp" / "model_artifacts" / "frozen_multimodal_user_bpr_seed42.pt"
)
MULTIMODAL_SUMMARY = (
    REPO_ROOT / "recommendation_mvp" / "model_artifacts" / "multimodal_evaluation_summary_seed42.json"
)


@st.cache_data(show_spinner=False)
def load_ui_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    catalog = pd.read_parquet(
        CATALOG_PATH,
        columns=["app_id", "title", "rating", "positive_ratio", "price_final"],
    )
    tags = pd.read_csv(REPO_ROOT / "recommendation_mvp" / "available_tags.csv")
    labels = [f"{title}  [app_id={app_id}]" for app_id, title in zip(catalog.app_id, catalog.title)]
    label_to_id = dict(zip(labels, catalog.app_id.astype(int)))
    return catalog, tags, labels, label_to_id


@st.cache_resource(show_spinner="기존 사용자 추천 모델을 불러오는 중입니다...")
def load_known_pipeline() -> KnownUserRecommendationPipeline:
    return KnownUserRecommendationPipeline(
        checkpoint_dir=CHECKPOINT_DIR,
        hybrid_summary_path=HYBRID_SUMMARY,
        text_prefix=TEXT_PREFIX,
        tabular_prefix=None,
        catalog_path=CATALOG_PATH,
        data_dir=None,
        history_path=HISTORY_PATH,
        device="cpu",
        history_scope="all",
        multimodal_prefix=MULTIMODAL_PREFIX,
        multimodal_checkpoint=MULTIMODAL_CHECKPOINT,
        multimodal_summary_path=MULTIMODAL_SUMMARY,
    )


@st.cache_resource(show_spinner="신규 사용자 추천 데이터를 불러오는 중입니다...")
def load_cold_pipeline() -> ColdStartRecommendationPipeline:
    return ColdStartRecommendationPipeline(
        text_prefix=TEXT_PREFIX,
        catalog_path=CATALOG_PATH,
        train_path=None,
        popularity_path=POPULARITY_PATH,
        multimodal_prefix=MULTIMODAL_PREFIX,
    )


def csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def steam_header_image_url(app_id: int) -> str:
    """Return Steam's public store header image URL for an app_id."""
    return (
        "https://shared.akamai.steamstatic.com/store_item_assets/"
        f"steam/apps/{int(app_id)}/header.jpg"
    )


def build_game_card_html(row: pd.Series) -> str:
    """Build one escaped recommendation card with a resilient image fallback."""
    app_id = int(row["app_id"])
    rank = int(row["rank"])
    title = escape(str(row.get("title", f"app_id {app_id}")), quote=True)
    model = escape(str(row.get("model", "")), quote=True)
    reason = escape(str(row.get("recommendation_reason", "")), quote=True)
    rating = escape(str(row.get("rating", "정보 없음")), quote=True)
    positive = row.get("positive_ratio")
    price = row.get("price_final")
    positive_text = (
        f"{int(positive)}% 긍정" if pd.notna(positive) else "긍정 비율 정보 없음"
    )
    if pd.isna(price):
        price_text = "가격 정보 없음"
    elif float(price) == 0:
        price_text = "무료"
    else:
        price_text = f"${float(price):.2f}"
    image_url = steam_header_image_url(app_id)
    legacy_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg"
    store_url = f"https://store.steampowered.com/app/{app_id}"
    # If both current and legacy CDN paths fail, the card keeps a neutral
    # gradient background and hides the broken image icon.
    return f"""
    <article class="steam-game-card">
      <div class="steam-image-wrap">
        <img src="{image_url}" data-fallback="{legacy_url}" alt="{title} header image"
             loading="lazy"
             onerror="if(this.dataset.fallback){{this.src=this.dataset.fallback;this.dataset.fallback='';}}
                      else{{this.style.display='none';}}">
        <span class="steam-image-placeholder">Steam 이미지 없음</span>
        <span class="steam-rank">#{rank}</span>
      </div>
      <div class="steam-card-body">
        <a class="steam-title" href="{store_url}" target="_blank" rel="noopener noreferrer">
          {title}
        </a>
        <div class="steam-meta">{rating} · {positive_text} · {price_text}</div>
        {f'<div class="steam-model">{model}</div>' if model else ''}
        {f'<div class="steam-reason">{reason}</div>' if reason else ''}
      </div>
    </article>
    """


def render_game_cards(frame: pd.DataFrame) -> None:
    """Render recommendations as two-column visual cards, grouped by model."""
    st.html(
        """
        <style>
        .steam-game-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            overflow: hidden;
            margin-bottom: 1rem;
            background: rgba(128, 128, 128, 0.06);
            min-height: 100%;
        }
        .steam-image-wrap {
            position: relative;
            width: 100%;
            aspect-ratio: 460 / 215;
            overflow: hidden;
            background: linear-gradient(135deg, #243447, #101820);
        }
        .steam-image-wrap img {
            position: relative;
            z-index: 1;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .steam-image-placeholder {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #b8c5d1;
            font-size: 0.85rem;
        }
        .steam-rank {
            position: absolute;
            z-index: 2;
            top: 0.55rem;
            left: 0.55rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            color: white;
            background: rgba(0, 0, 0, 0.78);
            font-weight: 700;
        }
        .steam-card-body { padding: 0.8rem 0.9rem 0.95rem; }
        .steam-title {
            color: inherit !important;
            font-size: 1.05rem;
            font-weight: 750;
            text-decoration: none !important;
        }
        .steam-title:hover { text-decoration: underline !important; }
        .steam-meta { margin-top: 0.35rem; font-size: 0.84rem; opacity: 0.78; }
        .steam-model {
            display: inline-block;
            margin-top: 0.5rem;
            padding: 0.15rem 0.45rem;
            border-radius: 999px;
            background: rgba(108, 92, 231, 0.15);
            font-size: 0.75rem;
        }
        .steam-reason { margin-top: 0.55rem; font-size: 0.88rem; line-height: 1.42; }
        </style>
        """
    )

    model_groups = list(frame.groupby("model", sort=False)) if "model" in frame else [("추천", frame)]
    tabs = st.tabs([str(name) for name, _ in model_groups]) if len(model_groups) > 1 else None
    for group_index, (name, group) in enumerate(model_groups):
        container = tabs[group_index] if tabs is not None else st.container()
        with container:
            if tabs is None and "model" in frame:
                st.caption(f"추천 모델: {name}")
            ordered = group.sort_values("rank", kind="stable")
            columns = st.columns(2)
            for card_index, (_, row) in enumerate(ordered.iterrows()):
                with columns[card_index % 2]:
                    # st.html renders nested card elements as HTML. st.markdown
                    # can expose a nested <div> as literal text in some versions.
                    st.html(build_game_card_html(row))


def render_results(frame: pd.DataFrame, filename: str) -> None:
    st.success(f"{len(frame):,}개 추천 결과를 생성했습니다.")
    render_game_cards(frame)
    with st.expander("표 형태로 보기"):
        display_columns = [
            column
            for column in [
                "rank", "title", "app_id", "model", "score", "rating", "positive_ratio",
                "price_final", "matched_preferred_tags", "recommendation_reason", "original_rank",
            ]
            if column in frame.columns
        ]
        st.dataframe(
            frame[display_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn("순위", format="%d"),
                "title": "게임",
                "app_id": st.column_config.NumberColumn("app_id", format="%d"),
                "model": "모델",
                "score": st.column_config.NumberColumn("점수", format="%.4f"),
                "rating": "Steam 평가",
                "positive_ratio": st.column_config.NumberColumn("긍정 비율", format="%d%%"),
                "price_final": st.column_config.NumberColumn("가격", format="$%.2f"),
                "matched_preferred_tags": "일치 태그",
                "recommendation_reason": "추천 이유",
                "original_rank": st.column_config.NumberColumn("원래 순위", format="%d"),
            },
        )
    st.download_button(
        "CSV 다운로드",
        data=csv_bytes(frame),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )
    with st.expander("전체 출력 컬럼 보기"):
        st.dataframe(frame, hide_index=True, width="stretch")


def known_user_form(top_k: int, diversity: bool, diversity_lambda: float) -> None:
    st.subheader("기존 사용자 추천")
    st.caption("학습 시점에 positive Train 이력이 있는 49,742명의 사용자용입니다.")
    user_id = st.number_input("사용자 ID", min_value=0, step=1, value=13)
    model_labels = {
        "MF + Multimodal Hybrid (recommended)": "mf_multimodal_hybrid",
        "Multimodal BPR": "multimodal_bpr",
        "Balanced Hybrid": "balanced_hybrid",
        "MF-BPR": "mf_bpr",
        "Text-BPR": "text_bpr",
    }
    selected_labels = st.multiselect(
        "비교할 모델",
        options=list(model_labels),
        default=["MF + Multimodal Hybrid (recommended)"],
    )
    if st.button("기존 사용자 추천받기", type="primary", width="stretch"):
        if not selected_labels:
            st.warning("모델을 하나 이상 선택해 주세요.")
            return
        try:
            pipeline = load_known_pipeline()
            pool_k = top_k * 10 if diversity else top_k
            result = pipeline.recommend(
                [int(user_id)],
                top_k=pool_k,
                models=[model_labels[label] for label in selected_labels],
            )
            if diversity:
                result = mmr_rerank(
                    result,
                    (
                        pipeline.multimodal_bank.numpy()
                        if pipeline.multimodal_bank is not None
                        else pipeline.text_bank.numpy()
                    ),
                    pipeline.app_to_row,
                    top_k=top_k,
                    lambda_relevance=diversity_lambda,
                    group_columns=["user_id", "model"],
                )
        except (KeyError, ValueError, RuntimeError) as error:
            st.error(str(error))
            return
        st.session_state["last_result"] = result
        st.session_state["last_filename"] = f"known_user_{int(user_id)}_top{top_k}.csv"


def new_user_form(
    top_k: int,
    diversity: bool,
    diversity_lambda: float,
    tags: pd.DataFrame,
    game_labels: list[str],
    label_to_id: dict[str, int],
) -> None:
    st.subheader("신규 사용자 추천")
    st.caption("선호 입력이 있으면 MiniLM 콘텐츠 프로필을, 없으면 Train Popularity를 사용합니다.")
    profile_name = st.text_input("프로필 이름", value="new_user_001")
    tag_options = [f"{row.tag} ({int(row.game_count):,}개)" for row in tags.itertuples()]
    tag_display_to_value = dict(zip(tag_options, tags.tag))
    selected_tag_labels = st.multiselect(
        "선호 태그",
        options=tag_options,
        placeholder="RPG, Open World처럼 여러 개 선택",
    )
    selected_games = st.multiselect(
        "좋아하는 게임",
        options=game_labels,
        max_selections=5,
        placeholder="게임명을 검색해 최대 5개 선택",
    )
    content_weight = st.slider(
        "콘텐츠 반영 비중",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="선호 정보가 없으면 이 값과 관계없이 Popularity 100%로 동작합니다.",
    )
    if st.button("신규 사용자 추천받기", type="primary", width="stretch"):
        try:
            pipeline = load_cold_pipeline()
            pool_k = top_k * 10 if diversity else top_k
            result = pipeline.recommend(
                profile_name=profile_name.strip() or "new_user",
                top_k=pool_k,
                preferred_tags=[tag_display_to_value[label] for label in selected_tag_labels],
                liked_app_ids=[label_to_id[label] for label in selected_games],
                content_weight=content_weight,
            )
            if diversity:
                result = mmr_rerank(
                    result,
                    (
                        pipeline.multimodal_items
                        if pipeline.multimodal_items is not None and selected_games
                        else pipeline.text_items
                    ),
                    pipeline.app_to_row,
                    top_k=top_k,
                    lambda_relevance=diversity_lambda,
                )
        except (KeyError, ValueError, RuntimeError) as error:
            st.error(str(error))
            return
        st.session_state["last_result"] = result
        st.session_state["last_filename"] = f"{profile_name.strip() or 'new_user'}_top{top_k}.csv"


def main() -> None:
    st.set_page_config(page_title="Steam 게임 추천 MVP", page_icon="🎮", layout="wide")
    st.title("🎮 Steam 게임 추천 MVP")
    st.write("기존 사용자는 학습된 Hybrid를, 신규 사용자는 선호 태그·게임 기반 Cold-start를 사용합니다.")
    _, tags, game_labels, label_to_id = load_ui_data()

    with st.sidebar:
        st.header("추천 설정")
        user_type = st.radio("사용자 유형", ["신규 사용자", "기존 사용자"])
        top_k = st.slider("추천 게임 수", 5, 30, 10, 5)
        diversity = st.toggle(
            "다양한 게임을 우선 추천",
            value=True,
            help="상위 후보 10배를 대상으로 의미·제목 중복을 줄이는 MMR을 적용합니다.",
        )
        diversity_lambda = st.slider(
            "관련성 유지 비중",
            min_value=0.50,
            max_value=1.00,
            value=0.65,
            step=0.05,
            disabled=not diversity,
            help="1에 가까울수록 원래 순위를, 낮을수록 다양성을 강하게 반영합니다.",
        )
        st.divider()
        st.caption("기준 카탈로그 50,872개 · seed 42 모델")

    if user_type == "기존 사용자":
        known_user_form(top_k, diversity, diversity_lambda)
    else:
        new_user_form(top_k, diversity, diversity_lambda, tags, game_labels, label_to_id)

    if "last_result" in st.session_state:
        st.divider()
        render_results(st.session_state["last_result"], st.session_state["last_filename"])

    st.divider()
    st.caption(
        "추천 이유는 모델 신호를 요약한 휴리스틱 설명입니다. "
        "신규 사용자 입력이 없으면 Train positive popularity로 fallback합니다."
    )


if __name__ == "__main__":
    main()
