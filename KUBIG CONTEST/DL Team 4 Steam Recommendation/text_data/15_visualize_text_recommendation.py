"""
15_visualize_text_recommendation.py - 입력 user_id별 추천 결과 HTML 시각화

특정 user_id에 대해 다음을 하나의 로컬 HTML로 만든다.
- 입력값: user_id와 train positive history
- 처리 결과: user vector 주변의 2D embedding map
- 출력값: Top-K 추천 카드와 추천 이유

예시:
python 15_visualize_text_recommendation.py --user-id 58 --top-k 10
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ARTIFACT_DIR = BASE_DIR / "text_recommender_artifacts"
RECOMMENDER_SCRIPT = BASE_DIR / "14_recommend_text_only.py"

spec = importlib.util.spec_from_file_location("text_recommender", RECOMMENDER_SCRIPT)
recommender = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(recommender)


def load_user_history(artifact_dir: Path, user_id: int, history_k: int) -> pd.DataFrame:
    history_path = artifact_dir / "user_train_positive_history.csv"
    usecols = ["user_id", "app_id", "title", "embedding_row", "hours", "hours_weight"]
    history = pd.read_csv(history_path, usecols=usecols)
    history = history.loc[history["user_id"].eq(user_id)].copy()
    history = history.sort_values("hours_weight", ascending=False).head(history_k)
    return history


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = max((x_max - x_min) * 0.08, 1e-6)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    out = coords.copy()
    out[:, 0] = (x - (x_min - x_pad)) / ((x_max + x_pad) - (x_min - x_pad))
    out[:, 1] = 1.0 - (y - (y_min - y_pad)) / ((y_max + y_pad) - (y_min - y_pad))
    return out


def make_map_points(
    item_emb: np.ndarray,
    user_vector: np.ndarray,
    item_catalog: pd.DataFrame,
    recommendations: pd.DataFrame,
    history: pd.DataFrame,
    background_sample: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    important_rows = set(recommendations["embedding_row"].astype(int).tolist())
    important_rows.update(history["embedding_row"].astype(int).tolist())

    candidate_rows = item_catalog.loc[~item_catalog["embedding_row"].isin(important_rows), "embedding_row"].to_numpy()
    sample_size = min(background_sample, len(candidate_rows))
    sampled_rows = rng.choice(candidate_rows, size=sample_size, replace=False)

    vectors = []
    raw_points = []

    background = item_catalog.set_index("embedding_row").loc[sampled_rows].reset_index()
    for row in background.itertuples(index=False):
        emb_row = int(row.embedding_row)
        vectors.append(item_emb[emb_row])
        raw_points.append(
            {
                "kind": "catalog sample",
                "app_id": int(row.app_id),
                "title": str(row.title_clean),
                "score": None,
                "hours": None,
                "reason": "",
                "tags": str(row.tags_text)[:160],
                "is_cold_train": bool(row.is_cold_train),
                "train_interactions": int(row.train_interactions),
            }
        )

    for row in history.itertuples(index=False):
        emb_row = int(row.embedding_row)
        vectors.append(item_emb[emb_row])
        raw_points.append(
            {
                "kind": "liked history",
                "app_id": int(row.app_id),
                "title": str(row.title),
                "score": None,
                "hours": float(row.hours),
                "reason": "input history",
                "tags": "",
                "is_cold_train": False,
                "train_interactions": None,
            }
        )

    for row in recommendations.itertuples(index=False):
        emb_row = int(row.embedding_row)
        vectors.append(item_emb[emb_row])
        raw_points.append(
            {
                "kind": "recommendation",
                "rank": int(row.rank),
                "app_id": int(row.app_id),
                "title": str(row.title_clean),
                "score": float(row.score),
                "hours": None,
                "reason": str(row.reason),
                "tags": str(row.tags_text)[:160],
                "is_cold_train": bool(row.is_cold_train),
                "train_interactions": int(row.train_interactions),
            }
        )

    vectors.append(user_vector)
    raw_points.append(
        {
            "kind": "user vector",
            "app_id": None,
            "title": "user taste vector",
            "score": None,
            "hours": None,
            "reason": "weighted mean of liked history",
            "tags": "",
            "is_cold_train": False,
            "train_interactions": None,
        }
    )

    coords = normalize_coords(pca_2d(np.vstack(vectors)))
    for point, coord in zip(raw_points, coords):
        point["x"] = round(float(coord[0]), 5)
        point["y"] = round(float(coord[1]), 5)
    return raw_points


def truncate_text(value: object, max_len: int) -> str:
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def build_html(
    user_id: int,
    recommendations: pd.DataFrame,
    history: pd.DataFrame,
    points: list[dict[str, object]],
) -> str:
    data_json = json.dumps(points, ensure_ascii=False)
    rec_cards = []
    for row in recommendations.itertuples(index=False):
        cold_label = "cold item" if bool(row.is_cold_train) else f"train interactions {int(row.train_interactions):,}"
        rec_cards.append(
            f"""
            <article class="rec-card">
              <div class="rank">#{int(row.rank)}</div>
              <div>
                <h3>{html.escape(str(row.title_clean))}</h3>
                <p class="meta">score {float(row.score):.4f} · {html.escape(cold_label)}</p>
                <p class="reason">{html.escape(str(row.reason))}</p>
                <p class="tags">{html.escape(truncate_text(row.tags_text, 180))}</p>
              </div>
            </article>
            """
        )

    history_items = []
    for row in history.itertuples(index=False):
        history_items.append(
            f"""
            <li>
              <span>{html.escape(str(row.title))}</span>
              <strong>{float(row.hours):.1f}h</strong>
            </li>
            """
        )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Text Recommendation Map - User {user_id}</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --surface: #ffffff;
      --ink: #17202a;
      --muted: #657084;
      --line: #d8dde8;
      --catalog: #c2c8d3;
      --history: #2f6fdd;
      --rec: #f28e2b;
      --user: #2ca58d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-end;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.15;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .pill {{
      border: 1px solid var(--line);
      background: var(--surface);
      padding: 8px 10px;
      font-size: 13px;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.85fr);
      gap: 18px;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      padding: 18px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 17px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 5px;
      vertical-align: -1px;
      border-radius: 50%;
    }}
    .map-wrap {{
      position: relative;
      min-height: 520px;
      border: 1px solid var(--line);
      background: #fbfcff;
    }}
    svg {{
      width: 100%;
      height: 520px;
      display: block;
    }}
    .tooltip {{
      position: absolute;
      z-index: 3;
      display: none;
      max-width: 280px;
      padding: 10px 12px;
      background: #111827;
      color: #fff;
      font-size: 12px;
      pointer-events: none;
    }}
    .tooltip strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
    }}
    .history ul {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 8px;
    }}
    .history li {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
      font-size: 13px;
    }}
    .history strong {{
      color: var(--history);
      white-space: nowrap;
    }}
    .rec-list {{
      display: grid;
      gap: 10px;
      max-height: 820px;
      overflow: auto;
      padding-right: 4px;
    }}
    .rec-card {{
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 10px;
      border: 1px solid var(--line);
      padding: 12px;
      background: #fff;
    }}
    .rank {{
      color: var(--rec);
      font-weight: 800;
      font-size: 18px;
    }}
    h3 {{
      margin: 0 0 4px;
      font-size: 14px;
    }}
    .meta, .reason, .tags {{
      margin: 0;
      font-size: 12px;
      line-height: 1.35;
    }}
    .meta {{ color: var(--muted); }}
    .reason {{ margin-top: 6px; color: #394150; }}
    .tags {{ margin-top: 6px; color: var(--muted); }}
    .panel-stack {{
      display: grid;
      gap: 18px;
    }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      header {{ display: block; }}
      .pill {{ display: inline-block; margin-top: 12px; }}
      .grid {{ grid-template-columns: 1fr; }}
      svg {{ height: 440px; }}
      .map-wrap {{ min-height: 440px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Text-only Recommendation Demo</h1>
        <p class="subtitle">입력 user_id의 liked history를 weighted mean으로 만들고, 전체 game embedding과 비교한 결과</p>
      </div>
      <div class="pill">input user_id: {user_id}</div>
    </header>

    <div class="grid">
      <section>
        <h2>Embedding Map</h2>
        <div class="legend">
          <span><i class="dot" style="background: var(--catalog)"></i>catalog sample</span>
          <span><i class="dot" style="background: var(--history)"></i>liked history</span>
          <span><i class="dot" style="background: var(--rec)"></i>recommendation</span>
          <span><i class="dot" style="background: var(--user)"></i>user vector</span>
        </div>
        <div class="map-wrap">
          <svg id="map" viewBox="0 0 760 520" role="img" aria-label="2D embedding map"></svg>
          <div id="tooltip" class="tooltip"></div>
        </div>
      </section>

      <div class="panel-stack">
        <section class="history">
          <h2>Input History</h2>
          <ul>
            {"".join(history_items)}
          </ul>
        </section>
        <section>
          <h2>Output Recommendations</h2>
          <div class="rec-list">
            {"".join(rec_cards)}
          </div>
        </section>
      </div>
    </div>
  </main>

  <script>
    const points = {data_json};
    const svg = document.getElementById("map");
    const tooltip = document.getElementById("tooltip");
    const width = 760;
    const height = 520;
    const pad = 28;
    const color = {{
      "catalog sample": "var(--catalog)",
      "liked history": "var(--history)",
      "recommendation": "var(--rec)",
      "user vector": "var(--user)"
    }};
    const radius = {{
      "catalog sample": 3,
      "liked history": 7,
      "recommendation": 7,
      "user vector": 9
    }};

    function x(v) {{ return pad + v * (width - pad * 2); }}
    function y(v) {{ return pad + v * (height - pad * 2); }}

    const frame = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    frame.setAttribute("x", pad);
    frame.setAttribute("y", pad);
    frame.setAttribute("width", width - pad * 2);
    frame.setAttribute("height", height - pad * 2);
    frame.setAttribute("fill", "none");
    frame.setAttribute("stroke", "var(--line)");
    svg.appendChild(frame);

    points.forEach((point) => {{
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x(point.x));
      circle.setAttribute("cy", y(point.y));
      circle.setAttribute("r", radius[point.kind] || 4);
      circle.setAttribute("fill", color[point.kind] || "var(--catalog)");
      circle.setAttribute("opacity", point.kind === "catalog sample" ? "0.55" : "0.95");
      circle.setAttribute("stroke", point.kind === "catalog sample" ? "none" : "#ffffff");
      circle.setAttribute("stroke-width", point.kind === "user vector" ? "2.5" : "1.5");
      circle.tabIndex = 0;
      circle.addEventListener("mousemove", (event) => showTooltip(event, point));
      circle.addEventListener("mouseleave", hideTooltip);
      circle.addEventListener("focus", (event) => showTooltip(event, point));
      circle.addEventListener("blur", hideTooltip);
      svg.appendChild(circle);

      if (point.kind === "recommendation") {{
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", x(point.x) + 9);
        label.setAttribute("y", y(point.y) + 4);
        label.setAttribute("font-size", "11");
        label.setAttribute("fill", "var(--ink)");
        label.textContent = "#" + point.rank;
        svg.appendChild(label);
      }}
    }});

    function showTooltip(event, point) {{
      const score = point.score == null ? "" : `<br>score: ${{Number(point.score).toFixed(4)}}`;
      const hours = point.hours == null ? "" : `<br>hours: ${{Number(point.hours).toFixed(1)}}`;
      const train = point.train_interactions == null ? "" : `<br>train interactions: ${{point.train_interactions}}`;
      const cold = point.is_cold_train ? "<br>cold item: yes" : "";
      const reason = point.reason ? `<br>${{escapeHtml(point.reason)}}` : "";
      tooltip.innerHTML = `<strong>${{escapeHtml(point.title)}}</strong>${{escapeHtml(point.kind)}}${{score}}${{hours}}${{train}}${{cold}}${{reason}}`;
      const box = event.currentTarget.getBoundingClientRect();
      const parent = tooltip.parentElement.getBoundingClientRect();
      tooltip.style.left = Math.min(box.left - parent.left + 12, parent.width - 290) + "px";
      tooltip.style.top = Math.max(box.top - parent.top - 8, 8) + "px";
      tooltip.style.display = "block";
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--reason-history-k", type=int, default=30)
    parser.add_argument("--background-sample", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    item_emb, user_vectors, item_catalog, user_index, user_seen = recommender.load_artifacts(args.artifact_dir)
    recommendations = recommender.recommend_for_user(
        user_id=args.user_id,
        top_k=args.top_k,
        item_emb=item_emb,
        user_vectors=user_vectors,
        item_catalog=item_catalog,
        user_index=user_index,
        user_seen=user_seen,
        exclude_seen=True,
    )
    recommendations = recommender.add_recommendation_reasons(
        recommendations,
        args.user_id,
        item_emb,
        args.artifact_dir,
        args.reason_history_k,
    )

    history = load_user_history(args.artifact_dir, args.user_id, args.history_k)
    user_row = int(user_index.loc[user_index["user_id"].eq(args.user_id)].iloc[0]["row"])
    points = make_map_points(
        item_emb=item_emb,
        user_vector=user_vectors[user_row],
        item_catalog=item_catalog,
        recommendations=recommendations,
        history=history,
        background_sample=args.background_sample,
        seed=args.seed,
    )

    output_html = args.output_html
    if output_html is None:
        output_html = args.artifact_dir / f"recommend_user_{args.user_id}_embedding_map.html"
    if not output_html.is_absolute():
        output_html = BASE_DIR / output_html
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(build_html(args.user_id, recommendations, history, points), encoding="utf-8")

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = args.artifact_dir / f"recommend_user_{args.user_id}_top{args.top_k}_with_reasons.csv"
    if not output_csv.is_absolute():
        output_csv = BASE_DIR / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output_csv, index=False)

    print(f"HTML -> {output_html}")
    print(f"CSV -> {output_csv}")


if __name__ == "__main__":
    main()
