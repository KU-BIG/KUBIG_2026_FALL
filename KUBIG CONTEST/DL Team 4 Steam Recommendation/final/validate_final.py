"""Smoke-test the final cold/warm handoff API."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final.engine import FinalRecommendationEngine


def main() -> None:
    engine = FinalRecommendationEngine()

    cold = engine.recommend_cold_detail(
        preferred_genres=["RPG", "Adventure"],
        liked_game_ids=[292030],
        interest_weight=1.0,
        k=5,
    )
    assert len(cold) == 5
    assert cold.app_id.nunique() == 5
    assert not cold.app_id.eq(292030).any()
    assert cold.score.is_monotonic_decreasing

    user_id = next(iter(engine.user_to_idx))
    warm = engine.recommend_warm_detail(user_id=user_id, interest_weight=0.6, k=5)
    assert len(warm) == 5
    assert warm.app_id.nunique() == 5
    assert warm.score.is_monotonic_decreasing

    print("cold_app_ids=", engine.recommend_cold(["RPG"], [292030], 1.0, 5))
    print("warm_user_id=", user_id)
    print("warm_app_ids=", engine.recommend_warm(user_id, 0.6, 5))
    print("FINAL_RECOMMENDER_OK")


if __name__ == "__main__":
    main()
