"""Small deterministic checks for history pooling and the residual initialization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from history_user_tower.experiment import build_prefix_examples, build_profiles
from history_user_tower.inference import HistoryUserEncoder
from history_user_tower.model import HistoryUserTower


def main() -> None:
    bank = np.zeros((3, 64), dtype=np.float32)
    bank[0, 0], bank[1, 1], bank[2, 2] = 1, 1, 1
    frame = pd.DataFrame({
        "user_id": [7, 7, 7], "item_row": [0, 1, 2],
        "hours": [0.0, 3.0, 8.0],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "review_id": [1, 2, 3],
    })
    history, users, target = build_prefix_examples(frame, bank)
    assert history.shape == (2, 64)
    assert users.tolist() == [7, 7] and target.tolist() == [1, 2]
    assert np.allclose(history[0], bank[0])  # zero-hour fallback uses simple mean
    assert history[1, 1] > 0 and history[1, 0] == 0  # only positive log-hour weight remains

    profile = build_profiles(frame, bank)
    assert profile.users.tolist() == [7]
    assert np.isclose(np.linalg.norm(profile.hours[0]), 1)
    model = HistoryUserTower()
    x = torch.from_numpy(profile.hours)
    assert torch.allclose(model(x), x, atol=1e-6)  # zero-init residual

    # Real artifact contract: both history-only and new-user intent return
    # normalized 64D vectors and selected games are excluded from retrieval.
    encoder = HistoryUserEncoder()
    existing = encoder.encode_user([10, 20], [2.0, 20.0])
    onboarding = encoder.encode_user([10], selected_genres=["Action"])
    assert existing.shape == onboarding.shape == (64,)
    assert np.isclose(np.linalg.norm(existing), 1, atol=1e-5)
    assert np.isclose(np.linalg.norm(onboarding), 1, atol=1e-5)
    recommended = encoder.recommend([10], selected_genres=["Action"], top_k=3)
    assert len(recommended) == 3 and 10 not in set(recommended.app_id)
    print("HISTORY_USER_TOWER_TEST_OK")


if __name__ == "__main__":
    main()
