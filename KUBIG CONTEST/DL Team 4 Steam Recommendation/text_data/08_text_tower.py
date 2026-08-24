"""
08_text_tower.py - fusion 모델에 꽂는 텍스트 타워

텍스트 파트가 넘기는 것은 두 가지뿐이다.
  1. 사전 계산된 MiniLM 텍스트 임베딩 파일 (emb_text_minilm.npy / .csv)
  2. 이 파일의 TextTower 클래스

Sentence-BERT 계열 encoder는 임베딩 생성 단계에서만 사용한다. fusion 학습 중에는
저장된 임베딩을 MLP로 사영한다.

사용 예
    from text_tower import TextTower, load_text_bank

    bank, id2row = load_text_bank("emb_text_minilm", app_ids=games.app_id.values)
    tower = TextTower(in_dim=384, out_dim=64)

    # 배치의 app_id -> 행 인덱스 -> 임베딩 -> 타워
    z_txt = tower(bank[batch_rows])          # (B, 64), L2 정규화됨
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextTower(nn.Module):
    """사전 계산된 텍스트 임베딩을 fusion용 차원으로 사영한다.

    all-MiniLM-L6-v2의 원래 임베딩 차원은 384이다. image tower와 같은 out_dim을
    쓰면 concat 이후 특정 모달리티가 차원 수 때문에 더 크게 반영되는 일을 줄일 수 있다.

    출력은 L2 정규화한다. 타워별 출력 scale을 맞추기 위한 처리다.
    """

    def __init__(self, in_dim=384, hidden=192, out_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


def load_text_bank(prefix, app_ids=None, device="cpu", fill_missing=True):
    """임베딩 파일을 텐서로 올리고 app_id -> 행 매핑을 만든다.

    app_ids를 주면 그 순서에 맞춰 정렬한다. 텍스트 임베딩이 없는 게임은
    fill_missing=True일 때 전체 평균 벡터로 채운다.

    반환: (bank 텐서 (N, D), app_id -> 행 인덱스 dict)
    """
    emb = np.load(f"{prefix}.npy").astype(np.float32)
    ids = pd.read_csv(f"{prefix}.csv").app_id.values

    if app_ids is None:
        bank, out_ids = emb, ids
    else:
        pos = {a: i for i, a in enumerate(ids)}
        mean_vec = emb.mean(0)
        mean_vec /= max(np.linalg.norm(mean_vec), 1e-12)

        missing = [a for a in app_ids if a not in pos]
        if missing and not fill_missing:
            raise KeyError(f"텍스트 임베딩 없는 app_id {len(missing)}건: {missing[:5]}")

        bank = np.stack([emb[pos[a]] if a in pos else mean_vec for a in app_ids])
        out_ids = np.asarray(app_ids)

        if missing:
            print(f"[text_tower] 텍스트 임베딩 없는 {len(missing)}건을 평균 벡터로 채움: "
                  f"{missing[:5]}")

    return (torch.from_numpy(bank).to(device),
            {a: i for i, a in enumerate(out_ids)})


if __name__ == "__main__":
    # 형태 확인용 스모크 테스트
    tower = TextTower(in_dim=384, out_dim=64)
    z = F.normalize(torch.randn(8, 384), dim=-1)
    out = tower(z)
    print(f"입력 {tuple(z.shape)} -> 출력 {tuple(out.shape)}")
    print(f"출력 행 노름 (모두 1이어야 함): "
          f"{out.norm(dim=-1).min():.4f} ~ {out.norm(dim=-1).max():.4f}")
    print(f"학습 파라미터 수: "
          f"{sum(p.numel() for p in tower.parameters() if p.requires_grad):,}")
