"""
07_image_tower.py — fusion 모델에 꽂는 이미지 타워

이미지 파트가 넘기는 것은 두 가지뿐이다.
  1. 사전 계산된 CLIP 임베딩 파일 (emb_clip_squash.npy / .csv)
  2. 이 파일의 ImageTower 클래스

CLIP 인코더는 동결이다. 이미 임베딩으로 뽑아 두었으므로 fusion 학습 중에
이미지를 다시 forward할 일이 없고, 학습되는 것은 ImageTower의 MLP뿐이다.
덕분에 fusion 학습 한 스텝의 비용이 정형/텍스트 타워와 같은 수준이 된다.

사용 예
    from image_tower import ImageTower, load_image_bank

    bank, id2row = load_image_bank("emb_clip_squash", app_ids=games.app_id.values)
    tower = ImageTower(in_dim=512, out_dim=64)

    # 배치의 app_id -> 행 인덱스 -> 임베딩 -> 타워
    z_img = tower(bank[batch_rows])          # (B, 64), L2 정규화됨
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTower(nn.Module):
    """사전 계산된 CLIP 임베딩을 fusion용 차원으로 사영한다.

    out_dim은 텍스트 타워와 같은 값으로 맞출 것. 차원이 다르면 concat 이후
    특정 모달리티가 벡터 길이를 더 많이 차지해 기여도가 왜곡된다.

    출력을 L2 정규화하는 이유도 같다. 세 타워의 출력 노름이 제각각이면
    concat 벡터에서 노름이 큰 모달리티가 자동으로 더 큰 목소리를 낸다.
    """

    def __init__(self, in_dim=512, hidden=256, out_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


def load_image_bank(prefix, app_ids=None, device="cpu", fill_missing=True):
    """임베딩 파일을 텐서로 올리고 app_id -> 행 매핑을 만든다.

    app_ids를 주면 그 순서에 맞춰 정렬한다. 이미지가 없는 게임은
    fill_missing=True일 때 전체 평균 벡터로 채운다. 영벡터를 쓰지 않는 이유는
    정규화된 공간에서 원점이 '모든 것과 무관'이라는 잘못된 의미를 갖기 때문이다.

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
            raise KeyError(f"이미지 없는 app_id {len(missing)}건: {missing[:5]}")

        bank = np.stack([emb[pos[a]] if a in pos else mean_vec for a in app_ids])
        out_ids = np.asarray(app_ids)

        if missing:
            print(f"[image_tower] 이미지 없는 {len(missing)}건을 평균 벡터로 채움: "
                  f"{missing}")

    return (torch.from_numpy(bank).to(device),
            {a: i for i, a in enumerate(out_ids)})


if __name__ == "__main__":
    # 형태 확인용 스모크 테스트
    tower = ImageTower(in_dim=512, out_dim=64)
    z = F.normalize(torch.randn(8, 512), dim=-1)
    out = tower(z)
    print(f"입력 {tuple(z.shape)} -> 출력 {tuple(out.shape)}")
    print(f"출력 행 노름 (모두 1이어야 함): "
          f"{out.norm(dim=-1).min():.4f} ~ {out.norm(dim=-1).max():.4f}")
    print(f"학습 파라미터 수: "
          f"{sum(p.numel() for p in tower.parameters() if p.requires_grad):,}")
