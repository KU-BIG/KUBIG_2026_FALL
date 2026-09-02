"""이미지 열화 함수. PROJECT_v2.md §4.2 — 목표는 의미 정보 감소, 픽셀 엔트로피 증가가 아니다."""

from PIL import Image, ImageFilter


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def downsample_upsample(img: Image.Image, scale: float) -> Image.Image:
    """scale=1.0 → 원본, scale<1.0일수록 강하게 열화."""
    if scale >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def center_crop(img: Image.Image, ratio: float) -> Image.Image:
    """ratio=1.0 → 원본. 분포 이동 없는 통제군(§4.2)."""
    if ratio >= 1.0:
        return img
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float):
    """대조군 전용(§4.2) — 정보 감소가 아니라 픽셀 엔트로피 증가."""
    import numpy as np

    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(noisy)


DEGRADATIONS = {
    "blur": gaussian_blur,
    "downsample": downsample_upsample,
    "crop": center_crop,
    "noise": gaussian_noise,
}


def apply_pixel_mask(img: Image.Image, mask) -> Image.Image:
    """mask==True인 픽셀을 이미지 평균색으로 채운다. mask.shape는 (H, W)."""
    import numpy as np

    arr = np.array(img.convert("RGB"))
    fill = arr.reshape(-1, 3).mean(axis=0).astype(arr.dtype)
    arr = arr.copy()
    arr[mask] = fill
    return Image.fromarray(arr)


def partial_mask(mask, fraction: float, seed: int):
    """mask 안에서 fraction 비율만큼 무작위로 선택한 부분집합 마스크. 강도 스윕용(0→1)."""
    import numpy as np

    idx = np.flatnonzero(mask)
    n = int(round(len(idx) * fraction))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(idx, size=n, replace=False) if n > 0 else idx[:0]
    flat = np.zeros(mask.size, dtype=bool)
    flat[chosen] = True
    return flat.reshape(mask.shape)


def random_area_matched_mask(exclude_mask, n_pixels: int, seed: int):
    """exclude_mask(예: 물체 마스크)를 제외한 영역에서 n_pixels개를 무작위로 골라 마스크를 만든다.
    물체 마스킹과 정확히 같은 면적으로 배경을 마스킹하는 통제 조건용."""
    import numpy as np

    h, w = exclude_mask.shape
    bg_idx = np.flatnonzero(~exclude_mask)
    rng = np.random.default_rng(seed)
    n = min(n_pixels, len(bg_idx))
    chosen = rng.choice(bg_idx, size=n, replace=False)
    flat = np.zeros(h * w, dtype=bool)
    flat[chosen] = True
    return flat.reshape(h, w)
