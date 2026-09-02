"""공용 시각화 스타일 — PPT 템플릿(버건디 #900020 액센트, 화이트/블랙 베이스) 톤 통일.

import하는 순간 matplotlib rcParams가 전역 적용된다(side effect). outputs/figures/의 모든
plotting 함수는 색상 리터럴("tab:blue" 등)이나 figsize 숫자, 제목 문자열을 직접 쓰지 말고
이 모듈의 COLORS / FIGSIZES / combo_display_name() / save_figure() / place_legend_outside()를
재사용해야 한다 — 새 그림을 추가할 때도 마찬가지. 자세한 사용법은
outputs/figures/STYLE_GUIDE.md 참고.

JSON 결과 수치와는 무관한 순수 표현(presentation) 레이어다 — 이 모듈은 어떤 계산도 하지 않는다.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "primary": "#900020",           # 버건디 — 강조/true/target
    "secondary": "#2E2E38",         # 차콜 — 대조군 2번째 계열
    "tertiary": "#C9A227",          # 머스타드 골드 — 3계열 필요시
    "neutral": "#9B9B9B",           # 회색 — 기준선/wrong pairs/baseline
    "sequential_light": "#F3D9DF",  # 연한 로즈 (시퀀셜 그라데이션 시작)
    "sequential_dark": "#900020",   # 시퀀셜 그라데이션 끝 (primary와 동일)
}

# family별 표준 figsize 슬롯 — 같은 슬롯끼리는 픽셀 치수가 완전히 동일해야 PPT에서 나란히
# 놓았을 때 정렬된다 (figsize x savefig.dpi = 픽셀). bbox_inches='tight'를 쓰지 않는 이유이기도
# 하다 — 'tight'는 제목/범례 길이에 따라 최종 캔버스 크기가 미세하게 흔들린다.
FIGSIZES = {
    "scatter_square": (7, 6.8),     # PCA/UMAP
    "single_panel": (8, 6),         # temperature landscape, pair_margin hist류, label distribution, classification
    "dual_panel_wide": (16, 5.5),   # step7c retrieval (I->T / T->I) -- 서브플롯마다 우측 바깥 범례 자리 확보용으로 폭을 넓힘
    "grid_2x2": (11, 8),            # step7a shift diagnostics
    "heatmap_square": (9, 8),       # confusion matrix
}

BURGUNDY_CMAP = LinearSegmentedColormap.from_list(
    "burgundy_sequential", [COLORS["sequential_light"], COLORS["sequential_dark"]],
)


def sequential_colors(n: int) -> list:
    """시퀀셜 그라데이션(연한 로즈 -> 버건디)에서 n개 색을 균등 간격으로 뽑는다.
    예: temperature landscape의 tau 6단계."""
    return [BURGUNDY_CMAP(x) for x in np.linspace(0.0, 1.0, n)]


def _pick_available_font() -> str:
    from matplotlib import font_manager

    candidates = ["Pretendard", "Malgun Gothic", "NanumGothic", "Nanum Gothic", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "sans-serif"


mpl.rcParams.update({
    "font.family": _pick_available_font(),
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "axes.unicode_minus": False,  # Malgun Gothic 등 CJK 폰트가 유니코드 마이너스 글리프를 안 갖고 있어 깨짐 방지
})


def combo_display_name(key: str, treated: bool = False, suffix: str = "") -> str:
    """콤보 키("stage_a_visual" 등) -> 표시명 매핑, 전 그림에서 재사용.

    stage_a_visual -> "Stage A: image vs visual"
    stage_a_contextual -> "Stage A: image vs contextual"
    combo_display_name("stage_b_visual", treated=True) -> "Stage B (treated-only): image vs visual"
    (treated-only 서브셋은 combos 딕셔너리 키 자체가 "stage_b_visual"과 동일해서 키만으로는
    구분이 안 되므로 별도 인자로 받는다 — 키 문자열 파싱으로 추정하지 않는다.)
    suffix가 있으면 뒤에 괄호로 덧붙인다 (예: suffix="shift sweep").
    """
    parts = key.split("_")
    stage = parts[1].upper()
    condition = parts[2] if len(parts) > 2 else ""
    stage_label = f"Stage {stage}" + (" (treated-only)" if treated else "")
    name = f"{stage_label}: image vs {condition}"
    return f"{name} ({suffix})" if suffix else name


def new_figure(slot: str, **kwargs):
    """FIGSIZES 슬롯 크기로 fig, ax(또는 axes)를 생성. kwargs는 plt.subplots()에 그대로 전달."""
    return plt.subplots(figsize=FIGSIZES[slot], **kwargs)


def save_figure(fig, output_path) -> None:
    """고정 figsize + rcParams의 savefig.dpi(200)로 저장 — bbox_inches='tight' 미사용.
    (fig.tight_layout()로 서브플롯 내부 여백을 조정하는 건 괜찮다 — 캔버스 자체 크기는
    그대로 유지되므로 슬롯이 흔들리지 않는다.)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def place_legend_outside(ax, right_margin: float = 0.72) -> None:
    """단일 축 플롯의 범례를 우측 바깥 고정 위치에 놓는다 (loc='best' 자동배치 금지).
    축 영역을 right_margin까지 줄여 범례가 고정 캔버스 안에 온전히 들어오게 한다."""
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    ax.figure.subplots_adjust(right=right_margin)
