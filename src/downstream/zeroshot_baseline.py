"""Downstream task 유효성 검증용 baseline zero-shot classification — 설계 결정용 사전 확인, 정식 Step 아님.

label_check.py에서 type/school을 라벨 후보로 잠정 확정했지만, "이미지에서 실제로 관측 가능한
속성인가(=CLIP이 조금이라도 정확도를 낼 수 있는가)"는 아직 확인하지 않았다. 이걸 확인 없이
gap 스윕(embedding shift)까지 가면, 나중에 정확도 변화가 신호인지 노이즈인지 구분할 수 없다.

frozen CLIP 그대로(개입 없음, λ=0에 해당) 이미지 임베딩 vs 클래스명 프롬프트 텍스트 임베딩의
코사인 유사도로 top-1 예측만 본다 — 학습/fine-tuning 없음, 순수 조회. outputs/embeddings/*.npz의
기존 이미지 임베딩을 그대로 재사용하고, 클래스명 프롬프트만 frozen CLIP 텍스트 인코더에 새로
통과시킨다 — 기존 캡션 임베딩 재추출이 아니라 이 진단 전용의 새로운 소규모 텍스트 인코딩이므로
"임베딩 재추출 금지" 규칙과 무관하다. 프롬프트 수가 적어 CPU로 충분 — device="cpu" 고정
(CLAUDE.md: GPU는 src/embeddings/extract.py에서만).

*** type 재검증 (2차) ***
1차 baseline(단일 템플릿)에서 genre 클래스만 0.5%로 나왔는데, genre에만 여러 프롬프트를
추가로 시도하는 건 결과를 본 후 특정 클래스에만 개입하는 선택 편향이라 기각됐다. 대신:
- 작업 A: 여러 템플릿을 10개 클래스 **전부에 동일하게** 적용해 평균한 앙상블 임베딩으로
  재분류 (CLIP/Fahim et al. 스타일 prompt ensembling) — "single_prompt" 결과는 남겨두고
  "ensemble_prompt" 결과를 나란히 추가.
- 작업 B: 오분류 패턴 자체는 관찰만 한다 (confusion matrix) — 원인을 단정하지 않음.

*** type 최종 클래스셋 확정 (3차) ***
- FACT: 최종 클래스셋은 전체 표본(9,356) 대비 비율이 MIN_PREVALENCE_RATIO(1%) 미만인 클래스만
  기계적으로 제외한다. 이 규칙은 어떤 컬럼/클래스의 정확도 결과를 보기 전에 정한 사전(prior) 기준이며,
  특정 클래스(예: 정확도가 낮았던 genre)를 겨냥해 만든 것이 아니다 — 정확도는 이 결정에 전혀
  관여하지 않는다. genre는 표본 비율이 1%를 훨씬 넘으므로(778/9356 ≈ 8.3%) 이 필터로 제외되지
  않고, 정확도가 낮다는 사실도 그대로 결과에 남는다 (notes.genre_low_accuracy_retained_by_design).
- label_check.py의 절대값(<10) rare-class 판정을 find_rare_classes()로 일반화해 비율(min_ratio)
  파라미터로도 쓸 수 있게 재사용한다 (새로 만들지 않음).
- "final_prevalence_filtered" 결과가 Step 7b(embedding shift zero-shot 스윕)에서 실제로 쓸
  최종 클래스셋/설정이다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.downstream.label_check import find_rare_classes
from src.embeddings.extract import MODEL_NAME, encode_texts, load_model
from src.metrics.gap import _relpath
from src.viz.style import BURGUNDY_CMAP, new_figure, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "clip_vitb32_embeddings.npz"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "semart_v1_modality_gap_dataset.parquet"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "downstream_zeroshot_baseline.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

DEVICE = "cpu"

# 컬럼별 단일 프롬프트 템플릿 (PROPOSAL — CLIP 원 논문의 "a photo of a {}." / Mind the Gap 스타일을
# SemArt(회화) 도메인에 맞게 변형). 문구를 고른 이유는 결과 JSON에도 prompt_rationale로 남긴다.
PROMPT_TEMPLATES = {
    # type 라벨(religious/portrait/landscape/...)은 그림이 "묘사하는 대상"이므로 depicting이 자연스러움.
    "type": "a painting depicting {}.",
    # school 라벨(Italian/Dutch/Flemish/...)은 화파/국적 귀속이라 미술사에서 관용적으로 쓰는
    # "the {} school" 표현을 따름 — "a painting of {}"처럼 대상을 묘사하는 것으로 오독되지 않게.
    "school": "a painting from the {} school.",
}
PROMPT_RATIONALE = {
    "type": (
        "'a painting depicting {}.' -- type labels (religious/portrait/landscape/...) describe what "
        "the painting depicts, so 'depicting' phrasing mirrors CLIP's original 'a photo of a {}.' "
        "template adapted to painting subject matter."
    ),
    "school": (
        "'a painting from the {} school.' -- school labels (Italian/Dutch/Flemish/...) are an "
        "art-historical school/nationality attribution, not depicted content, so this follows the "
        "conventional art-historical phrase 'the X school' rather than treating the label as a "
        "depicted subject (which 'a painting of {}.' would incorrectly imply)."
    ),
}

# 앙상블 프롬프트 템플릿 — type 컬럼 전용, 10개 클래스 전부에 동일하게 적용 (특정 클래스만
# 골라 템플릿을 바꾸는 건 결과를 본 후 개입하는 선택 편향이라 팀이 기각했음 -- 반드시 전 클래스
# 동일 처리를 유지할 것). "a painting."은 CLIP 원 논문의 제네릭 anchor 템플릿에 해당하며
# {} 자리표시자가 없어 모든 클래스에 동일한 상수 벡터로 기여한다 (특정 클래스 편애 아님).
ENSEMBLE_PROMPT_TEMPLATES = {
    "type": [
        "a painting.",
        "a painting depicting {}.",
        "a {} painting.",
        "an artwork in the {} genre.",
    ],
}
ENSEMBLE_PROMPT_RATIONALE = {
    "type": (
        "CLIP/Fahim et al. style prompt ensembling: encode each of the 4 templates below for every "
        "class (no class-specific template selection), L2-normalize each per-template embedding, "
        "average across templates per class, then re-normalize -- this is the standard 'prompt "
        "ensembling' technique from the CLIP paper, applied identically to all 10 classes."
    ),
}

# "학습 가능(informative)" 판정 임계값 (PROPOSAL, 잠정치) — micro_accuracy_above_majority가
# 이보다 크면 이미지가 다수결보다 유의미하게 더 기여한다고 본다. 통계적 유의성 검정이 아니라
# 실무적 컷오프이므로, 최종 판단은 팀이 실제 수치(특히 per_class_accuracy)를 보고 내려야 한다.
MEANINGFUL_DIFF_THRESHOLD = 0.05  # 5%p

# confusion matrix에서 오분류 패턴을 관찰할 대상 클래스 (1차 baseline에서 정확도가 가장 낮았던 genre).
GENRE_MISCLASSIFICATION_TARGET = "genre"

# 최종 클래스셋 확정 기준 (FACT — 정확도 결과를 보기 전에 정한 사전 기준, 특정 클래스를
# 겨냥한 것이 아님): 전체 표본 대비 비율이 이 값 미만인 클래스만 기계적으로 제외한다.
MIN_PREVALENCE_RATIO = 0.01  # 1%


def _majority_and_chance(labels: np.ndarray, n_classes: int) -> tuple[float, float]:
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    majority_baseline_accuracy = float(counts.max()) / n
    chance_accuracy = 1.0 / n_classes
    return majority_baseline_accuracy, chance_accuracy


def _per_class_accuracy(labels: np.ndarray, correct: np.ndarray, class_names: list[str]) -> dict:
    per_class = {}
    for name in class_names:
        mask = labels == name
        n_in_class = int(mask.sum())
        per_class[name] = {
            "n": n_in_class,
            "accuracy": float(correct[mask].mean()) if n_in_class > 0 else None,
        }
    return per_class


def _macro_accuracy(per_class_accuracy: dict) -> float | None:
    accs = [v["accuracy"] for v in per_class_accuracy.values() if v["accuracy"] is not None]
    return float(np.mean(accs)) if accs else None


def compute_zeroshot_baseline(
    image_emb: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    prompt_template: str,
    model=None,
    processor=None,
) -> dict:
    """frozen CLIP 그대로(개입 없음)의 단일 템플릿 baseline zero-shot classification.

    class_names 각각을 prompt_template.format(class_name)으로 감싸 frozen CLIP 텍스트
    인코더에 통과시키고, image_emb와의 코사인 유사도 argmax로 예측한다 (둘 다 L2 정규화되어
    있으므로 내적 = 코사인). micro_accuracy(표본 단위 평균)와 macro_accuracy(클래스별 accuracy의
    단순평균, 클래스 크기 불균형 영향을 안 받음) 둘 다 계산한다.

    반환 dict는 JSON 직렬화 가능한 필드 + "_raw"(pred_labels, confusion matrix용 — 호출부가
    pop해서 JSON 저장 전에 제거해야 함)로 구성된다.
    """
    if model is None or processor is None:
        model, processor = load_model(DEVICE)

    prompts = [prompt_template.format(name) for name in class_names]
    text_emb = encode_texts(prompts, model, processor, batch_size=len(prompts))

    sims = image_emb @ text_emb.T  # (n, n_classes)
    pred_idx = sims.argmax(axis=1)
    class_names_arr = np.array(class_names)
    pred_labels = class_names_arr[pred_idx]

    labels = np.asarray(labels)
    correct = pred_labels == labels
    micro_accuracy = float(correct.mean())

    n = len(labels)
    n_classes = len(class_names)
    majority_baseline_accuracy, chance_accuracy = _majority_and_chance(labels, n_classes)
    per_class_accuracy = _per_class_accuracy(labels, correct, class_names)
    macro_accuracy = _macro_accuracy(per_class_accuracy)

    return {
        "prompt_template": prompt_template,
        "n": n,
        "n_classes": n_classes,
        "micro_accuracy": micro_accuracy,
        "macro_accuracy": macro_accuracy,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "chance_accuracy": chance_accuracy,
        "micro_accuracy_above_majority": micro_accuracy - majority_baseline_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "_raw": {"pred_labels": pred_labels},
    }


def compute_zeroshot_baseline_ensemble(
    image_emb: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    prompt_templates: list[str],
    model=None,
    processor=None,
) -> dict:
    """CLIP/Fahim et al. 스타일 prompt ensembling — 여러 템플릿을 모든 클래스에 동일하게 적용.

    각 템플릿마다 class_names 전체를 인코딩(L2 정규화됨), 템플릿 축으로 평균한 뒤 재정규화해
    클래스당 하나의 앙상블 임베딩을 만든다 (표준 CLIP prompt-ensembling 절차). 특정 클래스만
    다른 템플릿을 쓰는 일은 없다 — 어떤 클래스도 특별 취급하지 않는다.
    """
    if model is None or processor is None:
        model, processor = load_model(DEVICE)

    template_embs = []
    for template in prompt_templates:
        prompts = [template.format(name) for name in class_names]
        text_emb = encode_texts(prompts, model, processor, batch_size=len(prompts))
        template_embs.append(text_emb)

    stacked = np.stack(template_embs, axis=0)  # (n_templates, n_classes, d)
    class_emb = stacked.mean(axis=0)
    class_emb = class_emb / np.linalg.norm(class_emb, axis=1, keepdims=True)

    sims = image_emb @ class_emb.T
    pred_idx = sims.argmax(axis=1)
    class_names_arr = np.array(class_names)
    pred_labels = class_names_arr[pred_idx]

    labels = np.asarray(labels)
    correct = pred_labels == labels
    micro_accuracy = float(correct.mean())

    n = len(labels)
    n_classes = len(class_names)
    majority_baseline_accuracy, chance_accuracy = _majority_and_chance(labels, n_classes)
    per_class_accuracy = _per_class_accuracy(labels, correct, class_names)
    macro_accuracy = _macro_accuracy(per_class_accuracy)

    return {
        "prompt_templates": list(prompt_templates),
        "n": n,
        "n_classes": n_classes,
        "micro_accuracy": micro_accuracy,
        "macro_accuracy": macro_accuracy,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "chance_accuracy": chance_accuracy,
        "micro_accuracy_above_majority": micro_accuracy - majority_baseline_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "_raw": {"pred_labels": pred_labels},
    }


def _apply_verdict(result: dict) -> dict:
    if result["micro_accuracy_above_majority"] > MEANINGFUL_DIFF_THRESHOLD:
        verdict = "informative"
    else:
        verdict = "not_informative"
    result["verdict"] = verdict
    result["verdict_threshold"] = MEANINGFUL_DIFF_THRESHOLD
    result["verdict_note"] = (
        "This threshold (5 percentage points) is a provisional practical cutoff, not a "
        "statistical significance test. The team should make the final call after reviewing "
        "the actual micro_accuracy_above_majority value and the per-class accuracy breakdown, "
        "especially for borderline cases."
    )
    return result


def compute_confusion_matrix(labels: np.ndarray, pred_labels: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """실제 클래스 x 예측 클래스 confusion matrix (class_names 순서로 정렬된 정방 DataFrame)."""
    cm = pd.crosstab(
        pd.Series(labels, name="actual"), pd.Series(pred_labels, name="predicted"),
    )
    return cm.reindex(index=class_names, columns=class_names, fill_value=0)


def _plot_confusion_matrix(cm: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = new_figure("heatmap_square")
    im = ax.imshow(cm.values, cmap=BURGUNDY_CMAP)
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)

    vmax = cm.values.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm.values[i, j]
            color = "white" if val > vmax * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()

    save_figure(fig, output_path)


def summarize_misclassification(cm: pd.DataFrame, true_class: str, top_n: int = 3) -> dict:
    """true_class 행에서 어디로 오분류됐는지 관찰만 기록 — 원인은 판단하지 않는다."""
    row = cm.loc[true_class]
    total = int(row.sum())
    correct = int(row[true_class])
    off_diag = row.drop(index=true_class)
    top = off_diag.sort_values(ascending=False).iloc[:top_n]

    top_list = [
        {
            "predicted_class": str(cls),
            "count": int(cnt),
            "fraction_of_true_class": float(cnt) / total if total > 0 else 0.0,
        }
        for cls, cnt in top.items()
    ]
    top_n_sum = sum(item["count"] for item in top_list)

    return {
        "true_class": true_class,
        "n_total": total,
        "n_correct": correct,
        "accuracy": correct / total if total > 0 else None,
        f"top{top_n}_misclassified_as": top_list,
        f"top{top_n}_fraction_of_total": top_n_sum / total if total > 0 else 0.0,
        "n_other_classes_receiving_misclassifications": int((off_diag > 0).sum()),
    }


def compute_class_prevalence(labels: np.ndarray, class_names: list[str], min_ratio: float) -> dict:
    """전체 표본 대비 각 클래스 비율을 계산하고, find_rare_classes(비율 파라미터)로 제외 대상을
    기계적으로 판정한다. label_check.py의 절대값(<10) rare-class 로직을 재사용/일반화한 것이지,
    새로 만든 로직이 아니다."""
    n_total = len(labels)
    value_counts = pd.Series(labels).value_counts().reindex(class_names)

    rare = find_rare_classes(value_counts, n_total, min_ratio=min_ratio)
    excluded = rare.index.tolist()

    table = [
        {
            "class_name": name,
            "n": int(value_counts[name]),
            "ratio": float(value_counts[name]) / n_total,
            "excluded": name in excluded,
        }
        for name in class_names
    ]

    return {
        "n_total": n_total,
        "min_ratio_threshold": min_ratio,
        "class_prevalence": table,
        "excluded_classes": excluded,
        "kept_classes": [name for name in class_names if name not in excluded],
    }


def build_final_prevalence_filtered(
    image_emb: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    prompt_template: str,
    min_ratio: float = MIN_PREVALENCE_RATIO,
    model=None,
    processor=None,
) -> dict:
    """1% 미만 클래스 제외 규칙 적용 후 최종 클래스셋으로 baseline을 재계산.

    FACT: min_ratio 필터는 정확도를 전혀 참조하지 않는다 — labels의 클래스별 표본 수만 본다.
    제외된 클래스가 정답이거나(행 자체가 사라짐) 예측 후보에서 빠지므로(class_names에서
    제외했으니 애초에 예측될 수 없음) 두 경우 모두 자연히 처리된다.
    """
    prevalence = compute_class_prevalence(labels, class_names, min_ratio)
    kept_classes = prevalence["kept_classes"]

    keep_mask = np.isin(labels, kept_classes)
    filtered_image_emb = image_emb[keep_mask]
    filtered_labels = labels[keep_mask]

    result = compute_zeroshot_baseline(
        filtered_image_emb, filtered_labels, kept_classes, prompt_template,
        model=model, processor=processor,
    )
    result.pop("_raw")
    result = _apply_verdict(result)

    result["class_prevalence"] = prevalence["class_prevalence"]
    result["min_ratio_threshold"] = prevalence["min_ratio_threshold"]
    result["excluded_classes"] = prevalence["excluded_classes"]
    result["n_excluded_rows"] = int((~keep_mask).sum())
    result["notes"] = {
        "rule_applied_before_seeing_accuracy": (
            "This 1% prevalence rule is a prior criterion decided before looking at any accuracy "
            "results, and is not targeted at any specific class. It only depends on class sample "
            "counts, never on classification accuracy."
        ),
        "genre_low_accuracy_retained_by_design": (
            "genre is well above the 1% threshold and is NOT excluded by this filter, even though "
            "its single_prompt accuracy is very low (see columns.type.single_prompt."
            "per_class_accuracy.genre). Accuracy plays no role in this filter by design -- this is "
            "intentionally retained for the report footnote."
        ),
        "used_for": (
            "This final_prevalence_filtered configuration is the one to be used in Step 7b "
            "(embedding shift zero-shot sweep)."
        ),
    }
    return result


def run_zeroshot_baseline_analysis(
    npz_path: Path = NPZ_PATH,
    parquet_path: Path = PARQUET_PATH,
    results_path: Path = RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> dict:
    data = np.load(npz_path)
    image_emb = data["image_emb"]
    filenames = data["filenames"]

    df = pd.read_parquet(parquet_path).set_index("filename")
    # npz의 filenames 순서와 parquet 원본 순서가 같다고 가정하지 않고 명시적으로 재정렬
    # (gap.py의 get_treated_mask_by_filename과 같은 원칙).
    df_aligned = df.loc[filenames]

    model, processor = load_model(DEVICE)

    columns_results = {}
    for column, prompt_template in PROMPT_TEMPLATES.items():
        labels = df_aligned[column].to_numpy()
        class_names = sorted(pd.unique(labels).tolist())

        single = compute_zeroshot_baseline(
            image_emb, labels, class_names, prompt_template, model=model, processor=processor,
        )
        single_raw = single.pop("_raw")
        single["prompt_rationale"] = PROMPT_RATIONALE[column]
        single = _apply_verdict(single)

        column_result = {"single_prompt": single}

        if column in ENSEMBLE_PROMPT_TEMPLATES:
            ensemble = compute_zeroshot_baseline_ensemble(
                image_emb, labels, class_names, ENSEMBLE_PROMPT_TEMPLATES[column],
                model=model, processor=processor,
            )
            ensemble.pop("_raw")
            ensemble["prompt_rationale"] = ENSEMBLE_PROMPT_RATIONALE[column]
            ensemble = _apply_verdict(ensemble)
            column_result["ensemble_prompt"] = ensemble

        if column == "type":
            # 작업 B: genre 오분류 진단 — single_prompt 예측 기준, 숫자는 건드리지 않는 순수 관찰.
            cm = compute_confusion_matrix(labels, single_raw["pred_labels"], class_names)
            cm_fig_path = figures_dir / "downstream_type_confusion_matrix.png"
            _plot_confusion_matrix(cm, cm_fig_path, "type: confusion matrix (single_prompt)")
            column_result["confusion_matrix_figure"] = _relpath(cm_fig_path)
            column_result["genre_misclassification_summary"] = summarize_misclassification(
                cm, GENRE_MISCLASSIFICATION_TARGET,
            )

            # 최종 클래스셋 확정 (3차): 비율 1% 미만 클래스만 기계적으로 제외, 정확도는 무관.
            column_result["final_prevalence_filtered"] = build_final_prevalence_filtered(
                image_emb, labels, class_names, prompt_template,
                min_ratio=MIN_PREVALENCE_RATIO, model=model, processor=processor,
            )

        columns_results[column] = column_result

    results = {
        "source_npz": _relpath(npz_path),
        "source_parquet": _relpath(parquet_path),
        "model": MODEL_NAME,
        "device": DEVICE,
        "columns": columns_results,
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    return results


def _print_summary(results: dict) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=True))

    print("\n[Zero-shot Baseline] single_prompt column comparison (lambda=0, no intervention):")
    header = f"{'column':<10}{'micro':>8}{'macro':>8}{'majority':>10}{'chance':>10}{'above_maj(pp)':>16}{'verdict':>18}"
    print(header)
    print("-" * len(header))
    for column, col_result in results["columns"].items():
        r = col_result["single_prompt"]
        diff_pp = r["micro_accuracy_above_majority"] * 100
        macro = r["macro_accuracy"]
        print(
            f"{column:<10}{r['micro_accuracy']:>8.3f}{macro:>8.3f}{r['majority_baseline_accuracy']:>10.3f}"
            f"{r['chance_accuracy']:>10.3f}{diff_pp:>+16.1f}{r['verdict']:>18}"
        )

    for column, col_result in results["columns"].items():
        if "ensemble_prompt" not in col_result:
            continue
        single = col_result["single_prompt"]
        ensemble = col_result["ensemble_prompt"]

        print(f"\n[Zero-shot Baseline] {column}: single_prompt vs ensemble_prompt:")
        print(
            f"{'':<10}{'micro':>8}{'macro':>8}{'above_maj(pp)':>16}{'verdict':>18}"
        )
        print(
            f"{'single':<10}{single['micro_accuracy']:>8.3f}{single['macro_accuracy']:>8.3f}"
            f"{single['micro_accuracy_above_majority'] * 100:>+16.1f}{single['verdict']:>18}"
        )
        print(
            f"{'ensemble':<10}{ensemble['micro_accuracy']:>8.3f}{ensemble['macro_accuracy']:>8.3f}"
            f"{ensemble['micro_accuracy_above_majority'] * 100:>+16.1f}{ensemble['verdict']:>18}"
        )

        print(f"\n[Zero-shot Baseline] {column}: per-class accuracy, single vs ensemble:")
        print(f"{'class':<14}{'n':>6}{'single':>10}{'ensemble':>10}{'delta':>10}")
        for cls in single["per_class_accuracy"]:
            s_acc = single["per_class_accuracy"][cls]["accuracy"]
            e_acc = ensemble["per_class_accuracy"][cls]["accuracy"]
            n = single["per_class_accuracy"][cls]["n"]
            delta = (e_acc - s_acc) if (s_acc is not None and e_acc is not None) else None
            delta_str = f"{delta:>+10.3f}" if delta is not None else f"{'--':>10}"
            print(f"{cls:<14}{n:>6}{s_acc:>10.3f}{e_acc:>10.3f}{delta_str}")

    if "genre_misclassification_summary" in results["columns"].get("type", {}):
        g = results["columns"]["type"]["genre_misclassification_summary"]
        top_key = [k for k in g if k.startswith("top") and k.endswith("misclassified_as")][0]
        frac_key = [k for k in g if k.startswith("top") and k.endswith("fraction_of_total")][0]
        print(f"\n[Confusion Matrix] genre observation (single_prompt, n={g['n_total']}, accuracy={g['accuracy']:.3f}):")
        for item in g[top_key]:
            print(
                f"  -> {item['predicted_class']:<14} count={item['count']:>5} "
                f"({item['fraction_of_true_class']:.1%} of all genre samples)"
            )
        print(f"  top{len(g[top_key])} combined = {g[frac_key]:.1%} of all genre samples")
        print(f"  misclassifications spread across {g['n_other_classes_receiving_misclassifications']} other classes")
        print(f"  confusion matrix figure: {results['columns']['type']['confusion_matrix_figure']}")

    if "final_prevalence_filtered" in results["columns"].get("type", {}):
        f = results["columns"]["type"]["final_prevalence_filtered"]
        n_total_original = sum(c["n"] for c in f["class_prevalence"])
        print(
            f"\n[Final Class Set] type: class prevalence vs {f['min_ratio_threshold']:.0%} threshold "
            f"(n_total={n_total_original}):"
        )
        print(f"{'class':<14}{'n':>6}{'ratio':>9}{'excluded':>10}")
        for c in sorted(f["class_prevalence"], key=lambda c: -c["n"]):
            flag = "EXCLUDED" if c["excluded"] else ""
            print(f"{c['class_name']:<14}{c['n']:>6}{c['ratio']:>9.2%}{flag:>10}")

        print(f"\n[Final Class Set] excluded (ratio < {f['min_ratio_threshold']:.0%}): {f['excluded_classes']}")
        print(f"[Final Class Set] kept classes ({f['n_classes']}): n={f['n']} (excluded {f['n_excluded_rows']} rows)")
        print(
            f"[Final Class Set] micro_accuracy={f['micro_accuracy']:.3f}  macro_accuracy={f['macro_accuracy']:.3f}  "
            f"majority_baseline={f['majority_baseline_accuracy']:.3f}  chance={f['chance_accuracy']:.3f}  "
            f"above_majority={f['micro_accuracy_above_majority'] * 100:+.1f}pp  verdict={f['verdict']}"
        )


if __name__ == "__main__":
    _print_summary(run_zeroshot_baseline_analysis())
