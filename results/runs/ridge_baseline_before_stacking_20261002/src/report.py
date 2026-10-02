"""Save concise tables and diagnostic plots after evaluation, without fitting."""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ess_analytics_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"train": "#3975A4", "valid": "#729CC0", "test_b2": "#D07836", "test_b3": "#39866B"}
NAMES = {"train": "B1 train", "valid": "B1 holdout", "test_b2": "B2 test", "test_b3": "B3 test"}
LABELS = {"dq100_10_log_variance": "Delta-Q log variance",
          "cycle10_rms_c_timeweighted": "Cycle-10 RMS (C-rate)",
          "capacity_mad_relative": "Relative capacity MAD (%)"}


def load_evaluation(frame: pd.DataFrame, model_dir: Path, output_dir: Path) -> dict:
    from .features import data_fingerprint
    output_dir = Path(output_dir)
    metadata = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    if data_fingerprint(frame) != metadata["data_fingerprint"]:
        raise ValueError("학습·평가에 쓴 특징 표와 다릅니다. 같은 자료로 보고서를 작성하세요.")
    return {"metadata": metadata,
            "summary": json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8")),
            "performance": pd.read_csv(output_dir / "model_performance.csv"),
            "predictions": pd.read_csv(output_dir / "predictions.csv", float_precision="round_trip")}


def markdown_table(frame: pd.DataFrame) -> str:
    """Avoid a tabulate dependency and preserve human-readable precision."""
    def format_value(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns = list(frame.columns)
    lines = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(format_value(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_feature_diagnostics(frame: pd.DataFrame, output_dir: Path):
    """Exploratory target associations, not predictive performance."""
    from .features import FEATURES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scopes = [("ALL", frame)] + list(frame.groupby("batch"))
    scopes.append(("ALL_batch_centered", frame))
    for scope, group in scopes:
        for feature in FEATURES:
            values = group[["batch", feature, "cycle_life"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
            if scope == "ALL_batch_centered":
                values[[feature, "cycle_life"]] -= values.groupby("batch")[[feature, "cycle_life"]].transform("mean")
            rows.append({"scope": scope, "feature": feature, "n": len(values),
                         "Pearson_r": values[feature].corr(values.cycle_life, method="pearson"),
                         "Spearman_rho": values[feature].corr(values.cycle_life, method="spearman")})
    pd.DataFrame(rows).to_csv(output_dir / "feature_life_correlations.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, feature in zip(axes, FEATURES):
        scale = 100 if feature == "capacity_mad_relative" else 1
        for batch, group in frame.groupby("batch"):
            ax.scatter(group[feature] * scale, group.cycle_life, s=24, alpha=.7,
                       label=batch, color={"B1": COLORS["train"], "B2": COLORS["test_b2"], "B3": COLORS["test_b3"]}[batch])
        ax.set(xlabel=LABELS[feature], ylabel="Stored cycle life", title=LABELS[feature])
        ax.grid(alpha=.15)
        ax.legend()
    fig.suptitle("Early predictors and stored lifetime: exploratory EDA")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_features_vs_life.png", dpi=180)
    plt.close(fig)


def write_reports(evaluation: dict, frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    predictions, metadata = evaluation["predictions"], evaluation["metadata"]
    roles = list(predictions.role.drop_duplicates())
    features = metadata["selected_features"]
    # 1. One scale across independent evaluation subsets.
    fig, axes = plt.subplots(1, len(roles), figsize=(5 * len(roles), 4.5), squeeze=False)
    limit = float(max(predictions.actual_cycle_life.max(), predictions.predicted_cycle_life.max()) * 1.05)
    for ax, role in zip(axes[0], roles):
        part = predictions.loc[predictions.role.eq(role)]
        ax.scatter(part.actual_cycle_life, part.predicted_cycle_life, c=COLORS[role], alpha=.8, s=32)
        ax.plot([0, limit], [0, limit], color="gray", linestyle="--", linewidth=1)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual life (cycles)", ylabel="Predicted life (cycles)",
               title=f"{NAMES[role]} | n={len(part)}")
        ax.grid(alpha=.15)
    fig.suptitle("Same saved model: actual vs predicted life")
    fig.tight_layout()
    fig.savefig(figures / "01_actual_vs_predicted.png", dpi=180)
    plt.close(fig)
    # 2. Positive signed error means overprediction, not a larger lifetime.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for role in roles:
        part = predictions.loc[predictions.role.eq(role)]
        axes[0].scatter(part.actual_cycle_life, part.signed_error_percent, color=COLORS[role], label=NAMES[role], alpha=.8)
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].set(xlabel="Actual life (cycles)", ylabel="Signed error (%)", title="Positive = overprediction")
    axes[0].legend()
    boxes = axes[1].boxplot([predictions.loc[predictions.role.eq(role), "signed_error_percent"].to_numpy() for role in roles],
                            patch_artist=True)
    for box, role in zip(boxes["boxes"], roles):
        box.set_facecolor(COLORS[role])
        box.set_alpha(.65)
    axes[1].set_xticks(range(1, len(roles) + 1), [NAMES[role] for role in roles])
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set(ylabel="Signed error (%)", title="Batch-level error bias")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(figures / "02_signed_error.png", dpi=180)
    plt.close(fig)
    # 3. Training coverage and held-out feature distributions.
    manifest = pd.read_csv(output_dir / "split_manifest.csv")
    role_data = frame.merge(manifest[["cell_key", "role"]], on="cell_key", validate="one_to_one")
    displayed_roles = ["train", *roles]
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4.5), squeeze=False)
    training = role_data.loc[role_data.role.eq("train")]
    for ax, feature in zip(axes[0], features):
        scale = 100 if feature == "capacity_mad_relative" else 1
        ax.axvspan(training[feature].min() * scale, training[feature].max() * scale,
                   color=COLORS["train"], alpha=.1, label="B1 training range")
        for role in displayed_roles:
            values = role_data.loc[role_data.role.eq(role), feature] * scale
            if len(values):
                ax.hist(values, bins=10, density=True, histtype="step", linewidth=1.8,
                        color=COLORS[role], label=f"{NAMES[role]} n={len(values)}")
        ax.set(xlabel=LABELS[feature], ylabel="Density", title=LABELS[feature])
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle("Input distributions and B1 training coverage")
    fig.tight_layout()
    fig.savefig(figures / "03_feature_distribution_shift.png", dpi=180)
    plt.close(fig)
    top_n = metadata["config"]["error_cases_per_dataset"]
    errors = predictions.sort_values("APE_percent", ascending=False).groupby("role", sort=False).head(top_n).copy()
    errors.to_csv(output_dir / "error_analysis.csv", index=False)
    lines = ["# 큰 오차 사례", "", f"평가셋별 APE 상위{top_n}개 셀을 평가 전에 정한 기준으로 정리합니다.", ""]
    lines.append(markdown_table(errors[["role", "cell_key", "policy", "actual_cycle_life", "predicted_cycle_life", "APE_percent"]]))
    for row in errors.itertuples():
        direction = "과대" if row.error_cycles > 0 else "과소" if row.error_cycles < 0 else "정확"
        outside = "없음" if pd.isna(row.out_of_training_range_features) or not row.out_of_training_range_features else row.out_of_training_range_features
        lines.append(f"\n- {row.cell_key}: {direction} 예측, 오차{row.error_cycles:+.1f}cycle / APE{row.APE_percent:.2f}%. "
                     f"선택 입력 중 학습 범위 밖: {outside}. 정책: {row.policy}. 정답 기록 규칙: {row.target_rule_note}.")
    lines.extend(["", "셀별 초기 품질 정보가 있는 열은 error_analysis.csv에 함께 남깁니다. "
                  "범위 이탈·프로토콜·배치·라벨 차이는 원인 후보이며 특정 열화 기전의 증명이 아닙니다.", "",
                  "ESS에서는 상태 점검 우선순위와 교체 계획 검토의 근거가 될 수 있습니다. "
                  "운영 연수·잔여 수명으로 바로 환산하지 않으며 calendar aging, SOC·온도·팩 차이와 현장 외부 검증이 추가로 필요합니다."])
    (output_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics_frame = pd.read_csv(output_dir / "model_metrics.csv")
    report = ["# 성능 결과", "", "MAPE가 낮을수록 수명 예측이 더 정확합니다. Gap의 단위는%p이며 양수는 뒤 평가의 오차 증가입니다.", "",
              markdown_table(evaluation["performance"][["구분", "MAPE (%)", "비고"]]), "",
              "## 보조 지표와 중앙 수명 기준 모델", "", markdown_table(metrics_frame), "",
              "MAE는 평균 사이클 오차, RMSE는 큰 오차의 영향을 보여줍니다. "
              "중앙 수명 기준값은 CV에서 각 fold 학습 정답의 중앙값, Valid/B2/B3에서 학습35개 정답의 중앙값입니다.", "",
              "## 해석 범위", "", "Train은 모델 선택에 사용한 그룹 CV 점수이며 재대입 성능이 아닙니다. "
              "B2/B3는 DAY1 EDA에서 확인했으므로 완전히 미관측한 개발 외부 자료는 아닙니다. "
              "논문9.1%는 과제 참고 Target이며, 이번 코호트·분할·라벨 규칙으로 정확한 논문 재현을 주장하지 않습니다.", "",
              "![실제값과 예측값](figures/01_actual_vs_predicted.png)", "",
              "![부호 있는 상대 오차](figures/02_signed_error.png)", "",
              "![입력 특징 분포 차이](figures/03_feature_distribution_shift.png)", "",
              "[큰 오차 사례](error_analysis.md)"]
    (output_dir / "performance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
