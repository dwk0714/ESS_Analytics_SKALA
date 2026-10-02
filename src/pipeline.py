"""Separate CLI commands: EDA / features / training / evaluation / report.

Example: python -m src.pipeline features --raw-dir ../archive
Training and evaluation never start just by importing a notebook/module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    return json.loads((Path(path) if path else ROOT / "config/model.json").read_text(encoding="utf-8"))


def choose_raw_dir(root: Path, config: dict, override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    candidates = [Path(root) / config["raw_dir"], Path(root).parent / "archive"]
    for folder in candidates:
        if all((folder / name).is_file() for name in list(config["batches"].values())[:2]):
            return folder.resolve()
    return candidates[0].resolve()


def read_feature_csv(path: Path, config: dict) -> pd.DataFrame:
    from .features import check_feature_table
    frame = pd.read_csv(path, float_precision="round_trip")
    return check_feature_table(frame, config)


def save_features(frame: pd.DataFrame, sources: list[dict], root: Path, seconds: float) -> Path:
    path = Path(root) / "data/processed/cell_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")
    out = Path(root) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_manifest.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {"status": "features_extracted_not_trained", "records": len(frame), "feature_latest_cycle": 100,
             "labeled_by_batch": frame.loc[frame.cycle_life.notna()].groupby("batch").size().to_dict(),
             "missing_life": int(frame.cycle_life.isna().sum()), "cell_id_unique": frame.cell_key.is_unique,
             "raw_extraction_and_hash_seconds": seconds}
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="ESS: 특징 추출 → 학습 → 별도 평가")
    command.add_argument("--config", type=Path, default=ROOT / "config/model.json")
    sub = command.add_subparsers(dest="stage", required=True)
    eda = sub.add_parser("eda", help="기존 원본에서 DAY1 EDA 재생성; 모델 학습 없음")
    eda.add_argument("--raw-dir", type=Path)
    features = sub.add_parser("features", help="초기 특징 및 원본에 저장된 정답 CSV 생성")
    features.add_argument("--raw-dir", type=Path)
    features.add_argument("--batches", nargs="+", choices=["B1", "B2", "B3"], default=["B1", "B2", "B3"])
    for name, help_text in (("train", "B1 내부 CV 선택·재학습; holdout/B2 점수 계산 없음"),
                            ("evaluate", "저장 모델의 holdout/B2 평가; 재학습 없음"),
                            ("report", "저장된 평가 결과로 도표와 보고서 생성")):
        stage = sub.add_parser(name, help=help_text)
        stage.add_argument("--features", type=Path, default=ROOT / "data/processed/cell_features.csv")
        stage.add_argument("--run-dir", type=Path, default=ROOT / "results")
        stage.add_argument("--model-dir", type=Path)
        if name == "evaluate":
            stage.add_argument("--include-b3", action="store_true")
    return command


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.stage == "eda":
        from .eda.runner import run_eda
        run_eda(ROOT, choose_raw_dir(ROOT, config, args.raw_dir))
        return
    if args.stage == "features":
        from .features import extract_features
        from .report import write_feature_diagnostics
        start = time.perf_counter()
        frame, sources = extract_features(choose_raw_dir(ROOT, config, args.raw_dir), config, args.batches)
        path = save_features(frame, sources, ROOT, time.perf_counter() - start)
        write_feature_diagnostics(frame, ROOT / "results/feature_diagnostics")
        print(f"특징 저장: {path}; {len(frame)}셀, 모델 학습 없음")
        return
    frame = read_feature_csv(args.features, config)
    out = args.run_dir.resolve()
    model_dir = args.model_dir or (ROOT / "models" if out == (ROOT / "results").resolve() else out / "model")
    if args.stage == "train":
        from .train import train_model
        metadata = train_model(frame, config, out, model_dir)
        print(f"학습 완료: {metadata['model']}, {metadata['selected_features']}; holdout/B2 평가 전")
    elif args.stage == "evaluate":
        from .evaluate import evaluate_model
        evaluation = evaluate_model(frame, model_dir, out, args.include_b3)
        print(evaluation["performance"][["구분", "MAPE (%)", "비고"]].to_string(index=False))
        print(f"평가 결과 저장: {out}; report 명령으로 도표와 보고서를 만드세요.")
    else:
        from .report import load_evaluation, write_reports
        write_reports(load_evaluation(frame, model_dir, out), frame, out)
        print(f"성능 보고서: {out / 'performance_report.md'}")


if __name__ == "__main__":
    from .runtime import ensure_openmp_process
    ensure_openmp_process()
    main()
