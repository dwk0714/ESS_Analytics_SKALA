"""Train the selected four-input Stacking with repeated nested protocol CV."""
from __future__ import annotations
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from .evaluate import metrics, summarize_cv
from .features import RANKING_CANDIDATES, STACKING_FEATURES, data_fingerprint
from .preprocess import file_sha256
from .splits import make_splits, development_data
from .stacking import StackingTrainer, grouped_folds, rank_candidates


def train_stacking(frame, config, output_dir, model_dir):
    from .train import runtime_environment
    started = time.perf_counter()
    output_dir, model_dir = Path(output_dir), Path(model_dir)
    if (output_dir / 'model_performance.csv').exists() or (model_dir / 'metadata.json').exists():
        raise FileExistsError('기존 학습·평가를 보존합니다. 새 --run-dir로 실행하세요.')
    settings = config['stacking']
    if settings['inner_folds'] != 3 or settings['base_ridge_alpha'] != 1 or settings['meta_ridge_alpha'] != 1:
        raise ValueError('검증한 Stacking은 내부 3fold와 Ridge alpha=1을 사용합니다.')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_splits(frame, config)
    development = development_data(frame, manifest)
    trainer = StackingTrainer(RANKING_CANDIDATES, development.cell_key, settings['parameter_count'])
    predictions, folds, partitions, repeats = [], [], [], []
    y = development.cycle_life.to_numpy(float)
    for repeat, seed in enumerate(settings['outer_seeds'], 1):
        predicted, baseline = np.full(len(y), np.nan), np.full(len(y), np.nan)
        fold_ids = np.full(len(y), -1, dtype=int)
        for fold, (tr, va) in enumerate(grouped_folds(development, settings['outer_folds'], seed), 1):
            training = development.iloc[tr].reset_index(drop=True)
            model = trainer.fit('stacking', training, True, seed, f'repeat{repeat}/outer{fold}')
            predicted[va] = model.predict(development.iloc[va])
            baseline[va] = np.median(y[tr])
            fold_ids[va] = fold
            folds.append({'repeat':repeat,'seed':seed,'fold':fold,'n_train':len(tr),'n_valid':len(va),
                          **metrics(y[va],predicted[va]),
                          'fit_MAPE_percent':metrics(y[tr],model.predict(training))['MAPE_percent'],
                          'selected_extra_features':';'.join(model.selected_extra_features),
                          'selected_parameters':json.dumps(model.parameters,sort_keys=True)})
            for role, positions in [('train',tr),('valid',va)]:
                partitions.extend({'repeat':repeat,'seed':seed,'fold':fold,'role':role,
                                   'cell_key':development.iloc[i].cell_key,'policy':development.iloc[i].policy} for i in positions)
        score = metrics(y,predicted)
        repeats.append({'repeat':repeat,'seed':seed,**score})
        predictions.extend({'repeat':repeat,'seed':seed,'fold':int(fold_ids[i]),'cell_key':key,
                            'actual_cycle_life':float(y[i]),'predicted_cycle_life':float(predicted[i]),
                            'baseline_cycle_life':float(baseline[i])} for i,key in enumerate(development.cell_key))
        print(f'Stacking 그룹 CV {repeat}/{len(settings["outer_seeds"])}: MAPE {score["MAPE_percent"]:.3f}%',flush=True)
    cv_seconds = time.perf_counter()-started
    refit_started = time.perf_counter()
    final_model = trainer.fit('stacking',development,True,config['random_state'],'final_B1_train35')
    used_features = set(f for selected in final_model.features for f in selected)
    final_features = list(config['selected_features'])
    if used_features != set(STACKING_FEATURES) or final_features != STACKING_FEATURES:
        raise ValueError(f'B1에서 선택한 최종 특징이 승인 구성과 다릅니다: {final_features}')
    joblib.dump(final_model,model_dir/'stacking.joblib')
    refit_seconds = time.perf_counter()-refit_started
    cv = pd.DataFrame(predictions)
    cv.to_csv(output_dir/'cv_predictions.csv',index=False)
    cv_metrics, base_metrics = summarize_cv(cv,set(development.cell_key),len(repeats))
    pd.DataFrame(repeats).to_csv(output_dir/'cv_repeat_results.csv',index=False)
    pd.DataFrame(folds).to_csv(output_dir/'cv_fold_results.csv',index=False)
    pd.DataFrame(partitions).to_csv(output_dir/'cv_split_manifest.csv',index=False)
    pd.DataFrame(trainer.inner_search).to_csv(output_dir/'cv_search_results.csv',index=False)
    pd.DataFrame(trainer.rankings).to_csv(output_dir/'feature_selection_ranking.csv',index=False)
    pd.DataFrame(rank_candidates(development,RANKING_CANDIDATES)[1]).to_csv(output_dir/'final_feature_ranking.csv',index=False)
    pd.DataFrame(trainer.meta_predictions).to_csv(output_dir/'stacking_meta_oof_predictions.csv',index=False)
    (output_dir/'selection_scope_audit.json').write_text(json.dumps(trainer.selection_scopes,ensure_ascii=False,indent=2)+'\n')
    manifest.to_csv(output_dir/'split_manifest.csv',index=False)
    weighted_fit = sum(row['fit_MAPE_percent']*row['n_train'] for row in folds)/sum(row['n_train'] for row in folds)
    metadata={'status':'trained_not_evaluated','model':'Stacking (OLS + Ridge + XGBoost → Ridge)',
              'model_filename':'stacking.joblib','model_type':'stacking','selected_features':final_features,
              'base_features':final_model.features,'selected_extra_features':final_model.selected_extra_features,
              'parameters':final_model.parameters,'base_ridge_alpha':1.0,'meta_ridge_alpha':1.0,
              'forecast_cycle':100,'target':'provided cycle_life','target_transform':'log10',
              'inverse_transform':'10**prediction','train_cell_keys':development.cell_key.tolist(),
              'training_rows':len(y),'training_policy_groups':int(development.policy.nunique()),
              'cv_repeats':len(repeats),'cv_method':'3 repeated 5-fold outer group CV; feature ranking, XGB tuning and stacking OOF within each training subset',
              'cv_metrics':cv_metrics,'cv_baseline_metrics':base_metrics,
              'repeat_MAPE_sd':float(np.std([r['MAPE_percent'] for r in repeats],ddof=1)),
              'outer_training_fit_MAPE_percent':float(weighted_fit),
              'outer_cv_minus_fit_gap_percent_points':cv_metrics['MAPE_percent']-weighted_fit,
              'baseline_train_median':float(np.median(y)),'fit_count':trainer.fit_count,
              'data_fingerprint':data_fingerprint(frame),'config':config,
              'model_file_sha256':file_sha256(model_dir/'stacking.joblib'),
              'cv_predictions_sha256':file_sha256(output_dir/'cv_predictions.csv'),
              'meta_standardized_coefficients_log10':final_model.meta.named_steps['ridge'].coef_.tolist(),
              'environment':runtime_environment(),'same_model_for_valid_b2_b3':True,
              'limitations':['B2/B3 already used in EDA and model comparisons; not untouched tests',
                             'stored label ending rules differ across batches',
                             'two added predictors are highly correlated',
                             'model-family choice used repeated development comparisons']}
    (model_dir/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    runtime={'cv_seconds':cv_seconds,'refit_seconds':refit_seconds,
             'training_total_seconds':time.perf_counter()-started,'fit_count':trainer.fit_count,
             'device':'CPU','environment':metadata['environment']}
    (output_dir/'training_runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
    (output_dir/'model_card.md').write_text('# Stacking 모델 카드\n\n'
        f'- 학습: B1 {len(y)}개, holdout 제외. 입력: {", ".join(final_features)}\n'
        '- 구조: OLS·Ridge·XGBoost의 log10 수명 예측 → 표준화·Ridge(alpha=1) → 역변환\n'
        '- 메타 입력은 그룹 OOF 예측이며 해당 셀의 정답을 사용해 기저 예측을 만들지 않습니다.\n'
        f'- Train CV: {cv_metrics["MAPE_percent"]:.3f}%, 3반복×5fold. 반복별 점수 평균이며 독립 셀은 35개입니다.\n'
        '- 변수 선정·표준화·XGB 파라미터 선택은 각 학습 부분에서 수행합니다.\n'
        '- 동일 모델을 holdout/B2/B3에 적용합니다. 전체 B1 재학습은 하지 않습니다.\n'
        '- 이후 평가는 별도 명령이며 B2/B3의 EDA·비교 노출 및 라벨 차이를 한계로 기록합니다.\n')
    return metadata
