# 결과와 실행 상태

최종 모델: **Stacking (OLS·Ridge·XGBoost → Ridge),4입력**.

현재 상태: **DAY2 특징 추출·모델 학습·B1 Hold-out 및 B2/B3 평가·보고서 생성 완료**.

- `eda/`: 기존 DAY1 분석 결과를 제출 구조에 복사한 자료. 새 DAY2 모델 성능이 아닙니다.
- 학습: `cv_search_results.csv`, `cv_repeat_results.csv`, `cv_fold_results.csv`, `cv_predictions.csv`, `split_manifest.csv`, `cv_split_manifest.csv`, `training_runtime.json`, `model_card.md`에 저장.
- 평가: `model_performance.csv`, `model_metrics.csv`, `predictions.csv`, `evaluation.json`에 저장.
- 보고서: `performance_report.md`, `error_analysis.csv`, `error_analysis.md`, `figures/`에 저장.

주 지표 MAPE가 낮을수록 수명 예측이 더 정확합니다. Gap은 Valid−Train, Test−Valid, Test−9.1이며 양수는 뒤 평가의 오차 증가입니다. Gap 단위는%p입니다.

기존 모델이나 성능 파일이 있는 곳에 새 학습/평가를 덮어쓰지 않습니다. 후속 실험은 `--run-dir results/runs/새이름`으로 분리합니다. 각 학습 실행은 B1 학습 부분만으로 변수·조정값을 선택합니다. B2/B3는 개발 비교에 이미 노출된 배치임을 보고서에 기록했습니다.

- `runs/final_stacking_20261002/`: 최종 기본 파이프라인 재학습·평가 결과.
- `runs/top2_feature_comparison_20261002/`: 같은 조건의 모델·추가 변수 비교.
- `runs/ridge_baseline_before_stacking_20261002/`: 최초 Ridge 모델·특징·성과 보관.
- Train은3반복 그룹5fold OOF 평균, 독립 셀은35개입니다.

