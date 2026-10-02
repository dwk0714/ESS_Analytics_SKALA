# 저장 모델

B1 학습35개로 OLS·Ridge·XGBoost 기저 예측을 메타 Ridge로 결합한 Stacking을 학습했습니다.

`stacking.joblib`에는 특징 열 선택·표준화·log10 수명 예측·메타 결합·역변환이 포함됩니다. 최종 입력은 ΔQ 로그분산·cycle10 RMS·ΔQ 5% 분위수·ΔQ 평균입니다. 설정·학습 셀·CV·모델 해시는 `metadata.json`에 저장했습니다.

동일 모델로 B1 holdout·B2·B3를 평가했습니다. 전체 B1로 다시 학습하지 않았습니다. [최종 성능](../results/performance_report.md), [초기 Ridge 보관](../results/runs/ridge_baseline_before_stacking_20261002/models/metadata.json).
