# 저장 모델

Batch 1 학습35개로 Ridge를 학습하고 모델을 저장했습니다. 선택 입력은 ΔQ 로그 분산과 cycle10 RMS, alpha는1입니다.

`ridge.joblib`와 `metadata.json`에 모델과 선택 설정을 저장했습니다. 모델에는 입력 열 선택·학습 부분의 표준화·Ridge·목표 log10 역변환이 함께 포함됩니다.

같은 저장 모델로 B1 holdout·B2·B3를 평가했습니다. B2 평가 전에 전체 B1로 다시 학습하지 않았습니다. 결과는 [성능 보고서](../results/performance_report.md)에 기록했습니다.
