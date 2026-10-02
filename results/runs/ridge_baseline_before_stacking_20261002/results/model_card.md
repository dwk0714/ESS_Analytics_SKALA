# Ridge 모델 카드

- 상태: 학습 모델 저장; B1 holdout·B2·B3 평가 결과는 model_performance.csv에 저장
- 입력: dq100_10_log_variance, cycle10_rms_c_timeweighted
- alpha: 1
- 학습: B1 35개, 정책 18그룹
- 선택 CV MAPE: 6.987% (선택에 사용한 개발 점수)
- 초기100사이클 후 제공 수명 cycle_life를 예측합니다. ESS 운영연수/RUL과 다릅니다.
- 표준화·열 선택·목표 역변환이 저장 모델에 포함됩니다.
- 모델 선택 이후 동일 모델로 Valid/B2/B3를 평가합니다. 전체 B1 재학습은 하지 않습니다.
- 배치/라벨 규칙 차이와 파일 간 물리적 셀 동일성 미확인, 전체 EDA 노출의 한계가 있습니다.
