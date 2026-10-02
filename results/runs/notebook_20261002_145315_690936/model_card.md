# Stacking 모델 카드

- 학습: B1 35개, holdout 제외. 입력: dq100_10_log_variance, cycle10_rms_c_timeweighted, dq100_10_q05_ah, dq100_10_mean_ah
- 구조: OLS·Ridge·XGBoost의 log10 수명 예측 → 표준화·Ridge(alpha=1) → 역변환
- 메타 입력은 그룹 OOF 예측이며 해당 셀의 정답을 사용해 기저 예측을 만들지 않습니다.
- Train CV: 6.645%, 3반복×5fold. 반복별 점수 평균이며 독립 셀은 35개입니다.
- 변수 선정·표준화·XGB 파라미터 선택은 각 학습 부분에서 수행합니다.
- 동일 모델을 holdout/B2/B3에 적용합니다. 전체 B1 재학습은 하지 않습니다.
- 이후 평가는 별도 명령이며 B2/B3의 EDA·비교 노출 및 라벨 차이를 한계로 기록합니다.
