# 큰 오차 사례

평가셋별 APE 상위3개 셀을 평가 전에 정한 기준으로 정리합니다.

| role | cell_key | policy | actual_cycle_life | predicted_cycle_life | APE_percent |
| --- | --- | --- | --- | --- | --- |
| test_b2 | B2c15 | 3.6C(9%)-5C | 396.000 | 661.520 | 67.051 |
| test_b2 | B2c6 | 3.6C(9%)-5C | 393.000 | 630.523 | 60.438 |
| test_b2 | B2c18 | 5.2C(50%)-4.25C | 449.000 | 715.772 | 59.415 |
| test_b3 | B3c38 | 5C(67%)-4C-newstructure | 1935.000 | 1006.499 | 47.985 |
| test_b3 | B3c7 | 4.8C(80%)-4.8C-newstructure | 1836.000 | 1170.786 | 36.232 |
| test_b3 | B3c45 | 4.8C(80%)-4.8C-newstructure | 1801.000 | 1152.220 | 36.023 |
| valid | B1c1 | 3.6C(80%)-3.6C | 1179.000 | 1439.916 | 22.130 |
| valid | B1c0 | 3.6C(80%)-3.6C | 1190.000 | 1446.706 | 21.572 |
| valid | B1c15 | 5.4C(60%)-3C | 719.000 | 850.890 | 18.344 |

- B2c15: 과대 예측, 오차+265.5cycle / APE67.05%. 선택 입력 중 학습 범위 밖: 없음. 정책: 3.6C(9%)-5C. 정답 기록 규칙: first Qd<0.88Ah event matches stored label.

- B2c6: 과대 예측, 오차+237.5cycle / APE60.44%. 선택 입력 중 학습 범위 밖: 없음. 정책: 3.6C(9%)-5C. 정답 기록 규칙: first Qd<0.88Ah event matches stored label.

- B2c18: 과대 예측, 오차+266.8cycle / APE59.41%. 선택 입력 중 학습 범위 밖: 없음. 정책: 5.2C(50%)-4.25C. 정답 기록 규칙: first Qd<0.88Ah event matches stored label.

- B3c38: 과소 예측, 오차-928.5cycle / APE47.98%. 선택 입력 중 학습 범위 밖: 없음. 정책: 5C(67%)-4C-newstructure. 정답 기록 규칙: stored label matches last observed cycle+1.

- B3c7: 과소 예측, 오차-665.2cycle / APE36.23%. 선택 입력 중 학습 범위 밖: 없음. 정책: 4.8C(80%)-4.8C-newstructure. 정답 기록 규칙: stored label matches last observed cycle+1.

- B3c45: 과소 예측, 오차-648.8cycle / APE36.02%. 선택 입력 중 학습 범위 밖: 없음. 정책: 4.8C(80%)-4.8C-newstructure. 정답 기록 규칙: stored label matches last observed cycle+1.

- B1c1: 과대 예측, 오차+260.9cycle / APE22.13%. 선택 입력 중 학습 범위 밖: dq100_10_log_variance;cycle10_rms_c_timeweighted;dq100_10_q05_ah;dq100_10_mean_ah. 정책: 3.6C(80%)-3.6C. 정답 기록 규칙: stored label matches last observed cycle+1.

- B1c0: 과대 예측, 오차+256.7cycle / APE21.57%. 선택 입력 중 학습 범위 밖: dq100_10_log_variance;cycle10_rms_c_timeweighted;dq100_10_q05_ah;dq100_10_mean_ah. 정책: 3.6C(80%)-3.6C. 정답 기록 규칙: stored label matches last observed cycle+1.

- B1c15: 과대 예측, 오차+131.9cycle / APE18.34%. 선택 입력 중 학습 범위 밖: 없음. 정책: 5.4C(60%)-3C. 정답 기록 규칙: stored label matches last observed cycle+1.

셀별 초기 품질 정보가 있는 열은 error_analysis.csv에 함께 남깁니다. 범위 이탈·프로토콜·배치·라벨 차이는 원인 후보이며 특정 열화 기전의 증명이 아닙니다.

ESS에서는 상태 점검 우선순위와 교체 계획 검토의 근거가 될 수 있습니다. 운영 연수·잔여 수명으로 바로 환산하지 않으며 calendar aging, SOC·온도·팩 차이와 현장 외부 검증이 추가로 필요합니다.
