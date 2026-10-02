# 큰 오차 사례

평가셋별 APE 상위3개 셀을 평가 전에 정한 기준으로 정리합니다.

| role | cell_key | policy | actual_cycle_life | predicted_cycle_life | APE_percent |
| --- | --- | --- | --- | --- | --- |
| test_b2 | B2c6 | 3.6C(9%)-5C | 393.000 | 691.233 | 75.886 |
| test_b2 | B2c15 | 3.6C(9%)-5C | 396.000 | 671.128 | 69.477 |
| test_b2 | B2c18 | 5.2C(50%)-4.25C | 449.000 | 732.784 | 63.204 |
| test_b3 | B3c38 | 5C(67%)-4C-newstructure | 1935.000 | 949.399 | 50.935 |
| valid | B1c1 | 3.6C(80%)-3.6C | 1179.000 | 1670.933 | 41.725 |
| valid | B1c0 | 3.6C(80%)-3.6C | 1190.000 | 1655.752 | 39.139 |
| test_b3 | B3c7 | 4.8C(80%)-4.8C-newstructure | 1836.000 | 1146.537 | 37.552 |
| test_b3 | B3c45 | 4.8C(80%)-4.8C-newstructure | 1801.000 | 1141.084 | 36.642 |
| valid | B1c2 | 3.6C(80%)-3.6C | 1177.000 | 1487.008 | 26.339 |

- B2c6: 과대 예측, 오차+298.2cycle / APE75.89%. 선택 입력 중 학습 범위 밖: 없음. 정책: 3.6C(9%)-5C. 정답 기록 규칙: first Qd<0.88Ah event matches stored label.

- B2c15: 과대 예측, 오차+275.1cycle / APE69.48%. 선택 입력 중 학습 범위 밖: 없음. 정책: 3.6C(9%)-5C. 정답 기록 규칙: first Qd<0.88Ah event matches stored label.

- B2c18: 과대 예측, 오차+283.8cycle / APE63.20%. 선택 입력 중 학습 범위 밖: 없음. 정책: 5.2C(50%)-4.25C. 정답 기록 규칙: first Qd<0.88Ah event matches stored label.

- B3c38: 과소 예측, 오차-985.6cycle / APE50.94%. 선택 입력 중 학습 범위 밖: 없음. 정책: 5C(67%)-4C-newstructure. 정답 기록 규칙: stored label matches last observed cycle+1.

- B1c1: 과대 예측, 오차+491.9cycle / APE41.72%. 선택 입력 중 학습 범위 밖: dq100_10_log_variance;cycle10_rms_c_timeweighted. 정책: 3.6C(80%)-3.6C. 정답 기록 규칙: stored label matches last observed cycle+1.

- B1c0: 과대 예측, 오차+465.8cycle / APE39.14%. 선택 입력 중 학습 범위 밖: dq100_10_log_variance;cycle10_rms_c_timeweighted. 정책: 3.6C(80%)-3.6C. 정답 기록 규칙: stored label matches last observed cycle+1.

- B3c7: 과소 예측, 오차-689.5cycle / APE37.55%. 선택 입력 중 학습 범위 밖: 없음. 정책: 4.8C(80%)-4.8C-newstructure. 정답 기록 규칙: stored label matches last observed cycle+1.

- B3c45: 과소 예측, 오차-659.9cycle / APE36.64%. 선택 입력 중 학습 범위 밖: 없음. 정책: 4.8C(80%)-4.8C-newstructure. 정답 기록 규칙: stored label matches last observed cycle+1.

- B1c2: 과대 예측, 오차+310.0cycle / APE26.34%. 선택 입력 중 학습 범위 밖: dq100_10_log_variance;cycle10_rms_c_timeweighted. 정책: 3.6C(80%)-3.6C. 정답 기록 규칙: stored label matches last observed cycle+1.

셀별 초기 품질 정보가 있는 열은 error_analysis.csv에 함께 남깁니다. 범위 이탈·프로토콜·배치·라벨 차이는 원인 후보이며 특정 열화 기전의 증명이 아닙니다.

ESS에서는 상태 점검 우선순위와 교체 계획 검토의 근거가 될 수 있습니다. 운영 연수·잔여 수명으로 바로 환산하지 않으며 calendar aging, SOC·온도·팩 차이와 현장 외부 검증이 추가로 필요합니다.
