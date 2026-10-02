# 성능 결과

MAPE가 낮을수록 수명 예측이 더 정확합니다. Gap의 단위는%p이며 양수는 뒤 평가의 오차 증가입니다.

| 구분 | MAPE (%) | 비고 |
| --- | --- | --- |
| Train (Batch 1 CV) | 6.645 | B1 그룹 OOF CV; 낮을수록 예측이 더 정확함 |
| Valid (Batch 1 Hold-out) | 11.385 | B1 holdout; 동일 저장 모델 |
| Test (Batch 2) | 27.717 | 필수 최종 배치 평가 |
| Gap (Train-Valid) | 4.740 | Valid−Train (%p); (+) 과적합/집단 차이 검토 |
| Gap (Valid-Test) | 16.332 | Test−Valid (%p); (+) 배치 간 일반화 저하 검토 |
| Gap (Target-Test) | 18.617 | Test−9.1 (%p); 논문 Target과 다른 실험 구성 |
| Test (Batch 3) | 12.780 | 동일 모델 추가 평가 |
| Gap (Batch2-Batch3) | -14.937 | B3−B2 (%p); (+) B3에서 오차 증가 |
| Gap (Target-Test) [Batch 3] | 3.680 | B3−9.1 (%p); 추가 참고 비교 |

## 보조 지표와 중앙 수명 기준 모델

| scope | model | MAPE_percent | MAE_cycles | RMSE_cycles | n | n_repeats |
| --- | --- | --- | --- | --- | --- | --- |
| train_cv | Stacking (OLS + Ridge + XGBoost → Ridge) | 6.645 | 54.421 | 71.498 | 35 | 3.000 |
| valid | Stacking (OLS + Ridge + XGBoost → Ridge) | 11.385 | 107.739 | 134.866 | 11 | — |
| test_b2 | Stacking (OLS + Ridge + XGBoost → Ridge) | 27.717 | 138.527 | 154.494 | 39 | — |
| test_b3 | Stacking (OLS + Ridge + XGBoost → Ridge) | 12.780 | 159.621 | 249.636 | 44 | — |
| train_cv | Training median | 19.887 | 150.700 | 185.326 | 35 | 3.000 |
| valid | Training median | 16.488 | 154.818 | 192.923 | 11 | — |
| test_b2 | Training median | 72.400 | 341.051 | 364.606 | 39 | — |
| test_b3 | Training median | 20.232 | 251.341 | 370.602 | 44 | — |

MAE는 평균 사이클 오차, RMSE는 큰 오차의 영향을 보여줍니다. 중앙 수명 기준값은 CV에서 각 fold 학습 정답의 중앙값, Valid/B2/B3에서 학습35개 정답의 중앙값입니다.

## 해석 범위

Train은 B1 내부 그룹 OOF CV 점수이며 재대입 성능이 아닙니다. Stacking은 3반복×5fold의 반복별 점수 평균이며 독립 학습 셀은 35개입니다. B2/B3는 EDA와 모델 비교에서 확인했으므로 완전히 미관측한 외부 자료는 아닙니다. 논문9.1%는 과제 참고 Target이며, 이번 코호트·분할·라벨 규칙으로 정확한 논문 재현을 주장하지 않습니다.

![실제값과 예측값](figures/01_actual_vs_predicted.png)

![부호 있는 상대 오차](figures/02_signed_error.png)

![입력 특징 분포 차이](figures/03_feature_distribution_shift.png)

[큰 오차 사례](error_analysis.md)
