# 세 배치 QD Knee 및 열화 가속 EDA

## 분석의 범위와 판단 기준

- 원본의 실제 `summary.cycle` 및 `QDischarge`만 사용한다. cycle_life 결측은 관측길이로 대체하지 않는다. 전체 곡선 Knee·후기 기울기는 수명 종료 후 관측을 포함하는 사후 기술량이며 초기 예측 입력으로 사용할 수 없다.
- 모든 값이 0인 첫 placeholder, 비유한/비양수 QD, 1.65 Ah 초과(QD nominal 1.1 Ah의 1.5배)와 고립 스파이크만 fit에서 제외한다. 1.65 Ah는 보수적 품질검사 경계이며 물리적 상한이 확정됐다는 뜻이 아니다. 어느 시점이든 이 경계를 넘은 셀은 정상 범위 셀과 별도 집계한다.
- 고립 스파이크는 이전 2개·이후 2개 값이 주변 중앙값 ±0.03 Ah 안에 모이고 해당 점만 0.10 Ah 초과 이탈할 때다. 양 끝에서는 이 규칙을 적용하지 않는다. 0.88 Ah 이하인 점을 일괄 제거하지 않으며, 급락도 이 조건 없이 제거하지 않는다. 제외 값과 이유는 `quality_flags.csv`에 보존한다.
- cycle 20부터 centered rolling median 15(최소 관측 3점)를 사용하고, 연속 2구간 선형 hinge `QD=a+b·cycle+c·max(cycle−k,0)`를 적합한다. 전체 160점/159 cycle 이상, 각 구간 최소 40점이면서 전체의 15% 이상이어야 한다. 약 400개 실제 cycle 후보 중 SSE 최소 시점을 선택한다.
- 단일 직선 대비 SSE 20% 이상 개선, 후기 slope가 더 음수이고 차이 ≥0.00001 Ah/cycle, 전기 slope가 충분히 음수면 후기/전기 비율 ≥1.5인 경우에만 `candidate_knee`다. 전기가 거의 평탄하거나 상승하면 후기 음수 전환을 별도로 허용한다. 조건 불일치는 일정속도 확정이 아니라 불확실, 부족한 관측은 unresolved다.
- Knee는 급격한 열화 전환을 근사한 회고적·현상론적 후보다. 매끈한 곡률도 hinge 개선을 만들 수 있으므로 특정 물리기전·통계적 change point 확정과 동일하지 않다. SSE는 적합도이며 cycle 간 자기상관을 무시한 유의확률이 아니다.
- 시작 cycle 20 뒤 관측 span의 처음/마지막 20% 선형 기울기를 독립 보완지표로 계산한다. 초기 2–100 slope는 관측과 유효 label(있는 경우)이 100에 도달할 때만 계산한다. 같은 셀의 여러 cycle은 독립 표본으로 취급하지 않는다.

## 배치별 정상 용량 범위 셀 결과

| 배치 | 정상 범위 셀 / 전체 | 적합 가능 | Knee 후보 | 후보 median cycle | median 관측위치 | 후기/전기 20% 비율 median (n) | 20% 구간 가속 셀 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 45/46 | 45 | 44 | 620 | 74.5% | 22.69 (45) | 45 |
| B2 | 39/47 | 39 | 39 | 365 | 73.9% | 30.28 (12) | 39 |
| B3 | 46/46 | 46 | 46 | 832 | 79.3% | 20.53 (46) | 46 |

배치 정의는 B1=2017-05-12, B2=2018-02-20, B3=2018-04-12이며 파일 3개만 명시적으로 읽는다.
후기/전기 slope 비율은 두 구간 모두 감소하고 전기 slope <−0.000001 Ah/cycle일 때만 정의한다. B2처럼 처음에 capacity가 상승/평탄한 셀은 비율에서 빠지므로 비율 median의 유효 n을 함께 읽어야 한다.
관측위치는 (knee−적합 첫cycle)/(적합 마지막cycle−적합 첫cycle)다. 수명 label 대비 비율이 아니다. B1/B2/B3의 유한 life label 수는 각각 46/39/44다. 결측 label은 관측길이로 대체하지 않았다.

## 고용량 및 이상점 분리

- 고용량 경험 셀은 9개(B1c18, B2c22, B2c23, B2c35, B2c36, B2c37, B2c38, B2c39, B2c40)이고 총 고용량 제외점은 12개다. 그 밖의 고립 spike 제외점은 13개다.
- B1 고용량 strata: 1셀, 적합가능 1, 후보 1; 이 값을 정상 strata 결론에 혼합하지 않는다.
- B2 고용량 strata: 8셀, 적합가능 6, 후보 5; 이 값을 정상 strata 결론에 혼합하지 않는다.

## Window와 cutoff 민감도

- median 7 vs 15 window에서 양쪽 모두 후보이고 Knee 위치 차이가 관측 span의 5% 이하이면 window-stable로 표시한다. 최초 QD≤0.88까지 포함하는 대안 cutoff와 fit 시작cycle 50도 별도 비교했다. 최초 EOL 통과가 없으면 full과 동일하다. 이 민감도는 missing life가 완성됐음을 의미하지 않는다.
- B1: window 안정 후보 44/44, |Δknee| median/max=0/2 cycle; EOL-cut 안정 44/44, |Δ| median/max=0/0; start50 |Δ| median/max=2/7.
- B2: window 안정 후보 39/39, |Δknee| median/max=1/2 cycle; EOL-cut 안정 29/39, |Δ| median/max=18/55; start50 |Δ| median/max=4/11.
- B3: window 안정 후보 46/46, |Δknee| median/max=0/1 cycle; EOL-cut 안정 46/46, |Δ| median/max=0/0; start50 |Δ| median/max=2/5.

## 대표 그래프 해석 근거

대표 그래프는 B1/B2/B3에서 고용량 셀을 제외한 유한 life label의 최단·최장 셀이다. 붉은 hinge와 점선 단일 fit의 오차, 전·후 기울기, EOL 선, raw vs smoothed를 함께 확인할 수 있다.
- B1c20: label life=534, 관측끝=533, candidate_knee, 후보=418, 전/후 slope=-0.276/-0.716 mAh/cycle, 직선 SSE 대비 개선=92.0%.
- B1c4: label life=1227, 관측끝=1226, candidate_knee, 후보=898, 전/후 slope=-0.040/-0.086 mAh/cycle, 직선 SSE 대비 개선=86.0%.
- B2c19: label life=392, 관측끝=411, candidate_knee, 후보=308, 전/후 slope=-0.143/-2.108 mAh/cycle, 직선 SSE 대비 개선=94.4%.
- B2c34: label life=1186, 관측끝=1252, candidate_knee, 후보=961, 전/후 slope=-0.062/-0.606 mAh/cycle, 직선 SSE 대비 개선=98.1%.
- B3c28: label life=541, 관측끝=540, candidate_knee, 후보=379, 전/후 slope=-0.238/-0.489 mAh/cycle, 직선 SSE 대비 개선=95.2%.
- B3c38: label life=1935, 관측끝=1934, candidate_knee, 후보=1523, 전/후 slope=-0.043/-0.289 mAh/cycle, 직선 SSE 대비 개선=97.3%.

## 충전 정책과 열화 속도

- `policy_degradation_correlations.csv`는 고용량 제외/고용량 분리/고용량 제외+유한label strata, pooled 및 배치내 Spearman 상관을 유효 셀 n과 함께 보존한다. 후기 slope와 Knee는 label이 없어도 관측 기술량으로 계산할 수 있다. 파일 내 가변 정책/slow-cycle 진단 셀은 고정 2단계 정책 숫자를 임의 부여하지 않는다.
- 아래는 정상 범위 셀의 pooled 주요 비교다. 음의 slope와 policy의 음의 상관은 더 빠른 capacity 감소와 연결된다. 여러 비교의 p는 보정하지 않은 탐색적 값이며, 배치·프로토콜·SOC 전환 조건의 교란이 있어 원인효과로 해석하지 않는다.

| 정책 | 초기 2–100 slope: rho (n) | 후기 20% slope: rho (n) |
|---|---:|---:|
| c1 | -0.187 (130) | -0.083 (130) |
| c2 | -0.173 (130) | -0.080 (130) |
| switch_soc_pct | 0.194 (130) | 0.096 (130) |
| c_eff_0_80 | -0.269 (130) | -0.242 (130) |

배치내 주요 후기20% slope 상관:

| 배치 | C1 rho (n) | C2 rho (n) | switch rho (n) |
|---|---:|---:|---:|
| B1 | -0.554 (45) | 0.036 (45) | 0.239 (45) |
| B2 | -0.197 (39) | 0.394 (39) | 0.024 (39) |
| B3 | -0.537 (46) | 0.466 (46) | 0.326 (46) |

- 후기 속도와 C1의 pooled 상관은 약한데 B1/B3 내에서는 더 음수인 연결이 나타난다. 배치별 capacity baseline·관측길이·정책 분포 차이가 혼합되므로 pooled 값만으로 정책 영향이 없다고 결론내릴 수 없다. C2의 방향도 C1과 다르게 나타나며, 두 단계의 current와 switch SOC를 동시에 설계한 실험이라는 점을 고려해야 한다.
- B1c0는 최적 hinge의 SSE 개선이 13.9%, slope 비율도 기준보다 작아 후보로 분류되지 않는다. B1c4는 관측 마지막 QD가 EOL보다 높고 완만한 가속 후보가 나타난다. 따라서 모든 후보를 EOL 직전의 급격한 붕괴라고 동일시하지 않는다. B2c19/B2c34는 후기 붉은 hinge가 뚜렷하게 더 가파르고, B3c28은 처음부터 감소가 상당한 셀이라 slope 비율이 다른 대표보다 작다.

## 해석 한계

- 하나의 hinge는 초기 conditioning·중기 plateau·다단계 열화 등 복잡한 곡선을 두 구간으로 축약한다. Knee 위치는 관측 종료길이와 분석 window/cutoff에 의존한다. 최소 전후길이 조건 때문에 극초기/종료 직전 전환은 찾기 어렵다.
- 일부 terminal 급락은 고립 spike 조건을 충족하지 않아 보존된다. 실제 최후 열화와 기록 이상을 summary만으로 구별할 수 없다. `terminal_qd_jump_gt_0_10` 표시 셀과 start/EOL-cut 민감도를 확인해야 한다.
- 관측 중단 셀의 끝은 수명 종료와 다르다. 실험 label·관측길이·EOL 값은 독립적으로 보존하며, label이 없는 셀(가변충전/진단 셀 포함)의 결과를 수명 상관에 사용하지 않는다.
- 다중 비교, policy 반복/배치 교란, smoothing에 따른 오차 상관, label endpoint 이상으로 인해 상관은 설명적 연결이다. 실제 충전 전류 파형이나 물리기전을 확인하려면 cycle 수준의 전류/전압 추가 검증이 필요하다.
- 원본 파일/노트북을 변경하지 않았고 모델 학습을 하지 않았다. 모든 CSV와 PNG는 이 script로 재현한다.

## 산출물

- `cell_knee_metrics.csv`: 셀별 QC·label·정책·기울기·후보·민감도
- `knee_fit_variants.csv`: 모든 셀의 4개 fit variant
- `batch_knee_statistics.csv`: 전체/정상 범위/고용량 strata별 정량 요약
- `quality_flags.csv`: 제외점의 원시값과 사유
- `policy_degradation_correlations.csv`: 유효 n 포함 탐색적 Spearman
- `representative_cells.csv`: 대표 선정과 정량 근거
- `verification_audit.json`: 직접 least-squares 6셀 및 합성 Knee/직선/단기관측/cutoff 검증
- `qd_full_curves_by_batch.png`, `knee_representative_fits.png`, `degradation_acceleration_and_sensitivity.png`, `policy_vs_early_and_late_degradation.png`
