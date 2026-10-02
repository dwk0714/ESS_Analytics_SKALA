# 선정 변수와 제공 수명 라벨의 상관 도표

선정된 ΔQ 로그 분산, cycle10 실측 RMS C-rate, 초기 상대 용량 MAD의 표본 내 관련성을 자료용 도표로 정리했다. 대상은 B1=2017-05-12, B2=2018-02-20, B3=2018-04-12이다. 파일 내139개 기록 중 제공된 양수·유한 수명 라벨129개를 사용하며 결측10개는 채우지 않는다. 배치별 표본 수는46/39/44다. 교차 파일 물리적 셀 독립성과 라벨의 실제 EOL 의미는 확정하지 않았다.

## 결과와 해석

| 변수 | 전체 Pearson | 전체 Spearman | 배치 중심화 Pearson | 배치 중심화 Spearman | VIF |
|---|---:|---:|---:|---:|---:|
| dq100_10_log_variance | -0.850609 | -0.892074 | -0.773090 | -0.801879 | 1.893292 |
| cycle10_rms_c_timeweighted | -0.431664 | -0.322946 | -0.460654 | -0.469497 | 1.423698 |
| capacity_mad_relative | -0.521803 | -0.649046 | -0.406149 | -0.357410 | 1.453890 |

ΔQ 로그 분산은 전체 및 세 배치에서 음의 관계가 유지된다. RMS 전류와 MAD의 관계는 배치마다 강도가 다르다. 특히 B3 상대 MAD의 Spearman은 −0.077172로 약하므로, 전체 Pearson만으로 모든 배치에서 안정적인 순위 신호라고 말할 수 없다. VIF1.42–1.89는 이 세 변수의 큰 선형 중복이 보이지 않는다는 진단이며 MAD의 추가 예측 가치를 검증한 결과는 아니다.

배치 중심화는 각 변수와 목표에서 해당 배치 평균을 모두 뺀 뒤129개를 합쳐 상관을 계산한다. 중심화 Spearman은 그렇게 변환한 값들의 순위 상관이며 배치별 Spearman 평균이 아니다. 집단 평균 차이만을 제거한 설명적 비교이며, 정책·SOC·시험 조건을 모두 통제하거나 인과 효과를 추정하지 않는다.

원 목표 도표의 y는 제공 cycle_life 자체다. 별도 log10 목표 도표는 먼저 양수 수명을 log10으로 바꾼 값에 대한 Pearson을 표시한다. log10 target 중심화는 로그 변환 후 배치 평균을 빼며, 중심화 원 목표에 로그를 취하지 않는다. 양의 원 목표에 대한 로그는 순서를 보존하므로 조정 없는 Spearman은 원 목표와 같다. 목표 변환으로 표본 내 상관이 커져도 검증 성능이 좋아진다고 결론내릴 수 없다.

## 도표와 자료 사용

- 01_selected_features_vs_cycle_life: 세 변수와 제공 수명의 산점도. 배치별 색과 유효 n, 전체 Pearson/Spearman을 표기했다.
- 02_batch_correlation_comparison: 전체/B1/B2/B3/배치 중심화의 Pearson과 Spearman을 같은 범위−1…1로 비교한다.
- 03_feature_redundancy_and_vif: 변수 간 Pearson heatmap과 VIF 진단이다. 목표값은 VIF 계산에 쓰지 않는다.
- 04_selected_features_vs_log10_life: log10 목표 산점도이며 원 목표 그림과 분리했다.

각 그림은 300dpi PNG, 글자와 점을 보존한 SVG, PDF로 저장했다. ΔQ 특징에는 추가 로그를 적용하지 않았다. RMS 단위는 C-rate다. MAD는 원 CSV의 fraction을 축에서100배하여 %로 표시한다. 산점도의 점선은 표시된 표본의 탐색적 직선 추세이며, 학습·검증된 수명 회귀 모델이나 인과 효과가 아니다. 신뢰구간·검증 R²·예측 성능은 표시하지 않는다.

plotting_data.csv는129행의 ID·배치·원 변수·목표·표시용 MAD%·log10 목표다. feature_target_correlations.csv는30개 상관행, feature_pair_correlations.csv는3×3 변수 쌍의 Pearson/Spearman, selected_vif.csv는3개 VIF다. feature_dictionary_selected.csv에 원 계산식·단위·선정 이유·한계를 보존했다. verification.json은 행 정렬과 기존 상관/VIF 대조, 입력 및 기존 노트북 보존을 기록한다.

## 재현

프로젝트 루트에서 다음 명령을 실행한다. 기존 .venv의 NumPy·pandas·SciPy·Matplotlib·IPython을 사용하며 설치와 MAT 재로딩이 없다.

```sh
.venv/bin/python analysis/ess_selected_feature_plots.py
```

도표와 표를 새로 계산하고 실행 결과가 포함된 새32번 노트북도 생성하려면 다음 명령을 사용한다. 이 옵션은32번 파일을 다시 작성한다.

```sh
.venv/bin/python analysis/ess_selected_feature_plots.py --notebook
```

32-ESSHealth-Feature-Correlation.ipynb의 셀을 실행해도 같은 로컬 입력과 코드로 자료를 다시 생성한다. 원본 MAT,30번 scratch,31번 EDA 노트북과 기존 분석 결과는 수정하지 않는다. 이 작업은 도표 및 통계 재현이며 수명 예측 모델 학습을 수행하지 않는다.
