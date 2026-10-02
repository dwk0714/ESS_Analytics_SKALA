# 세 배치 cycle10 실제 충전 전류 패턴 EDA

## 정의와 범위

- B1=2017-05-12, B2=2018-02-20, B3=2018-04-12의 139셀만 명시적으로 읽는다. `summary.cycle==10`의 행과 같은 인덱스에서 `cycles.I/t` 파형만 선별 읽는다. 배열 길이/필드 기록 수/NaN/비양수 dt를 별도로 기록한다.
- `I`는 A가 아니라 정격 1.1A로 정규화된 C-rate, `t`는 분이다. [원저자 cell_analysis 변환](https://github.com/chueh-ermon/BMS-autoanalysis/blob/c82ab7704211a6aee75bd926714b82d1f95e1a0e/cell_analysis.m)과 [로컬 도메인 검토](../domain_review.md)에 근거한다.
- 첫 I<−0.5C 방전 시작 전의 파형에서 인접 두 sample 모두 I>0.1C이고 유한 I/t·dt>0인 interval만 active 충전으로 집계한다. 휴지/낮은 전류/invalid interval은 active 시간 분모에 포함하지 않는다.
- 각 interval에서 전류가 선형으로 바뀐다고 근사하여 평균은 dt·(I0+I1)/2, 제곱평균은 dt·(I0²+I0I1+I1²)/3으로 적분한다. RMS=sqrt(제곱평균), 시간 가중 분산=제곱평균−평균²이다. 5C 초과 fraction도 선형 threshold 교차를 반영한 active 시간 비중이다.
- 특징은 **cycle10 한 회 측정값**이다. 초기100 평균·SOC별 CC 특징·전체 충전정책의 전생애 대표값으로 부르지 않는다. rest는 제거되지만 양의 CC/CV 구간은 구분 없이 포함되므로 batch/변형 schedule에서 동일 프로토콜 구간이라는 보장은 없다.
- 원문 life NaN은 그대로 보존한다. 초기2–100 QD slope와 후기20% slope는 앞선 knee EDA 셀 테이블에서 결합한다. 후기 slope는 설명용 사후 outcome이며 초기 예측 입력이 아니다.

## 유효 표본 및 특징 분포

| 배치 | 유효 cycle10 / 전체 | 유한 life | 평균 C median | RMS C median | active duration median (min) | >5C active-time median |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 46/46 | 46 | 2.339 | 3.121 | 25.005 | 15.5% |
| B2 | 47/47 | 39 | 2.390 | 3.141 | 24.670 | 14.6% |
| B3 | 46/46 | 44 | 2.429 | 3.164 | 24.097 | 15.7% |

batch_stats.csv의 mean/median/std/min/max와 유효 n은 파일 내 셀간 분포다. current_features.csv의 variance는 한 파형 내부의 active-time 전류 분산이며 서로 다른 분산이다.

## 실제 전류 특징과 label/열화의 연결

아래는 전체 유효 파형의 탐색적 Spearman이다. 수명 분석은 유한 label만, slope 분석은 해당 유효 slope만 pairwise 사용한다. 전체/배치별 Pearson·Spearman과 n, 고용량 경험 셀 제외 민감도는 correlations.csv에 모두 있다.

| 배치 | 특징 | life rho (n) | 초기2–100 slope rho (n) | 후기20% slope rho (n) |
|---|---|---:|---:|---:|
| ALL | mean_c_timeweighted | -0.135 (129) | -0.389 (139) | -0.118 (139) |
| ALL | rms_c_timeweighted | -0.323 (129) | -0.399 (139) | -0.273 (139) |
| ALL | active_charge_duration_min | 0.131 (129) | 0.352 (139) | 0.089 (139) |
| ALL | time_fraction_above_5c | -0.191 (129) | -0.046 (139) | -0.261 (139) |
| B1 | mean_c_timeweighted | -0.717 (46) | -0.442 (46) | -0.688 (46) |
| B1 | rms_c_timeweighted | -0.856 (46) | -0.563 (46) | -0.815 (46) |
| B1 | active_charge_duration_min | 0.711 (46) | 0.334 (46) | 0.671 (46) |
| B1 | time_fraction_above_5c | -0.332 (46) | -0.147 (46) | -0.190 (46) |
| B2 | mean_c_timeweighted | -0.279 (39) | -0.339 (47) | 0.099 (47) |
| B2 | rms_c_timeweighted | -0.382 (39) | -0.303 (47) | -0.027 (47) |
| B2 | active_charge_duration_min | 0.303 (39) | 0.288 (47) | -0.084 (47) |
| B2 | time_fraction_above_5c | -0.157 (39) | 0.009 (47) | -0.267 (47) |
| B3 | mean_c_timeweighted | 0.027 (44) | -0.507 (46) | 0.178 (46) |
| B3 | rms_c_timeweighted | -0.241 (44) | -0.527 (46) | -0.005 (46) |
| B3 | active_charge_duration_min | -0.036 (44) | 0.493 (46) | -0.208 (46) |
| B3 | time_fraction_above_5c | -0.302 (44) | -0.099 (46) | -0.332 (46) |

## 대표 파형 해석

각 배치에서 유한 제공 life label이 가장 짧은 셀과 가장 긴 셀의 cycle10 파형 두 개를 겹쳐 그린다. x축은 cycle record 시작 이후 실제 경과 분, y축은 C-rate다. 첫 방전 이전의 rest도 그래프에는 남겨 active 구간 정의와 구별할 수 있다.
- B1c20 (shortest label): label=534, policy=5.4C(80%)-5.4C; 평균/RMS=2.567/3.442 C, active=22.646분, >5C 시간=39.1%.
- B1c4 (longest label): label=1227, policy=4C(80%)-4C; 평균/RMS=2.224/2.768 C, active=26.338분, >5C 시간=0.0%.
- B2c19 (shortest label): label=392, policy=6C(60%)-3C; 평균/RMS=2.399/3.268 C, active=24.528분, >5C 시간=24.4%.
- B2c34 (longest label): label=1186, policy=5.6C(26%)-4.5C-newstructure; 평균/RMS=2.510/3.238 C, active=23.071분, >5C 시간=11.8%.
- B3c28 (shortest label): label=541, policy=3.7C(31%)-5.9C-newstructure; 평균/RMS=2.476/3.279 C, active=23.409분, >5C 시간=21.4%.
- B3c38 (longest label): label=1935, policy=5C(67%)-4C-newstructure; 평균/RMS=2.485/3.210 C, active=23.444분, >5C 시간=14.5%.

- B1 대표의 주 충전 단계는 단수명 label 셀의 약5.4C와 장수명 label 셀의 약4C이며, 두 곡선 모두 후반1C 및 감쇠 구간이 보인다. 제공 label과 연결되는 초기 실측 전류 차이를 볼 수 있지만, B1 연속 실험 앞부분/관측 EOL 여부를 고려해야 한다.
- B2 최단/최장 대표의 active 평균은 2.399/2.510C, RMS는 3.268/3.238C인데 label은392/1186이다. B3도 active 평균2.476/2.485C로 비슷하지만 label541/1935다. 최고값 또는 평균 하나로 life 순위를 설명할 수 없으며 단계 적용 시간과 순서를 함께 봐야 한다.
- 전체 RMS–life 연결은 Pearson r=−0.432, Spearman rho=−0.323(n129)다. 배치내 Spearman은 B1−0.856(n46), B2−0.382(n39), B3−0.241(n44)로 같지 않다. 실제 파형 특징도 배치·정책 조건별로 연결 강도가 다르다.
- RMS–초기2–100 QD slope Pearson은 전체 −0.672(n139)지만 고용량 경험9셀 제외 시 −0.206(n130)으로 약해진다. 후기20% slope는 전체 −0.263(n139), 고용량 제외 −0.421(n130)이다. RMS–life는 고용량 제외 −0.430(n128)으로 전체 −0.432(n129)와 비슷하다. 각 표본과 QC 정의를 구분하며 전체 초기 slope 관계를 안정된 전류–열화 효과로 단정하지 않는다.

## QC와 해석 한계

- B1: I/t 비유한 sample=0/0, 방전 전 비양수 dt interval=0, summary/cycles 개수 불일치 셀=0, 방전 boundary 누락=0.
- B2: I/t 비유한 sample=0/0, 방전 전 비양수 dt interval=0, summary/cycles 개수 불일치 셀=0, 방전 boundary 누락=0.
- B3: I/t 비유한 sample=0/0, 방전 전 비양수 dt interval=0, summary/cycles 개수 불일치 셀=0, 방전 boundary 누락=0.
- sample 간 시간 간격이 균등하지 않아 전류 sample의 단순 평균 대신 time weighting을 사용했다. current가 양수인 모든 구간을 하나로 집계하므로 다양한 전류 단계·CC/CV·진단·rest의 분리된 기전까지 알 수 없다.
- 5C 근처 plateau의 작은 측정/제어 변동은 엄격한 >5C fraction을 바꾼다. 5C 초과 fraction을 정확한 고속 충전 SOC 비중과 동일시하지 않는다. 파형상 peak와 RMS/평균은 서로 다른 정보를 담는다.
- 특징–수명/후기 속도의 상관은 배치·SOC 전환·휴지·시험조건이 혼합된 설명적 연결이다. 높은 전류가 특정 물리기전이나 짧은 수명을 유발했다는 인과결론과 held-out 예측 성능은 검증하지 않았다. 공급 life label과 관측된 EOL은 도메인 검토에서 구분한다.
- file-local 139행은 파형 가용 단위이며 물리적 셀 독립성이 확정됐다는 뜻은 아니다. p-value는 다중 비교 보정 없는 탐색적 값이다. life NaN 10셀은 life 상관에서 제외하고 파형/QD 설명에만 포함한다.
- 원본과 EDA 노트북을 변경하지 않았고 추가 모델 학습을 수행하지 않았다. 이 script가 current/ 산출물만 작성한다.

## 산출물

- current_features.csv: 139셀 cycle10 특징·QC·life·기울기
- cycle10_predischarge_profiles.csv: 방전 전 실제 sample time/current 파형
- batch_stats.csv: 배치·고용량 제외 strata의 유효 n 및 셀간 분포
- correlations.csv: 전체/배치별 Pearson·Spearman과 pair n
- representative_cells.csv 및 cycle10_charge_current_representatives.png: 최단/최장 label 파형 비교
- methodology.json: interval/적분/threshold/단위 정의
