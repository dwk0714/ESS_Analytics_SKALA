# 모델·앙상블 비교 실험

2026-10-02에 제공 수명 라벨과 기존 분할을 유지한 비교, 명확한 미완료 B1 셀을 제외한 민감도 분석을 수행했다. 기존 기본 모델·성능 파일은 보존했다.

## 비교 조건

- 주 비교: B1 학습35개·Hold-out11개, B2 평가39개, B3 평가44개.
- 민감도 분석: B1 학습28개·Hold-out8개. B2/B3 평가 대상은 동일하다.
- 모든 모델은 log10(cycle_life)를 학습하고 원래 사이클 단위로 역변환해 MAPE·MAE·RMSE를 계산했다.
- B1 충전 프로토콜을 분리한5-fold CV로만 모델·설정을 선택했다. Hold-out·B2·B3 점수로 후보를 재선택하지 않았다.
- XGBoost는 입력1/2/3개 각각40개 설정, 총120개 설정을 검토했다. 트리 깊이1~3, 학습률0.03/0.05/0.1, 트리 수50/100/200/300, min_child_weight3/5, reg_lambda1/10/30을 사용했다.
- Voting은 ΔQ 단독 OLS·기존 Ridge·선택 XGBoost의 사이클 단위 예측을 같은 비중으로 평균했다.
- Stacking은 같은 세 모델의 로그 수명 예측을 결합했다. 각 CV 학습 부분에서 프로토콜을 분리한 내부3-fold OOF 예측으로 Ridge 결합 모델을 학습했다. 외부 CV 평가 셀은 결합 모델을 학습하지 않았다. 결합 모델 alpha0.1/1/10/100 중 B1 CV로1을 선택했다.

## 결과 — MAPE (%), 낮을수록 좋음

### 제공 라벨·기존 분할

| 모델 | B1 CV | B1 Hold-out | B2 | B3 |
|---|---:|---:|---:|---:|
| Linear Regression — ΔQ 단독 | 7.211 | 18.203 | 28.864 | 12.595 |
| Linear Regression — 2변수 | 7.109 | 16.699 | 32.652 | 13.398 |
| Ridge — 기존 | 6.987 | 16.153 | 33.561 | 13.742 |
| XGBoost | 9.342 | 9.474 | 34.466 | 17.239 |
| Voting — 단순 평균 | 6.830 | 12.672 | 32.053 | 13.161 |
| Stacking — Ridge 결합 | 6.495 | 12.561 | 27.591 | 13.884 |

### 미완료 B1 셀 제외 — 별도 민감도 분석

| 모델 | B1 CV | B1 Hold-out | B2 | B3 |
|---|---:|---:|---:|---:|
| Linear Regression — ΔQ 단독 | 8.735 | 10.705 | 27.900 | 13.577 |
| Linear Regression — 2변수 | 8.285 | 9.159 | 31.104 | 12.540 |
| Ridge — 기존 | 8.331 | 8.892 | 32.154 | 12.928 |
| XGBoost | 9.249 | 9.595 | 31.328 | 17.891 |
| Voting — 단순 평균 | 8.520 | 9.341 | 29.985 | 12.725 |
| Stacking — Ridge 결합 | 8.518 | 9.305 | 29.626 | 14.880 |

## 해석과 범위

제공 라벨을 유지한 비교에서는 Stacking이 B1 CV 기준6.495%로 가장 낮았고, B2도 기존 Ridge33.561%에서27.591%로5.970%p 개선했다. B3는 기존 Ridge13.742%보다 소폭 높은13.884%였다. 과제 Target9.1% 대비 B2 Gap은+18.491%p로 목표에는 미달했다.

미완료 셀 제외 조건의 B1 CV 최저 모델은2변수 OLS(8.285%)였다. 이 모델의 B2는31.104%였다. 해당 조건에서 B2가 가장 낮았던 ΔQ 단독 OLS(27.901%)를 B2 점수만 보고 최종 모델로 선택하지 않았다. 두 조건은 학습·Hold-out 표본 수가 달라 B1 점수를 직접적인 개선율로 비교할 수 없다.

민감도 분석은 원저자 로딩 코드가 후속 실험을 연결하거나 미완료로 제외한 B1c0/1/2/3/4/8/10/12/13/22를 별도 제외했다. 해당10개 기록의 마지막 Qd가0.90Ah보다 높음을 원본에서 확인했다. 누락 수명을 채우거나 현재 B2의 다른 날짜 실험에 연속 실험 보정값을 적용하지 않았다. 제공 라벨 값 자체는 변경하지 않았다. 나머지 B1/B3의 기준 근처 종료도 엄격한0.88Ah 통과 관측과 구분한다.

B1 CV는 후보 선택에 사용한 개발 점수이며 완전한 nested tuning 평가가 아니다. B2/B3는 이전 EDA와 평가에서도 확인했으므로 새로 확보한 미관측 외부 배치의 성능을 보장하지 않는다.

## 재현

추가 패키지는 저장소 루트의 requirements-comparison.txt로 설치한다. XGBoost가 macOS에서 OpenMP를 찾지 못하면 공식 설치 안내의 libomp 설정이 필요하다. 이번 환경에서는 기존 scikit-learn wheel의 OpenMP 라이브러리 경로를 사용했다.

```bash
pip install -r requirements-comparison.txt
python -m src.compare_models --output results/runs/new_comparison
python -m src.compare_ensembles --comparison results/runs/new_comparison
```

현재 macOS 환경에서 사용한 명령:

```bash
DYLD_LIBRARY_PATH="../.venv/lib/python3.11/site-packages/sklearn/.dylibs" ../.venv/bin/python -m src.compare_models --output results/runs/new_comparison
DYLD_LIBRARY_PATH="../.venv/lib/python3.11/site-packages/sklearn/.dylibs" ../.venv/bin/python -m src.compare_ensembles --comparison results/runs/new_comparison
```

기존 결과가 있는 폴더는 덮어쓰지 않는다. 총 비교 계산 시간은 단일 모델 약5.695초, 앙상블 약1.789초였다. 패키지 설치·코드 작성·검증 시간은 포함하지 않는다.

## 저장 파일

- all_model_comparison.csv: 두 조건의6모델 비교, 총12행.
- target_audit.csv: 실제 종료 용량·임계값 통과·제외 대상 확인.
- provided_labels/, unfinished_b1_excluded_sensitivity/: 단일 모델·예측·설정.
- ensembles/: 앙상블 모델·예측·설정.
- experiment.json: 탐색 범위·실행 환경·원래 파일 해시.
- verification.json: 저장 모델12개 재로드, 예측·점수 재계산, 프로토콜 분리와 원래 파일 보존 확인.

[원저자 로딩 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb) · [XGBoost 설치 안내](https://xgboost.readthedocs.io/en/stable/install.html)
