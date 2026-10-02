# 원본 데이터와 특징 CSV

**현재 archive에 있는 기존 MAT 세 파일을 그대로 사용합니다.** 새 MAT를 만들거나 원본을 변경하지 않습니다.

## 원본 확보

실제 과제 데이터 출처: [Kaggle — MIT-Stanford Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle).

| 배치 | 사용할 파일 | 표시 크기 | 전체 셀 / 수명 정답 있음 |
|---|---|---:|---:|
| B1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3.03GB | 46 / 46 |
| B2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 2.02GB | 47 / 39 |
| B3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3.24GB | 46 / 44 |

Kaggle에는 네 파일이 있습니다. `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat`는 제외합니다. 다운로드 절차는 Kaggle의 안내를 따릅니다.

원 연구 출처: [MATR 공식 데이터 프로젝트](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204), [Severson 원논문](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf).
공식 포털에서 과제 B2=2018-02-20의 파일은 “Low rate data used to generate figure 4” 아래에 있습니다. 논문 기본 배치2=2017-06-30을 대신 사용하면 안 됩니다.

원본 세 파일은 합계8,284,609,982바이트입니다. [GitHub 일반 Git의 파일 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)으로 MAT를 코드 저장소에 커밋하지 않습니다. `data/raw`는 로컬 배치 위치이며 `.gitignore`로 제외됩니다.

## 원본 위치 지정

새로 받은 원본은 `data/raw`에 두거나 명령에 다른 폴더를 지정합니다. 현재 작업 환경의 기존 `../archive`를 그대로 읽을 수 있어8.3GB를 다시 복사할 필요가 없습니다.

```bash
python -m src.pipeline eda --raw-dir ../archive
python -m src.pipeline features --raw-dir ../archive
```

`--raw-dir`를 생략하면 `data/raw`를 먼저 찾고, 기존 형제 폴더 `../archive`가 있으면 사용합니다. 노트북에서도 이 경로를 표시하고 필요하면 직접 변경할 수 있습니다.

## 정답과 가공 표

MAT의 셀별 `cycle_life`가 수명 정답 y입니다. 초기 ΔQ·RMS 및 추가 통계는 입력 후보 X입니다. 정답이 없는10개 기록은 빈 값으로 유지하고 학습·오차 계산에서 제외합니다. 관측 길이로 수명을 만들어 채우지 않습니다.

`processed/cell_features.csv`에는 셀당 한 행으로 다음 값을 저장합니다.

- 셀 ID(`B1c20` 등), 배치, 충전 프로토콜, 원본 수명 정답.
- 최종4개 입력: `dq100_10_log_variance`, `cycle10_rms_c_timeweighted`, `dq100_10_q05_ah`, `dq100_10_mean_ah`.
- 기존 MAD와 IR을 제외한 순위후보9개. 후보 전체를 최종 추론에 넣지는 않습니다.
- 후보 선정은 B1 학습 부분에서만 수행합니다. 다른 배치의 endpoint 후보 결측은 보관하며 최종4개 입력은 모두 유한합니다.
- 원본 재추출 시 초기 품질 확인 결과. 이 열은 모델 입력이 아닙니다.

동봉 CSV는 기존 MAT 세 파일에서 초기 특징을 재추출해 저장한 자료입니다. 원본 파일명·크기·SHA256은 `../results/source_manifest.json`, 행 수와 결측 확인 결과는 `../results/data_audit.json`에 기록했습니다. `processed/provenance.json`은 최초 DAY1 자료를 제출 구조로 옮긴 시점의 이력이며, 현재 실행 상태를 나타내는 파일이 아닙니다.

| 재현 범위 | 필요한 자료 |
|---|---|
| 원본부터 EDA·특징 추출 | 기존 MAT 세 파일 + 코드·노트북 |
| 특징 추출 이후 학습·평가 | 동봉 특징 CSV + 코드·설정 |

원본 재추출은 `02_feature_engineering.ipynb`에 읽기·수식·계산·저장 순서로 보여줍니다. EDA는 전체 곡선을 설명할 수 있지만 **모델용 특징은 cycle100 이후 관측을 사용하지 않습니다.**
