# DAY2 파이프라인

```mermaid
flowchart LR
    subgraph S1["1. 특징 추출과 데이터 전처리"]
        direction TB
        A1[기존 MAT 읽기] --> A2[파일과 셀 ID 확인]
        A2 --> A3[초기 특징 계산·결측 확인]
        A3 --> A4[B1 학습·holdout 분리]
    end
    subgraph S2["2. 모델 학습"]
        direction TB
        B1[B1 반복 그룹 CV] --> B2[학습 부분에서 변수·XGB 설정 선택]
        B2 --> B3[그룹 OOF 예측으로 메타 Ridge 학습]
        B3 --> B4[최종 Stacking 저장]
    end
    subgraph S3["3. 검증과 평가"]
        direction TB
        C1[동일 모델로 holdout·B2 예측] --> C2[B3 선택적 추가 평가]
        C2 --> C3[MAPE·MAE·RMSE·Gap]
        C3 --> C4[큰 오차 셀·배치 차이 분석]
    end
    subgraph S4["4. 보고서 작성"]
        direction TB
        D1[EDA·변수 선정 근거] --> D2[모델 선정 이유·성능표]
        D2 --> D3[오류 사례·ESS 해석·한계]
        D3 --> D4[README·결과 도표]
    end
    A4 --> B1
    B4 --> C1
    C4 --> D1
```

| 큰 단계 | 할 일 | 코드/노트북 |
|---|---|---|
| 특징 추출·전처리 | 기존 MAT 읽기, 파일과 셀 ID 확인, 초기 특징·결측 확인, B1 그룹 분할 | 01/02 노트북, preprocess/features/splits |
| 모델 학습 | B1 반복 그룹 CV, 학습 부분에서 후보2개·XGB 설정 선택, OOF 메타 Ridge 학습 | 03 노트북, train/train_stacking/stacking |
| 검증·평가 | 동일 저장 모델의 holdout/B2, 선택 B3 평가와 오차·Gap | evaluate |
| 보고서 작성 | EDA/선정 이유/성과표/오류와 ESS 해석 | report, README |

MAPE가 낮을수록 수명 예측이 더 정확합니다. Gap은 Valid−Train, Test−Valid, Test−9.1이며 양수는 뒤 평가의 오차 증가입니다.

파일 확인은 사용할 배치 파일이 맞는지 보는 작업입니다. 셀 ID 확인은 같은 셀의 특징과 수명을 연결하는 작업입니다. B1c20 특징에 B1c21 수명을 붙이면 안 됩니다.

특징 추출·모델 학습·B1 Hold-out 및 B2/B3 평가·보고서 생성을 수행했습니다. 실행 결과는 [성능 보고서](../results/performance_report.md)에 저장했습니다.

최종 입력은 ΔQ 로그분산·cycle10 RMS·ΔQ 5% 분위수·ΔQ 평균입니다. 기저 OLS/Ridge/XGBoost와 메타 Ridge를 학습했습니다. 독립 학습 셀은35개이고 CV는3반복×5fold입니다. B2/B3는 EDA와 여러 비교 실험에서 이미 확인했으므로 새로운 미관측 외부 검증이 필요합니다.
