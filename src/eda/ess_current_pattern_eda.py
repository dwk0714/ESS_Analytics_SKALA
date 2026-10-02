#!/usr/bin/env python3
"""Cycle-10 measured charging-current EDA for the three explicitly named files.

Reads only cycle-10 I/t references plus summary cycle coordinates. I is stored
in C-rate, t in minutes. No model training or early-100 averaging is performed.
Run: python -m src.pipeline eda --raw-dir ../archive
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ess_current_mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'ess_current_cache'))
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FILES={'B1':'2017-05-12_batchdata_updated_struct_errorcorrect.mat',
       'B2':'2018-02-20_batchdata_updated_struct_errorcorrect.mat',
       'B3':'2018-04-12_batchdata_updated_struct_errorcorrect.mat'}
FEATURES=['cycle10_mean_c_timeweighted','cycle10_rms_c_timeweighted','cycle10_variance_c2_timeweighted',
          'cycle10_active_charge_duration_min','cycle10_time_fraction_above_5c']
TARGETS=['cycle_life','early_2_100_qd_slope','late20_slope']
METHOD={'batches':FILES,'cycle_number':10,'current_unit':'C-rate','time_unit':'minutes',
        'unit_source':'https://github.com/chueh-ermon/BMS-autoanalysis/blob/c82ab7704211a6aee75bd926714b82d1f95e1a0e/cell_analysis.m',
        'unit_audit':'results/eda/domain_review.md',
        'index_mapping':'Find summary.cycle == 10; use same zero-based row in cycles.I/t; record counts/alignment QC.',
        'charge_boundary':'Samples strictly before the first finite I < -0.5 C.',
        'active_interval':'Both adjacent I > 0.1 C, finite I/t endpoints, and positive dt.',
        'mean':'sum(dt*(I0+I1)/2)/sum(dt)',
        'rms':'sqrt(sum(dt*(I0^2+I0*I1+I1^2)/3)/sum(dt))',
        'variance':'E[I^2]-E[I]^2, active-time population variance',
        'above_5c':'Exact fraction of each active interval above 5 C assuming linear interpolation of endpoints, active-time denominator.',
        'no_early100_averaging':True,'life_missing_not_imputed':True,
        'late20_role':'Retrospective descriptive outcome; never an early-life input.',
        'correlation_subsets':['all_cycle10_valid','no_high_qd_cell_flag'],
        'correlation_units':'One row per file-local cell; physical cell independence is not proved.',
        'limitations':['Cycle 10 is one observed waveform; it need not represent the entire early-100 window or lifetime.',
                       'Positive-current gating includes any pre-discharge CC/CV segments and excludes rests; no SOC or protocol-step annotation is inferred.',
                       'The strict 5 C threshold is sensitive to measurement/control fluctuations near 5 C.',
                       'Exploratory correlations have batch/protocol confounding and unadjusted nominal p-values.']}


def measure_current(i,t):
    i,t=np.asarray(i,float),np.asarray(t,float)
    result={name:np.nan for name in FEATURES}
    result.update(n_current_samples=len(i),n_time_samples=len(t),n_current_nonfinite=int((~np.isfinite(i)).sum()),
                  n_time_nonfinite=int((~np.isfinite(t)).sum()),array_length_match=len(i)==len(t),
                  first_discharge_found=False,first_discharge_index=np.nan,first_discharge_time_min=np.nan,
                  n_predischarge_samples=0,n_predischarge_intervals=0,n_active_intervals=0,
                  n_nonpositive_dt_intervals=0,n_nonfinite_endpoint_intervals=0,
                  n_excluded_rest_or_low_current_intervals=0,charge_window_elapsed_min=np.nan,
                  active_time_fraction_charge_window=np.nan,cycle10_current_feature_valid=False,
                  qc_status='array_length_mismatch' if len(i)!=len(t) else 'unresolved')
    if len(i)!=len(t) or len(i)<3:return result,0
    neg=np.flatnonzero(np.isfinite(i)&(i<-.5))
    end=int(neg[0]) if len(neg) else len(i)
    result.update(first_discharge_found=bool(len(neg)),first_discharge_index=end if len(neg) else np.nan,
                  first_discharge_time_min=float(t[end]) if len(neg) and np.isfinite(t[end]) else np.nan,
                  n_predischarge_samples=end,n_predischarge_intervals=max(0,end-1))
    if end<3:result['qc_status']='insufficient_predischarge_samples';return result,end
    a,b=i[:end-1],i[1:end];left,right=t[:end-1],t[1:end];dt=right-left
    finite=np.isfinite(a)&np.isfinite(b)&np.isfinite(left)&np.isfinite(right)
    positive=(a>.1)&(b>.1);good=finite&(dt>0)&positive
    result.update(n_nonpositive_dt_intervals=int((np.isfinite(dt)&(dt<=0)).sum()),
                  n_nonfinite_endpoint_intervals=int((~finite).sum()),
                  n_excluded_rest_or_low_current_intervals=int((finite&(dt>0)&~positive).sum()),
                  n_active_intervals=int(good.sum()))
    if not len(neg):result['qc_status']='first_discharge_not_found';return result,end
    if good.sum()<2:result['qc_status']='insufficient_active_intervals';return result,end
    a,b,dt=a[good],b[good],dt[good];duration=dt.sum()
    mean=np.dot(dt,(a+b)/2)/duration;second=np.dot(dt,(a*a+a*b+b*b)/3)/duration
    above=np.zeros(len(a));above[(a>5)&(b>5)]=1
    up=(a<=5)&(b>5);down=(a>5)&(b<=5)
    above[up]=(b[up]-5)/(b[up]-a[up]);above[down]=(a[down]-5)/(a[down]-b[down])
    elapsed=t[end]-t[0] if np.isfinite(t[end]) and np.isfinite(t[0]) else np.nan
    result.update(cycle10_mean_c_timeweighted=float(mean),cycle10_rms_c_timeweighted=float(np.sqrt(second)),
                  cycle10_variance_c2_timeweighted=float(max(0,second-mean**2)),
                  cycle10_active_charge_duration_min=float(duration),cycle10_time_fraction_above_5c=float(np.dot(dt,above)/duration),
                  charge_window_elapsed_min=float(elapsed),active_time_fraction_charge_window=float(duration/elapsed) if elapsed>0 else np.nan,
                  cycle10_current_feature_valid=True,qc_status='valid_with_sample_qc' if result['n_nonpositive_dt_intervals'] or result['n_nonfinite_endpoint_intervals'] else 'valid')
    return result,end


def extract(metadata):
    rows=[];profiles={};rawrows=[]
    for batch,file in FILES.items():
        with h5py.File(Path(os.environ['ESS_RAW_DIR']) / file,'r') as f:
            for idx,cref in enumerate(f['batch']['cycles'][:,0]):
                key=f'{batch}c{idx}';g=f[cref];s=f[f['batch']['summary'][idx,0]]
                cyc=s['cycle'][()].ravel();match=np.flatnonzero(np.isclose(cyc,10));
                if len(match)!=1:raise ValueError(f'{key}: actual summary cycle 10 not unique')
                pos=int(match[0]);ni,nt=len(g['I']),len(g['t'])
                if pos>=min(ni,nt):raise ValueError(f'{key}: cycle-10 field unavailable')
                i=f[g['I'][pos,0]][()].ravel();t=f[g['t'][pos,0]][()].ravel();result,end=measure_current(i,t)
                rows.append({'cell_key':key,'batch':batch,'cell_index':idx,'source_file':file,'actual_summary_cycle':10,
                             'cycle_record_index':pos,'summary_cycles_n':len(cyc),'current_cycle_records_n':ni,'time_cycle_records_n':nt,
                             'summary_cycle_records_count_match':len(cyc)==ni==nt,**result})
                profiles[key]={'i':i,'t':t,'predischarge_end':end}
                for j in range(end):
                    rawrows.append({'cell_key':key,'batch':batch,'cycle':10,'sample_index':j,'elapsed_from_record_start_min':t[j]-t[0],
                                    'record_t_min':t[j],'current_c':i[j],'above_active_current_threshold':bool(np.isfinite(i[j]) and i[j]>.1)})
    features=pd.DataFrame(rows).merge(metadata[['cell_key','policy','policy_canonical','cycle_life','life_label_missing',
            'high_capacity_cell','early_2_100_qd_slope','late20_slope']],on='cell_key',validate='one_to_one')
    return features,profiles,pd.DataFrame(rawrows)


def correlations(d):
    rows=[]
    for batch in ['ALL',*FILES]:
        bd=d if batch=='ALL' else d[d.batch==batch]
        for subset,mask in [('all_cycle10_valid',bd.cycle10_current_feature_valid),
                            ('no_high_qd_cell_flag',bd.cycle10_current_feature_valid&~bd.high_capacity_cell)]:
            selected=bd[mask]
            for feat in FEATURES:
                for target in TARGETS:
                    pair=selected[[feat,target]].replace([np.inf,-np.inf],np.nan).dropna();n=len(pair)
                    good=n>=4 and pair[feat].nunique()>1 and pair[target].nunique()>1
                    r,pr=pearsonr(pair[feat],pair[target]) if good else (np.nan,np.nan)
                    rho,ps=spearmanr(pair[feat],pair[target]) if good else (np.nan,np.nan)
                    rows.append({'batch':batch,'subset':subset,'cycle10_current_feature':feat,'target':target,'n':n,
                                 'pearson_r':r,'spearman_rho':rho,'pearson_nominal_p':pr,'spearman_nominal_p':ps,
                                 'p_value_multiple_comparison_adjustment':'none_exploratory'})
    return pd.DataFrame(rows)


def batch_statistics(d):
    rows=[]
    for batch,bd in d.groupby('batch'):
        for subset,selected in [('all',bd),('no_high_qd_cell_flag',bd[~bd.high_capacity_cell])]:
            row={'batch':batch,'subset':subset,'n_cells':len(selected),'n_cycle10_feature_valid':int(selected.cycle10_current_feature_valid.sum()),
                 'n_finite_life':int(selected.cycle_life.notna().sum()),'n_life_nan':int(selected.cycle_life.isna().sum()),
                 'n_count_mapping_qc':int((~selected.summary_cycle_records_count_match).sum()),
                 'n_nonfinite_current_samples':int(selected.n_current_nonfinite.sum()),'n_nonfinite_time_samples':int(selected.n_time_nonfinite.sum()),
                 'n_nonpositive_dt_intervals':int(selected.n_nonpositive_dt_intervals.sum()),'n_missing_discharge_boundary':int((~selected.first_discharge_found).sum())}
            for feat in FEATURES:
                v=selected[feat].dropna();row.update({feat+'_n':len(v),feat+'_mean':v.mean(),feat+'_median':v.median(),feat+'_std_cells':v.std(),feat+'_min':v.min(),feat+'_max':v.max()})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_profiles(d,profiles,out):
    fig,axs=plt.subplots(1,3,figsize=(16,4.7));representatives=[]
    for ax,batch in zip(axs,FILES):
        labeled=d[(d.batch==batch)&d.cycle_life.notna()&d.cycle10_current_feature_valid]
        shortest=labeled.loc[labeled.cycle_life.idxmin()];longest=labeled.loc[labeled.cycle_life.idxmax()]
        for row,color,role in [(shortest,'#ca593d','shortest label'),(longest,'#3a708e','longest label')]:
            key=row.cell_key;p=profiles[key];end=p['predischarge_end'];elapsed=p['t'][:end]-p['t'][0]
            ax.plot(elapsed,p['i'][:end],c=color,lw=1.2,alpha=.9,label=f'{key}: life {row.cycle_life:.0f}\n{row.policy}')
            representatives.append({**row.to_dict(),'representative_role':role})
        ax.axhline(5,c='gray',ls='--',lw=.7);ax.axhline(.1,c='gray',ls=':',lw=.6)
        ax.set(title=f'{batch} | measured cycle 10',xlabel='Elapsed from cycle record start (min)',ylabel='Stored current (C-rate)')
        ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Measured charging profiles before first discharge: shortest/longest finite label per batch',fontsize=12)
    fig.tight_layout();fig.savefig(out/'cycle10_charge_current_representatives.png',dpi=180);plt.close(fig)
    return pd.DataFrame(representatives)


def findings(d,stats,corr,rep,out):
    lines=['# 세 배치 cycle10 실제 충전 전류 패턴 EDA','','## 정의와 범위','',
           '- B1=2017-05-12, B2=2018-02-20, B3=2018-04-12의 139셀만 명시적으로 읽는다. `summary.cycle==10`의 행과 같은 인덱스에서 `cycles.I/t` 파형만 선별 읽는다. 배열 길이/필드 기록 수/NaN/비양수 dt를 별도로 기록한다.',
           '- `I`는 A가 아니라 정격 1.1A로 정규화된 C-rate, `t`는 분이다. [원저자 cell_analysis 변환](https://github.com/chueh-ermon/BMS-autoanalysis/blob/c82ab7704211a6aee75bd926714b82d1f95e1a0e/cell_analysis.m)과 [로컬 도메인 검토](../domain_review.md)에 근거한다.',
           '- 첫 I<−0.5C 방전 시작 전의 파형에서 인접 두 sample 모두 I>0.1C이고 유한 I/t·dt>0인 interval만 active 충전으로 집계한다. 휴지/낮은 전류/invalid interval은 active 시간 분모에 포함하지 않는다.',
           '- 각 interval에서 전류가 선형으로 바뀐다고 근사하여 평균은 dt·(I0+I1)/2, 제곱평균은 dt·(I0²+I0I1+I1²)/3으로 적분한다. RMS=sqrt(제곱평균), 시간 가중 분산=제곱평균−평균²이다. 5C 초과 fraction도 선형 threshold 교차를 반영한 active 시간 비중이다.',
           '- 특징은 **cycle10 한 회 측정값**이다. 초기100 평균·SOC별 CC 특징·전체 충전정책의 전생애 대표값으로 부르지 않는다. rest는 제거되지만 양의 CC/CV 구간은 구분 없이 포함되므로 batch/변형 schedule에서 동일 프로토콜 구간이라는 보장은 없다.',
           '- 원문 life NaN은 그대로 보존한다. 초기2–100 QD slope와 후기20% slope는 앞선 knee EDA 셀 테이블에서 결합한다. 후기 slope는 설명용 사후 outcome이며 초기 예측 입력이 아니다.','','## 유효 표본 및 특징 분포','',
           '| 배치 | 유효 cycle10 / 전체 | 유한 life | 평균 C median | RMS C median | active duration median (min) | >5C active-time median |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in stats[stats.subset=='all'].itertuples():
        lines.append(f'| {r.batch} | {r.n_cycle10_feature_valid}/{r.n_cells} | {r.n_finite_life} | {r.cycle10_mean_c_timeweighted_median:.3f} | {r.cycle10_rms_c_timeweighted_median:.3f} | {r.cycle10_active_charge_duration_min_median:.3f} | {r.cycle10_time_fraction_above_5c_median:.1%} |')
    lines+=['','batch_stats.csv의 mean/median/std/min/max와 유효 n은 파일 내 셀간 분포다. current_features.csv의 variance는 한 파형 내부의 active-time 전류 분산이며 서로 다른 분산이다.','',
            '## 실제 전류 특징과 label/열화의 연결','',
            '아래는 전체 유효 파형의 탐색적 Spearman이다. 수명 분석은 유한 label만, slope 분석은 해당 유효 slope만 pairwise 사용한다. 전체/배치별 Pearson·Spearman과 n, 고용량 경험 셀 제외 민감도는 correlations.csv에 모두 있다.','',
            '| 배치 | 특징 | life rho (n) | 초기2–100 slope rho (n) | 후기20% slope rho (n) |',
            '|---|---|---:|---:|---:|']
    for batch in ['ALL',*FILES]:
        for feat in ['cycle10_mean_c_timeweighted','cycle10_rms_c_timeweighted','cycle10_active_charge_duration_min','cycle10_time_fraction_above_5c']:
            values=[]
            for target in TARGETS:
                row=corr[(corr.batch==batch)&(corr.subset=='all_cycle10_valid')&(corr.cycle10_current_feature==feat)&(corr.target==target)].iloc[0]
                values.append(f'{row.spearman_rho:.3f} ({int(row.n)})')
            lines.append(f'| {batch} | {feat.replace("cycle10_","")} | '+' | '.join(values)+' |')
    lines+=['','## 대표 파형 해석','',
            '각 배치에서 유한 제공 life label이 가장 짧은 셀과 가장 긴 셀의 cycle10 파형 두 개를 겹쳐 그린다. x축은 cycle record 시작 이후 실제 경과 분, y축은 C-rate다. 첫 방전 이전의 rest도 그래프에는 남겨 active 구간 정의와 구별할 수 있다.']
    for r in rep.itertuples():
        lines.append(f'- {r.cell_key} ({r.representative_role}): label={r.cycle_life:.0f}, policy={r.policy}; 평균/RMS={r.cycle10_mean_c_timeweighted:.3f}/{r.cycle10_rms_c_timeweighted:.3f} C, active={r.cycle10_active_charge_duration_min:.3f}분, >5C 시간={r.cycle10_time_fraction_above_5c:.1%}.')
    lines+=['','- B1 대표의 주 충전 단계는 단수명 label 셀의 약5.4C와 장수명 label 셀의 약4C이며, 두 곡선 모두 후반1C 및 감쇠 구간이 보인다. 제공 label과 연결되는 초기 실측 전류 차이를 볼 수 있지만, B1 연속 실험 앞부분/관측 EOL 여부를 고려해야 한다.',
            '- B2 최단/최장 대표의 active 평균은 2.399/2.510C, RMS는 3.268/3.238C인데 label은392/1186이다. B3도 active 평균2.476/2.485C로 비슷하지만 label541/1935다. 최고값 또는 평균 하나로 life 순위를 설명할 수 없으며 단계 적용 시간과 순서를 함께 봐야 한다.',
            '- 전체 RMS–life 연결은 Pearson r=−0.432, Spearman rho=−0.323(n129)다. 배치내 Spearman은 B1−0.856(n46), B2−0.382(n39), B3−0.241(n44)로 같지 않다. 실제 파형 특징도 배치·정책 조건별로 연결 강도가 다르다.',
            '- RMS–초기2–100 QD slope Pearson은 전체 −0.672(n139)지만 고용량 경험9셀 제외 시 −0.206(n130)으로 약해진다. 후기20% slope는 전체 −0.263(n139), 고용량 제외 −0.421(n130)이다. RMS–life는 고용량 제외 −0.430(n128)으로 전체 −0.432(n129)와 비슷하다. 각 표본과 QC 정의를 구분하며 전체 초기 slope 관계를 안정된 전류–열화 효과로 단정하지 않는다.',
            '','## QC와 해석 한계','']
    for r in stats[stats.subset=='all'].itertuples():
        lines.append(f'- {r.batch}: I/t 비유한 sample={r.n_nonfinite_current_samples}/{r.n_nonfinite_time_samples}, 방전 전 비양수 dt interval={r.n_nonpositive_dt_intervals}, summary/cycles 개수 불일치 셀={r.n_count_mapping_qc}, 방전 boundary 누락={r.n_missing_discharge_boundary}.')
    lines+=['- sample 간 시간 간격이 균등하지 않아 전류 sample의 단순 평균 대신 time weighting을 사용했다. current가 양수인 모든 구간을 하나로 집계하므로 다양한 전류 단계·CC/CV·진단·rest의 분리된 기전까지 알 수 없다.',
            '- 5C 근처 plateau의 작은 측정/제어 변동은 엄격한 >5C fraction을 바꾼다. 5C 초과 fraction을 정확한 고속 충전 SOC 비중과 동일시하지 않는다. 파형상 peak와 RMS/평균은 서로 다른 정보를 담는다.',
            '- 특징–수명/후기 속도의 상관은 배치·SOC 전환·휴지·시험조건이 혼합된 설명적 연결이다. 높은 전류가 특정 물리기전이나 짧은 수명을 유발했다는 인과결론과 held-out 예측 성능은 검증하지 않았다. 공급 life label과 관측된 EOL은 도메인 검토에서 구분한다.',
            '- file-local 139행은 파형 가용 단위이며 물리적 셀 독립성이 확정됐다는 뜻은 아니다. p-value는 다중 비교 보정 없는 탐색적 값이다. life NaN 10셀은 life 상관에서 제외하고 파형/QD 설명에만 포함한다.',
            '- 원본과 EDA 노트북을 변경하지 않았고 추가 모델 학습을 수행하지 않았다. 이 script가 current/ 산출물만 작성한다.','','## 산출물','',
            '- current_features.csv: 139셀 cycle10 특징·QC·life·기울기',
            '- cycle10_predischarge_profiles.csv: 방전 전 실제 sample time/current 파형',
            '- batch_stats.csv: 배치·고용량 제외 strata의 유효 n 및 셀간 분포',
            '- correlations.csv: 전체/배치별 Pearson·Spearman과 pair n',
            '- representative_cells.csv 및 cycle10_charge_current_representatives.png: 최단/최장 label 파형 비교',
            '- methodology.json: interval/적분/threshold/단위 정의']
    (out/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    out=ROOT/'results/eda/current';out.mkdir(parents=True,exist_ok=True)
    metadata=pd.read_csv(ROOT/'results/eda/knee/cell_knee_metrics.csv')
    d,profiles,raw=extract(metadata);stats=batch_statistics(d);corr=correlations(d)
    d.to_csv(out/'current_features.csv',index=False);raw.to_csv(out/'cycle10_predischarge_profiles.csv',index=False)
    stats.to_csv(out/'batch_stats.csv',index=False);corr.to_csv(out/'correlations.csv',index=False)
    rep=plot_profiles(d,profiles,out);rep.to_csv(out/'representative_cells.csv',index=False)
    findings(d,stats,corr,rep,out);(out/'methodology.json').write_text(json.dumps(METHOD,ensure_ascii=False,indent=2)+'\n')
    print(stats[stats.subset=='all'][['batch','n_cells','n_cycle10_feature_valid','n_finite_life','cycle10_mean_c_timeweighted_median','cycle10_rms_c_timeweighted_median','cycle10_active_charge_duration_min_median','cycle10_time_fraction_above_5c_median']].to_string(index=False))
    print(corr[(corr.batch=='ALL')&(corr.subset=='all_cycle10_valid')&(corr.cycle10_current_feature.isin(['cycle10_mean_c_timeweighted','cycle10_rms_c_timeweighted','cycle10_active_charge_duration_min','cycle10_time_fraction_above_5c']))][['cycle10_current_feature','target','n','pearson_r','spearman_rho']].to_string(index=False))
    print(f'Wrote cycle-10 measured-current EDA for {len(d)} cells to {out}')

if __name__=='__main__':main()
