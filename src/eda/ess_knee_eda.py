#!/usr/bin/env python3
"""Retrospective QD degradation EDA. Read MATLAB summary references only.

Run: python -m src.pipeline eda --raw-dir ../archive
Knees are descriptive candidates, never early-life prediction features.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ess_knee_mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'ess_knee_cache'))
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_FILES = {
    'B1':'2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'B2':'2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'B3':'2018-04-12_batchdata_updated_struct_errorcorrect.mat'}
COLORS = {'B1':'#406b8c','B2':'#c67543','B3':'#52896f'}
METHOD = {'capacity_nominal_ah':1.1, 'capacity_high_qc_ah':1.65,
          'capacity_high_rule':'A heuristic QC ceiling of 1.5 x nominal, not a proved physical maximum.',
          'isolated_spike_deviation_ah':0.10, 'isolated_neighbor_tolerance_ah':0.03,
          'isolated_neighbor_count_each_side':2, 'fit_start_cycle':20,
          'median_windows':[7,15], 'primary_median_window':15,
          'rolling_min_periods':3, 'min_fit_points':160, 'min_segment_points':40,
          'min_segment_fraction':0.15, 'candidate_max_count_approx':400,
          'min_sse_improvement_fraction':0.20, 'min_slope_acceleration_ratio':1.5,
          'min_slope_difference_ah_per_cycle':1e-5,
          'near_flat_pre_slope_ah_per_cycle':1e-6, 'eol_ah':0.88,
          'window_stability_max_knee_shift_observed_span_fraction':0.05,
          'alternative_cutoffs':['first_QD_le_0.88_inclusive','start_cycle_50'],
          'slope_windows':'first and last 20% of the cycle >=20 observed span',
          'early_slope_window':'actual cycles 2..100, only if observations and available life label reach 100',
          'no_fit_to_cycle_life':'All fit cutoffs use actual observations; missing labels are never imputed.',
          'interpretation':'Retrospective, phenomenological candidate; no mechanistic or predictive claim.'}


def ols_slope(x,y):
    m=np.isfinite(x)&np.isfinite(y); x,y=np.asarray(x)[m],np.asarray(y)[m]
    return float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)) if len(x)>=3 and np.ptp(x)>0 else np.nan


def text_value(ds):
    v=ds[()]
    return ''.join(chr(int(c)) for c in v.ravel() if c) if v.dtype.kind in 'ui' else str(v.item())


def extract(root):
    curves={}; rows=[]; flags=[]
    for batch,filename in BATCH_FILES.items():
        with h5py.File(Path(os.environ['ESS_RAW_DIR']) / filename,'r') as f:
            b=f['batch']
            for idx,ref in enumerate(b['summary'][:,0]):
                key=f'{batch}c{idx}';s=f[ref]
                x=s['cycle'][()].ravel().astype(float); y=s['QDischarge'][()].ravel().astype(float)
                placeholder=np.zeros(len(y),bool)
                placeholder[0]=all(s[k][()].ravel()[0]==0 for k in ['QDischarge','QCharge','IR','Tavg','Tmax','Tmin','chargetime'])
                invalid=(~np.isfinite(x))|(~np.isfinite(y))|(y<=0)
                high=np.isfinite(y)&(y>METHOD['capacity_high_qc_ah'])&~placeholder
                isolated=np.zeros(len(y),bool)
                for j in range(2,len(y)-2):
                    around=y[[j-2,j-1,j+1,j+2]]
                    if not invalid[j] and not high[j] and np.all(np.isfinite(around)):
                        center=np.median(around)
                        isolated[j]=abs(y[j]-center)>METHOD['isolated_spike_deviation_ah'] and np.max(abs(around-center))<=METHOD['isolated_neighbor_tolerance_ah']
                valid=~(placeholder|invalid|high|isolated)
                # Preserve actual coordinates. Do not synthesize cycles or observations.
                if not np.all(np.diff(x[np.isfinite(x)])>0):
                    raise ValueError(f'{key}: non-increasing actual summary cycles require manual review')
                clean=np.where(valid,y,np.nan)
                for reason,mask in [('first_all_zero_placeholder',placeholder),('nonfinite_or_nonpositive',invalid&~placeholder),('capacity_above_1_65_qc',high),('isolated_spike',isolated)]:
                    for j in np.flatnonzero(mask):
                        flags.append({'cell_key':key,'batch':batch,'row_index':int(j),'cycle':x[j],'qd_raw_ah':y[j],'reason':reason})
                life=float(f[b['cycle_life'][idx,0]][()].ravel()[0]);life=life if np.isfinite(life) and life>0 else np.nan
                policy=text_value(f[b['policy_readable'][idx,0]])
                canonical=re.sub(r'-newstructure$','',policy)
                match=re.fullmatch(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C',canonical)
                c1,switch,c2=map(float,match.groups()) if match else (np.nan,np.nan,np.nan)
                eff=0.8/(switch/100/c1+(0.8-switch/100)/c2) if match and 0<=switch<=80 and c1>0 and c2>0 else np.nan
                end=float(np.nanmax(x));eligible=end>=100 and (np.isnan(life) or life>=100)
                early=(x>=2)&(x<=100)&valid
                early_slope=ols_slope(x[early],y[early]) if eligible else np.nan
                row={'cell_key':key,'batch':batch,'cell_index':idx,'policy':policy,'policy_canonical':canonical,
                     'policy_type':'constant_two_step' if match else 'variable','c1':c1,'switch_soc_pct':switch,'c2':c2,'c_eff_0_80':eff,
                     'cycle_life':life,'life_label_missing':np.isnan(life),'n_raw':len(y),'summary_cycle_last':end,
                     'n_valid_qd':int(valid.sum()),'n_placeholder':int(placeholder.sum()),'n_invalid':int((invalid&~placeholder).sum()),
                     'n_high_capacity':int(high.sum()),'n_isolated_spike':int(isolated.sum()),'high_capacity_cell':bool(high.any()),
                     'early_2_100_qd_slope':early_slope,'early_2_100_n':int(early.sum()) if eligible else 0,
                     'qd_endpoint_raw':y[-1],'terminal_qd_jump_gt_0_10':bool(len(y)>6 and np.isfinite(y[-1]) and abs(y[-1]-np.nanmedian(y[-6:-1]))>.10)}
                rows.append(row);curves[key]={'x':x,'raw':y,'clean':clean,'valid':valid,'high':high,'isolated':isolated}
    return pd.DataFrame(rows),curves,pd.DataFrame(flags)


def fit_hinge(x,y,window=15,start=20,cutoff=None):
    smoothing_input=np.asarray(y).copy()
    if cutoff is not None:smoothing_input[x>cutoff]=np.nan
    smoothed=pd.Series(smoothing_input).rolling(window,center=True,min_periods=3).median().to_numpy()
    mask=(x>=start)&np.isfinite(x)&np.isfinite(smoothed)&np.isfinite(y)
    if cutoff is not None:mask&=x<=cutoff
    a,b=x[mask],smoothed[mask]
    result={'n_fit':len(a),'fit_start':float(a[0]) if len(a) else np.nan,'fit_end':float(a[-1]) if len(a) else np.nan,
            'best_hinge_cycle':np.nan,'candidate_knee_cycle':np.nan,'pre_slope':np.nan,'post_slope':np.nan,
            'slope_acceleration_ratio':np.nan,'slope_difference':np.nan,'sse_improvement_fraction':np.nan,
            'linear_slope':np.nan,'early20_slope':np.nan,'late20_slope':np.nan,'late20_over_early20_ratio':np.nan,
            'early20_n':0,'late20_n':0,'status':'unresolved_insufficient_observations'}
    model=None
    if len(a)>=3 and np.ptp(a)>0:
        span=np.ptp(a);e=a<=a[0]+.2*span;t=a>=a[-1]-.2*span
        s1,s2=ols_slope(a[e],b[e]),ols_slope(a[t],b[t])
        result.update(early20_slope=s1,late20_slope=s2,early20_n=int(e.sum()),late20_n=int(t.sum()),late20_over_early20_ratio=s2/s1 if s1<-1e-6 and s2<0 else np.nan)
    if len(a)<METHOD['min_fit_points'] or np.ptp(a)<159:return result,model
    origin=a[0];span=np.ptp(a);z=(a-origin)/span;n=len(a)
    linear=np.linalg.lstsq(np.c_[np.ones(n),z],b,rcond=None)[0];sse1=float(np.sum((b-linear[0]-linear[1]*z)**2))
    minside=max(METHOD['min_segment_points'],int(np.ceil(METHOD['min_segment_fraction']*n)))
    candidate_idx=np.arange(minside,n-minside,max(1,(n-2*minside)//METHOD['candidate_max_count_approx']))
    k=z[candidate_idx]
    # Sufficient statistics for all continuous hinge fits, no NxK matrix needed.
    tail_n=n-candidate_idx
    def tail(v):return np.cumsum(v[::-1])[::-1][candidate_idx]
    tz,ty,tz2,tzy=tail(z),tail(b),tail(z*z),tail(z*b)
    hsum=tz-k*tail_n;zhsum=tz2-k*tz;h2sum=tz2-2*k*tz+k*k*tail_n;hysum=tzy-k*ty
    matrices=np.zeros((len(k),3,3));matrices[:,0,0]=n;matrices[:,0,1]=matrices[:,1,0]=z.sum();matrices[:,1,1]=(z*z).sum()
    matrices[:,0,2]=matrices[:,2,0]=hsum;matrices[:,1,2]=matrices[:,2,1]=zhsum;matrices[:,2,2]=h2sum
    rhs=np.c_[np.full(len(k),b.sum()),np.full(len(k),np.dot(z,b)),hysum]
    coefs=np.linalg.solve(matrices,rhs[...,None])[...,0]
    sses=np.maximum(0,np.dot(b,b)-np.sum(coefs*rhs,axis=1));best=int(np.argmin(sses));coef=coefs[best];knee=float(a[candidate_idx[best]])
    pre=coef[1]/span;post=(coef[1]+coef[2])/span;diff=post-pre
    ratio=post/pre if pre<-METHOD['near_flat_pre_slope_ah_per_cycle'] and post<0 else np.nan
    improvement=1-float(sses[best])/sse1 if sse1>1e-14 else 0
    accelerated=post<0 and diff<=-METHOD['min_slope_difference_ah_per_cycle'] and (ratio>=METHOD['min_slope_acceleration_ratio'] if np.isfinite(ratio) else pre>=-METHOD['near_flat_pre_slope_ah_per_cycle'])
    candidate=improvement>=METHOD['min_sse_improvement_fraction'] and accelerated
    result.update(best_hinge_cycle=knee,candidate_knee_cycle=knee if candidate else np.nan,
                  pre_slope=float(pre),post_slope=float(post),slope_acceleration_ratio=float(ratio),slope_difference=float(diff),
                  sse_improvement_fraction=float(improvement),linear_slope=float(linear[1]/span),
                  status='candidate_knee' if candidate else 'uncertain_no_criteria_match',
                  pre_segment_n=int(candidate_idx[best]),post_segment_n=int(n-candidate_idx[best]),
                  sse_linear=sse1,sse_hinge=float(sses[best]))
    model={'x':a,'smooth':b,'linear':linear[0]+linear[1]*z,'hinge':coef[0]+coef[1]*z+coef[2]*np.maximum(z-k[best],0)}
    return result,model


def analyze(features,curves):
    fitrows=[];models={}
    for key,curve in curves.items():
        x,y=curve['x'],curve['clean'];below=np.flatnonzero(np.isfinite(y)&(y<=METHOD['eol_ah']))
        eolcut=float(x[below[0]]) if len(below) else None
        primary,model=fit_hinge(x,y);models[key]=model
        for variant,window,start,cut in [('primary_w15_full',15,20,None),('w7_full',7,20,None),('w15_first_eol',15,20,eolcut),('w15_start50',15,50,None)]:
            fit,_=fit_hinge(x,y,window,start,cut)
            fitrows.append({'cell_key':key,'variant':variant,'median_window':window,'cutoff_cycle':cut,'start_cycle':start,**fit})
    fits=pd.DataFrame(fitrows);p=fits[fits.variant=='primary_w15_full'].drop(columns=['variant','median_window','cutoff_cycle','start_cycle'])
    out=features.merge(p,on='cell_key',validate='one_to_one')
    w7=fits[fits.variant=='w7_full'].set_index('cell_key')
    cutoff=fits[fits.variant=='w15_first_eol'].set_index('cell_key')
    s50=fits[fits.variant=='w15_start50'].set_index('cell_key')
    for label,tab in [('w7',w7),('eol_cut',cutoff),('start50',s50)]:
        out[label+'_candidate_knee']=out.cell_key.map(tab.candidate_knee_cycle)
        out[label+'_status']=out.cell_key.map(tab.status)
        out[label+'_knee_shift']=out[label+'_candidate_knee']-out.candidate_knee_cycle
    out['knee_observed_position']=(out.candidate_knee_cycle-out.fit_start)/(out.fit_end-out.fit_start)
    out['candidate_knee_observed_end_fraction']=out.candidate_knee_cycle/out.summary_cycle_last
    out['window_stable_candidate']=(out.status=='candidate_knee')&(out.w7_status=='candidate_knee')&(abs(out.w7_knee_shift)<=METHOD['window_stability_max_knee_shift_observed_span_fraction']*(out.fit_end-out.fit_start))
    out['cutoff_stable_candidate']=(out.status=='candidate_knee')&(out.eol_cut_status=='candidate_knee')&(abs(out.eol_cut_knee_shift)<=.05*(out.fit_end-out.fit_start))
    out['late20_accelerates_vs_early20']=(out.late20_slope<0)&((out.late20_slope-out.early20_slope)<-METHOD['min_slope_difference_ah_per_cycle'])&((out.late20_over_early20_ratio>=1.5)|(out.early20_slope>=-1e-6))
    return out,fits,models


def batch_tables(cells):
    rows=[]
    for batch in ['ALL',*BATCH_FILES]:
        bd=cells if batch=='ALL' else cells[cells.batch==batch]
        for stratum,d in [('all',bd),('no_high_capacity',bd[~bd.high_capacity_cell]),('high_capacity_flagged',bd[bd.high_capacity_cell])]:
            candidates=d[d.status=='candidate_knee'];resolved=d[d.status!='unresolved_insufficient_observations']
            rows.append({'batch':batch,'stratum':stratum,'n_cells':len(d),'n_fit_eligible':len(resolved),'n_candidate':len(candidates),
                         'candidate_fraction_cells':len(candidates)/len(d) if len(d) else np.nan,
                         'candidate_fraction_fit_eligible':len(candidates)/len(resolved) if len(resolved) else np.nan,
                         'n_window_stable_candidate':int(d.window_stable_candidate.sum()),'n_cutoff_stable_candidate':int(d.cutoff_stable_candidate.sum()),
                         'n_late20_accelerates_vs_early20':int(d.late20_accelerates_vs_early20.sum()),
                         'median_knee_cycle':candidates.candidate_knee_cycle.median(),'median_knee_observed_position':candidates.knee_observed_position.median(),
                         'median_knee_fraction_observed_end':candidates.candidate_knee_observed_end_fraction.median(),
                         'median_pre_slope':candidates.pre_slope.median(),'median_post_slope':candidates.post_slope.median(),
                         'median_hinge_slope_ratio':candidates.slope_acceleration_ratio.median(),'median_sse_improvement':candidates.sse_improvement_fraction.median(),
                         'median_early20_slope':resolved.early20_slope.median(),'median_late20_slope':resolved.late20_slope.median(),
                         'median_late20_over_early20_ratio':resolved.late20_over_early20_ratio.median(),
                         'n_late20_ratio_defined':int(resolved.late20_over_early20_ratio.notna().sum()),
                         'n_early20_near_flat_or_recovery':int((resolved.early20_slope>=-1e-6).sum()),
                         'window_shift_abs_median':candidates.w7_knee_shift.abs().median(),'window_shift_abs_max':candidates.w7_knee_shift.abs().max(),
                         'eol_cut_shift_abs_median':candidates.eol_cut_knee_shift.abs().median(),'eol_cut_shift_abs_max':candidates.eol_cut_knee_shift.abs().max(),
                         'start50_shift_abs_median':candidates.start50_knee_shift.abs().median(),'start50_shift_abs_max':candidates.start50_knee_shift.abs().max()})
    return pd.DataFrame(rows)


def policy_relations(cells):
    rows=[]
    for batch in ['ALL',*BATCH_FILES]:
        bd=cells if batch=='ALL' else cells[cells.batch==batch]
        for subset,selected in [('no_high_capacity',~bd.high_capacity_cell),('high_capacity_flagged',bd.high_capacity_cell),('no_high_capacity_labeled',(~bd.high_capacity_cell)&bd.cycle_life.notna())]:
            d=bd[selected&(bd.policy_type=='constant_two_step')]
            for policy in ['c1','c2','switch_soc_pct','c_eff_0_80']:
                for target in ['early_2_100_qd_slope','early20_slope','late20_slope','post_slope','knee_observed_position']:
                    paired=d[[policy,target]].replace([np.inf,-np.inf],np.nan).dropna();n=len(paired)
                    good=n>=4 and paired[policy].nunique()>1 and paired[target].nunique()>1
                    rho,p=spearmanr(paired[policy],paired[target]) if good else (np.nan,np.nan)
                    rows.append({'batch':batch,'subset':subset,'policy_feature':policy,'degradation_feature':target,'n':n,
                                 'spearman_rho':rho,'nominal_p_value':p,'exploratory_multiple_comparisons':True})
    return pd.DataFrame(rows)


def figure_curves(cells,curves,out):
    fig,axs=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
    for ax,batch in zip(axs.flat,BATCH_FILES):
        d=cells[cells.batch==batch]
        for r in d.itertuples():
            c=curves[r.cell_key]; color='#b02f39' if r.high_capacity_cell else COLORS[batch]
            ax.plot(c['x'],c['clean'],color=color,alpha=.22 if len(d)>2 else .9,lw=.7 if len(d)>2 else 1.5)
            if np.isfinite(r.candidate_knee_cycle) and not r.high_capacity_cell:
                near=np.nanargmin(abs(c['x']-r.candidate_knee_cycle)); ax.plot(r.candidate_knee_cycle,c['clean'][near],'o',color=COLORS[batch],ms=2,alpha=.5)
        ax.axhline(.88,color='black',ls='--',lw=.8);ax.set(title=f'{batch} | n={len(d)}; red=high-QD cell flag',xlabel='Actual summary cycle',ylabel='Discharge capacity (Ah)',ylim=(.75,1.15));ax.grid(alpha=.15)
    fig.suptitle('QC QD curves; dots are retrospective candidate knees (no-high-QD stratum)',fontsize=12)
    fig.tight_layout();fig.savefig(out/'qd_full_curves_by_batch.png',dpi=180);plt.close(fig)


def select_representatives(cells):
    selected=[]
    normal=cells[(~cells.high_capacity_cell)&cells.cycle_life.notna()]
    for batch in ['B1','B2','B3']:
        d=normal[normal.batch==batch]
        selected.extend([d.loc[d.cycle_life.idxmin(),'cell_key'],d.loc[d.cycle_life.idxmax(),'cell_key']])
    return selected


def figure_examples(cells,curves,models,out):
    selected=select_representatives(cells);fig,axs=plt.subplots(3,2,figsize=(13,11))
    for ax,key in zip(axs.flat,selected):
        row=cells.set_index('cell_key').loc[key];curve=curves[key];model=models[key]
        raw_view=curve['raw'].copy()
        if raw_view[0]==0:raw_view[0]=np.nan
        ax.plot(curve['x'],raw_view,color='#a9aaad',alpha=.55,lw=.8,label='Raw QD (zero placeholder omitted)')
        ax.plot(curve['x'],curve['clean'],color=COLORS[row.batch],alpha=.4,lw=.8,label='QC QD')
        if model:
            ax.plot(model['x'],model['smooth'],color=COLORS[row.batch],lw=1.4,label='Median 15')
            ax.plot(model['x'],model['linear'],color='#838383',ls='--',lw=1,label='Single linear fit')
            ax.plot(model['x'],model['hinge'],color='#d55838',lw=1.6,label='Continuous hinge fit')
        if np.isfinite(row.candidate_knee_cycle):ax.axvline(row.candidate_knee_cycle,color='#d55838',ls=':',lw=1)
        ax.axhline(.88,color='black',ls='--',lw=.7)
        life=f'life={row.cycle_life:.0f}' if np.isfinite(row.cycle_life) else 'life label missing'
        knee=f'knee={row.candidate_knee_cycle:.0f}' if np.isfinite(row.candidate_knee_cycle) else row.status
        ax.set(title=f'{key} | {life} | {knee}\n{row.policy}',xlabel='Actual summary cycle',ylabel='QD (Ah)')
        ax.grid(alpha=.15)
    axs.flat[0].legend(fontsize=8);fig.suptitle('Representative observed curves: shortest/longest labeled per batch',fontsize=12)
    fig.tight_layout();fig.savefig(out/'knee_representative_fits.png',dpi=170);plt.close(fig)
    return cells[cells.cell_key.isin(selected)].copy()


def figure_diagnostics(cells,out):
    normal=cells[~cells.high_capacity_cell];fig,grid=plt.subplots(2,2,figsize=(12,9));axs=grid.ravel()
    for batch,d in normal.groupby('batch'):
        ds=d.dropna(subset=['early20_slope','late20_slope']);axs[0].scatter(-ds.early20_slope*1000,-ds.late20_slope*1000,s=22,color=COLORS[batch],label=batch,alpha=.65)
        ds=d.dropna(subset=['candidate_knee_cycle','w7_candidate_knee']);axs[1].scatter(ds.candidate_knee_cycle,ds.w7_candidate_knee,s=20,color=COLORS[batch],alpha=.65)
        ds=d.dropna(subset=['candidate_knee_cycle','summary_cycle_last']);axs[2].scatter(ds.summary_cycle_last,ds.candidate_knee_cycle,s=20,color=COLORS[batch],alpha=.65)
        ds=d.dropna(subset=['candidate_knee_cycle','eol_cut_candidate_knee']);axs[3].scatter(ds.candidate_knee_cycle,ds.eol_cut_candidate_knee,s=20,color=COLORS[batch],alpha=.65)
    xlim,ylim=axs[0].get_xlim(),axs[0].get_ylim();axs[0].plot(xlim,xlim,ls='--',c='gray',lw=.8);axs[0].set_xlim(xlim);axs[0].set_ylim(ylim)
    for ax in [axs[1],axs[3]]:
        lim=max(max(ax.get_xlim()),max(ax.get_ylim()));low=min(min(ax.get_xlim()),min(ax.get_ylim()));ax.plot([low,lim],[low,lim],ls='--',c='gray',lw=.8)
    axs[0].set(xlabel='Early 20% fade rate (mAh/cycle)',ylabel='Late 20% fade rate (mAh/cycle)',title='Later > earlier indicates acceleration')
    axs[1].set(xlabel='Knee: median window 15 (cycle)',ylabel='Knee: median window 7 (cycle)',title='Window sensitivity')
    axs[2].set(xlabel='Last observed actual cycle',ylabel='Retrospective candidate knee cycle',title='Observation duration is not life label')
    axs[3].set(xlabel='Knee: full observed curve (cycle)',ylabel='Knee: first QD <= 0.88 cutoff (cycle)',title='Cutoff sensitivity')
    axs[0].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'degradation_acceleration_and_sensitivity.png',dpi=180);plt.close(fig)


def figure_policy(cells,out):
    d=cells[(~cells.high_capacity_cell)&(cells.policy_type=='constant_two_step')]
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for col,policy in enumerate(['c1','c2','switch_soc_pct']):
        for row,target in enumerate(['early_2_100_qd_slope','late20_slope']):
            ax=axs[row,col]
            for batch,bd in d.groupby('batch'):
                ax.scatter(bd[policy],-1000*bd[target],color=COLORS[batch],s=21,alpha=.6,label=batch)
            ax.set(xlabel={'c1':'C1 (C-rate)','c2':'C2 (C-rate)','switch_soc_pct':'Switch SOC (%)'}[policy],ylabel='Fade rate (mAh/cycle)',title={'early_2_100_qd_slope':'Cycle 2–100','late20_slope':'Late 20% (retrospective)'}[target]);ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=8);fig.suptitle('Charging policy vs QD fade; no-high-QD stratum, batch colors',fontsize=12)
    fig.tight_layout();fig.savefig(out/'policy_vs_early_and_late_degradation.png',dpi=180);plt.close(fig)


def write_findings(cells,batches,corr,representatives,out):
    normal=batches[(batches.stratum=='no_high_capacity')&(batches.batch!='ALL')]
    lines=['# 세 배치 QD Knee 및 열화 가속 EDA','','## 분석의 범위와 판단 기준',
    '- 원본의 실제 `summary.cycle` 및 `QDischarge`만 사용한다. cycle_life 결측은 관측길이로 대체하지 않는다. 전체 곡선 Knee·후기 기울기는 수명 종료 후 관측을 포함하는 사후 기술량이며 초기 예측 입력으로 사용할 수 없다.',
    '- 모든 값이 0인 첫 placeholder, 비유한/비양수 QD, 1.65 Ah 초과(QD nominal 1.1 Ah의 1.5배)와 고립 스파이크만 fit에서 제외한다. 1.65 Ah는 보수적 품질검사 경계이며 물리적 상한이 확정됐다는 뜻이 아니다. 어느 시점이든 이 경계를 넘은 셀은 정상 범위 셀과 별도 집계한다.',
    '- 고립 스파이크는 이전 2개·이후 2개 값이 주변 중앙값 ±0.03 Ah 안에 모이고 해당 점만 0.10 Ah 초과 이탈할 때다. 양 끝에서는 이 규칙을 적용하지 않는다. 0.88 Ah 이하인 점을 일괄 제거하지 않으며, 급락도 이 조건 없이 제거하지 않는다. 제외 값과 이유는 `quality_flags.csv`에 보존한다.',
    '- cycle 20부터 centered rolling median 15(최소 관측 3점)를 사용하고, 연속 2구간 선형 hinge `QD=a+b·cycle+c·max(cycle−k,0)`를 적합한다. 전체 160점/159 cycle 이상, 각 구간 최소 40점이면서 전체의 15% 이상이어야 한다. 약 400개 실제 cycle 후보 중 SSE 최소 시점을 선택한다.',
    '- 단일 직선 대비 SSE 20% 이상 개선, 후기 slope가 더 음수이고 차이 ≥0.00001 Ah/cycle, 전기 slope가 충분히 음수면 후기/전기 비율 ≥1.5인 경우에만 `candidate_knee`다. 전기가 거의 평탄하거나 상승하면 후기 음수 전환을 별도로 허용한다. 조건 불일치는 일정속도 확정이 아니라 불확실, 부족한 관측은 unresolved다.',
    '- Knee는 급격한 열화 전환을 근사한 회고적·현상론적 후보다. 매끈한 곡률도 hinge 개선을 만들 수 있으므로 특정 물리기전·통계적 change point 확정과 동일하지 않다. SSE는 적합도이며 cycle 간 자기상관을 무시한 유의확률이 아니다.',
    '- 시작 cycle 20 뒤 관측 span의 처음/마지막 20% 선형 기울기를 독립 보완지표로 계산한다. 초기 2–100 slope는 관측과 유효 label(있는 경우)이 100에 도달할 때만 계산한다. 같은 셀의 여러 cycle은 독립 표본으로 취급하지 않는다.','','## 배치별 정상 용량 범위 셀 결과',
    '| 배치 | 정상 범위 셀 / 전체 | 적합 가능 | Knee 후보 | 후보 median cycle | median 관측위치 | 후기/전기 20% 비율 median (n) | 20% 구간 가속 셀 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in normal.itertuples():
        total=int((cells.batch==r.batch).sum())
        lines.append(f'| {r.batch} | {r.n_cells}/{total} | {r.n_fit_eligible} | {r.n_candidate} | {r.median_knee_cycle:.0f} | {r.median_knee_observed_position:.1%} | {r.median_late20_over_early20_ratio:.2f} ({r.n_late20_ratio_defined}) | {r.n_late20_accelerates_vs_early20} |')
    lines+=['','배치 정의는 B1=2017-05-12, B2=2018-02-20, B3=2018-04-12이며 파일 3개만 명시적으로 읽는다.','후기/전기 slope 비율은 두 구간 모두 감소하고 전기 slope <−0.000001 Ah/cycle일 때만 정의한다. B2처럼 처음에 capacity가 상승/평탄한 셀은 비율에서 빠지므로 비율 median의 유효 n을 함께 읽어야 한다.','관측위치는 (knee−적합 첫cycle)/(적합 마지막cycle−적합 첫cycle)다. 수명 label 대비 비율이 아니다. B1/B2/B3의 유한 life label 수는 각각 46/39/44다. 결측 label은 관측길이로 대체하지 않았다.','', '## 고용량 및 이상점 분리']
    flagged=cells[cells.high_capacity_cell]
    lines.append(f'- 고용량 경험 셀은 {len(flagged)}개({", ".join(flagged.cell_key)})이고 총 고용량 제외점은 {int(cells.n_high_capacity.sum())}개다. 그 밖의 고립 spike 제외점은 {int(cells.n_isolated_spike.sum())}개다.')
    for r in batches[(batches.stratum=='high_capacity_flagged')&(batches.n_cells>0)&(batches.batch!='ALL')].itertuples():
        lines.append(f'- {r.batch} 고용량 strata: {r.n_cells}셀, 적합가능 {r.n_fit_eligible}, 후보 {r.n_candidate}; 이 값을 정상 strata 결론에 혼합하지 않는다.')
    lines+=['','## Window와 cutoff 민감도',
    '- median 7 vs 15 window에서 양쪽 모두 후보이고 Knee 위치 차이가 관측 span의 5% 이하이면 window-stable로 표시한다. 최초 QD≤0.88까지 포함하는 대안 cutoff와 fit 시작cycle 50도 별도 비교했다. 최초 EOL 통과가 없으면 full과 동일하다. 이 민감도는 missing life가 완성됐음을 의미하지 않는다.']
    for r in normal.itertuples():
        lines.append(f'- {r.batch}: window 안정 후보 {r.n_window_stable_candidate}/{r.n_candidate}, |Δknee| median/max={r.window_shift_abs_median:.0f}/{r.window_shift_abs_max:.0f} cycle; EOL-cut 안정 {r.n_cutoff_stable_candidate}/{r.n_candidate}, |Δ| median/max={r.eol_cut_shift_abs_median:.0f}/{r.eol_cut_shift_abs_max:.0f}; start50 |Δ| median/max={r.start50_shift_abs_median:.0f}/{r.start50_shift_abs_max:.0f}.')
    lines+=['','## 대표 그래프 해석 근거',
    '대표 그래프는 B1/B2/B3에서 고용량 셀을 제외한 유한 life label의 최단·최장 셀이다. 붉은 hinge와 점선 단일 fit의 오차, 전·후 기울기, EOL 선, raw vs smoothed를 함께 확인할 수 있다.']
    for r in representatives.sort_values(['batch','cycle_life']).itertuples():
        life=f'{r.cycle_life:.0f}' if np.isfinite(r.cycle_life) else 'NaN'
        lines.append(f'- {r.cell_key}: label life={life}, 관측끝={r.summary_cycle_last:.0f}, {r.status}, 후보={r.candidate_knee_cycle:.0f}, 전/후 slope={r.pre_slope*1000:.3f}/{r.post_slope*1000:.3f} mAh/cycle, 직선 SSE 대비 개선={r.sse_improvement_fraction:.1%}.')
    lines+=['','## 충전 정책과 열화 속도',
    '- `policy_degradation_correlations.csv`는 고용량 제외/고용량 분리/고용량 제외+유한label strata, pooled 및 배치내 Spearman 상관을 유효 셀 n과 함께 보존한다. 후기 slope와 Knee는 label이 없어도 관측 기술량으로 계산할 수 있다. 파일 내 가변 정책/slow-cycle 진단 셀은 고정 2단계 정책 숫자를 임의 부여하지 않는다.',
    '- 아래는 정상 범위 셀의 pooled 주요 비교다. 음의 slope와 policy의 음의 상관은 더 빠른 capacity 감소와 연결된다. 여러 비교의 p는 보정하지 않은 탐색적 값이며, 배치·프로토콜·SOC 전환 조건의 교란이 있어 원인효과로 해석하지 않는다.',
    '| 정책 | 초기 2–100 slope: rho (n) | 후기 20% slope: rho (n) |', '|---|---:|---:|']
    for p in ['c1','c2','switch_soc_pct','c_eff_0_80']:
        vals=[]
        for target in ['early_2_100_qd_slope','late20_slope']:
            r=corr[(corr.batch=='ALL')&(corr.subset=='no_high_capacity')&(corr.policy_feature==p)&(corr.degradation_feature==target)].iloc[0]
            vals.append(f'{r.spearman_rho:.3f} ({int(r.n)})')
        lines.append(f'| {p} | {vals[0]} | {vals[1]} |')
    lines+=['','배치내 주요 후기20% slope 상관:', '| 배치 | C1 rho (n) | C2 rho (n) | switch rho (n) |','|---|---:|---:|---:|']
    for batch in ['B1','B2','B3']:
        vals=[]
        for p in ['c1','c2','switch_soc_pct']:
            r=corr[(corr.batch==batch)&(corr.subset=='no_high_capacity')&(corr.policy_feature==p)&(corr.degradation_feature=='late20_slope')].iloc[0]
            vals.append(f'{r.spearman_rho:.3f} ({int(r.n)})')
        lines.append(f'| {batch} | '+ ' | '.join(vals)+' |')
    lines+=['','- 후기 속도와 C1의 pooled 상관은 약한데 B1/B3 내에서는 더 음수인 연결이 나타난다. 배치별 capacity baseline·관측길이·정책 분포 차이가 혼합되므로 pooled 값만으로 정책 영향이 없다고 결론내릴 수 없다. C2의 방향도 C1과 다르게 나타나며, 두 단계의 current와 switch SOC를 동시에 설계한 실험이라는 점을 고려해야 한다.',
    '- B1c0는 최적 hinge의 SSE 개선이 13.9%, slope 비율도 기준보다 작아 후보로 분류되지 않는다. B1c4는 관측 마지막 QD가 EOL보다 높고 완만한 가속 후보가 나타난다. 따라서 모든 후보를 EOL 직전의 급격한 붕괴라고 동일시하지 않는다. B2c19/B2c34는 후기 붉은 hinge가 뚜렷하게 더 가파르고, B3c28은 처음부터 감소가 상당한 셀이라 slope 비율이 다른 대표보다 작다.',
    '','## 해석 한계',
    '- 하나의 hinge는 초기 conditioning·중기 plateau·다단계 열화 등 복잡한 곡선을 두 구간으로 축약한다. Knee 위치는 관측 종료길이와 분석 window/cutoff에 의존한다. 최소 전후길이 조건 때문에 극초기/종료 직전 전환은 찾기 어렵다.',
    '- 일부 terminal 급락은 고립 spike 조건을 충족하지 않아 보존된다. 실제 최후 열화와 기록 이상을 summary만으로 구별할 수 없다. `terminal_qd_jump_gt_0_10` 표시 셀과 start/EOL-cut 민감도를 확인해야 한다.',
    '- 관측 중단 셀의 끝은 수명 종료와 다르다. 실험 label·관측길이·EOL 값은 독립적으로 보존하며, label이 없는 셀(가변충전/진단 셀 포함)의 결과를 수명 상관에 사용하지 않는다.',
    '- 다중 비교, policy 반복/배치 교란, smoothing에 따른 오차 상관, label endpoint 이상으로 인해 상관은 설명적 연결이다. 실제 충전 전류 파형이나 물리기전을 확인하려면 cycle 수준의 전류/전압 추가 검증이 필요하다.',
    '- 원본 파일/노트북을 변경하지 않았고 모델 학습을 하지 않았다. 모든 CSV와 PNG는 이 script로 재현한다.','','## 산출물',
    '- `cell_knee_metrics.csv`: 셀별 QC·label·정책·기울기·후보·민감도',
    '- `knee_fit_variants.csv`: 모든 셀의 4개 fit variant',
    '- `batch_knee_statistics.csv`: 전체/정상 범위/고용량 strata별 정량 요약',
    '- `quality_flags.csv`: 제외점의 원시값과 사유',
    '- `policy_degradation_correlations.csv`: 유효 n 포함 탐색적 Spearman',
    '- `representative_cells.csv`: 대표 선정과 정량 근거',
    '- `verification_audit.json`: 직접 least-squares 6셀 및 합성 Knee/직선/단기관측/cutoff 검증',
    '- `qd_full_curves_by_batch.png`, `knee_representative_fits.png`, `degradation_acceleration_and_sensitivity.png`, `policy_vs_early_and_late_degradation.png`']
    report='\n'.join(lines)+'\n'
    report=re.sub(r'(^##[^\n]*\n)(?!\n)',r'\1\n',report,flags=re.MULTILINE)
    formatted=[]
    for line in report.splitlines():
        if line.startswith('|') and formatted and formatted[-1] and not formatted[-1].startswith('|'):formatted.append('')
        formatted.append(line)
    (out/'findings.md').write_text('\n'.join(formatted)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args()
    out=args.root/'results/eda/knee';out.mkdir(parents=True,exist_ok=True)
    features,curves,flags=extract(args.root);cells,fits,models=analyze(features,curves);batches=batch_tables(cells);corr=policy_relations(cells)
    cells.to_csv(out/'cell_knee_metrics.csv',index=False);fits.to_csv(out/'knee_fit_variants.csv',index=False);batches.to_csv(out/'batch_knee_statistics.csv',index=False)
    flags.to_csv(out/'quality_flags.csv',index=False);corr.to_csv(out/'policy_degradation_correlations.csv',index=False)
    figure_curves(cells,curves,out);rep=figure_examples(cells,curves,models,out);rep.to_csv(out/'representative_cells.csv',index=False)
    figure_diagnostics(cells,out);figure_policy(cells,out);write_findings(cells,batches,corr,rep,out)
    (out/'methodology.json').write_text(json.dumps(METHOD,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(batches[(batches.stratum=='no_high_capacity')&(batches.batch!='ALL')].to_string(index=False))
    print(f'Wrote {len(cells)} cell metrics and {len(fits)} fit variants to {out}')

if __name__=='__main__':main()
