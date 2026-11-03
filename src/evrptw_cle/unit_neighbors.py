"""Source-reference RMS metric with equal expected group contributions.

Geographic neighbors in the original training reference establish the scales of spatial,
log-area and independent log-height/log-level differences.  x/y form one spatial
group.  Fallback scales and constant-feature support are explicitly recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MetricConfig:
    k: int = 8
    scale_probe_count: int = 2000
    seed: int = 20261102
    min_reference_count: int = 1
    chunk_size: int = 8192


def numeric(frame, name):
    if name not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def use_types(frame):
    value=frame.get("use_type",pd.Series("generic",index=frame.index)).fillna("generic").astype(str).to_numpy()
    return np.where(np.isin(value,["house","apartments"]),value,"generic")


def weighted_quantile(values, weights, q):
    order=np.argsort(values,axis=-1,kind="stable")
    v=np.take_along_axis(values,order,axis=-1)
    w=np.take_along_axis(weights,order,axis=-1)
    sums=np.cumsum(w,axis=-1)
    pos=np.argmax(sums>=q*sums[...,-1:],axis=-1)
    return np.take_along_axis(v,pos[...,None],axis=-1)[...,0]


def identities(frame):
    ids=(frame.entity_id.astype(str).to_numpy() if "entity_id" in frame else np.asarray([f"row:{i}" for i in frame.index],object))
    group=(frame.source_group.fillna("").astype(str).to_numpy() if "source_group" in frame else np.full(len(frame),"",object))
    group=np.where(group!="",group,"entity:"+ids)
    return ids,group


def query_independent(tree, features, k, reference_ids, reference_groups, query_ids, query_groups):
    """Nearest candidates excluding the same entity or genuine source group."""
    nref=len(reference_ids);rows=len(features)
    if not rows:return np.empty((0,k)),np.empty((0,k),dtype=int),np.zeros(0,dtype=int)
    ask=min(nref,max(k+1,k))
    while True:
        distance,index=tree.query(features,k=ask,workers=1)
        if ask==1:distance,index=distance[:,None],index[:,None]
        valid=(reference_ids[index]!=query_ids[:,None])&(reference_groups[index]!=query_groups[:,None])
        enough=(valid.sum(axis=1)>=min(k,nref))
        if enough.all() or ask==nref:break
        ask=min(nref,max(ask+1,ask*2))
    order=np.argsort(~valid,axis=1,kind="stable")[:,:min(k,ask)]
    idx=np.take_along_axis(index,order,axis=1)
    dist=np.take_along_axis(distance,order,axis=1)
    ok=np.take_along_axis(valid,order,axis=1)
    dist=np.where(ok,dist,np.inf)
    if dist.shape[1]<k:
        pad=k-dist.shape[1]
        dist=np.pad(dist,((0,0),(0,pad)),constant_values=np.inf)
        idx=np.pad(idx,((0,0),(0,pad)),constant_values=0)
        ok=np.pad(ok,((0,0),(0,pad)),constant_values=False)
    return dist,idx,ok.sum(axis=1)


class EmpiricalMetricNeighborModel:
    """Fit one city/reference pool and reuse its data-derived metric.

    Accepted config keys are k, scale_probe_count, seed, min_reference_count and
    chunk_size.  Predictors are x/y (metres), area_m2, use_type and original
    independent height_m/levels.  Reference labels are original positive units.
    No NSI occupancy band, NSI-imputed floor count, or estimated unit is read.
    """
    def __init__(self,config=None):
        self.config=MetricConfig(**(config or {}))
        self._pools={};self._scale_rows=[];self._height_pools={}

    def _features(self,arrays,positions,pool):
        scales=pool["scales"];columns=[]
        if scales["spatial"]>0:
            columns += [arrays["x"][positions]/scales["spatial"],arrays["y"][positions]/scales["spatial"]]
        if scales["area"]>0:columns.append(np.log(arrays["area_m2"][positions])/scales["area"])
        if pool["branch"]!="area" and scales["vertical"]>0:
            columns.append(np.log(arrays[pool["branch"]][positions])/scales["vertical"])
        if not columns:return np.zeros((len(positions),1))
        return np.column_stack(columns)/np.sqrt(pool["active_groups"])

    def fit(self,reference_df):
        self._pools={};self._scale_rows=[];self._height_pools={}
        arrays={key:numeric(reference_df,key) for key in ["x","y","area_m2","height_m","levels","units"]}
        valid=(np.isfinite(arrays["x"])&np.isfinite(arrays["y"])&np.isfinite(arrays["area_m2"])&
               (arrays["area_m2"]>0)&np.isfinite(arrays["units"])&(arrays["units"]>=1))
        types=use_types(reference_df);ids,groups=identities(reference_df)
        city=str(reference_df.city.iloc[0]) if "city" in reference_df and len(reference_df) else ""
        for kind in ["house","apartments","generic"]:
            for branch in ["area","levels","height_m"]:
                mask=valid&(types==kind)
                if branch!="area":mask &= np.isfinite(arrays[branch])&(arrays[branch]>0)
                positions=np.flatnonzero(mask)
                if len(positions)<self.config.min_reference_count:continue
                xy=np.column_stack([arrays["x"][positions],arrays["y"][positions]])
                area=np.log(arrays["area_m2"][positions])
                vertical=np.log(arrays[branch][positions]) if branch!="area" else None
                seed=self.config.seed+int.from_bytes(hashlib.blake2b(f"{kind}:{branch}".encode(),digest_size=4).digest(),"little")
                rng=np.random.default_rng(seed)
                probes=np.sort(rng.choice(len(positions),size=min(self.config.scale_probe_count,len(positions)),replace=False))
                tree=cKDTree(xy)
                distances,neighbors,counts=query_independent(tree,xy[probes],self.config.k,
                    ids[positions],groups[positions],ids[positions[probes]],groups[positions[probes]])
                pair_ok=np.isfinite(distances)
                r2=np.square(distances[pair_ok])
                a2=np.square((area[neighbors]-area[probes,None])[pair_ok])
                z2=(np.square((vertical[neighbors]-vertical[probes,None])[pair_ok]) if vertical is not None else np.zeros_like(a2))
                moments={"spatial":float(r2.mean()) if len(r2) else 0.,
                         "area":float(a2.mean()) if len(a2) else 0.,
                         "vertical":float(z2.mean()) if len(z2) else 0.}
                global_moments={"spatial":float(2*np.var(xy,axis=0).sum()),
                                "area":float(2*np.var(area)),
                                "vertical":float(2*np.var(vertical)) if vertical is not None else 0.}
                # Exact equality distinguishes a genuinely constant observed
                # feature from a local pair sample that happened not to vary.
                # A floating-point variance can otherwise be tiny but positive
                # even for an array containing one identical value.
                globally_constant={"spatial":bool(np.all(xy==xy[0])),
                                     "area":bool(np.all(area==area[0])),
                                     "vertical":vertical is None or bool(np.all(vertical==vertical[0]))}
                scales={};schemes={};constants={}
                for name in ["spatial","area","vertical"]:
                    if name=="vertical" and branch=="area":
                        scales[name]=0.;schemes[name]="absent_branch";constants[name]=True
                    elif globally_constant[name]:
                        scales[name]=0.;schemes[name]="constant_feature";constants[name]=True
                    elif moments[name]>0:
                        scales[name]=float(np.sqrt(moments[name]));schemes[name]="geographic_neighbor_pairs";constants[name]=False
                    elif global_moments[name]>0:
                        scales[name]=float(np.sqrt(global_moments[name]));schemes[name]="pooled_random_pair_rms";constants[name]=False
                    else:
                        scales[name]=0.;schemes[name]="constant_feature";constants[name]=True
                active=sum(scales[n]>0 for n in scales)
                pool={"positions":positions,"scales":scales,"active_groups":active,"branch":branch,
                      "xy":xy,"log_units":np.log(arrays["units"][positions]),"area":arrays["area_m2"][positions],
                      "identifiers":ids[positions],"source_groups":groups[positions],
                      "constant_x":xy[0,0],"constant_y":xy[0,1],"constant_area":area[0],
                      "constant_vertical":vertical[0] if vertical is not None else None}
                pool["tree"]=cKDTree(self._features(arrays,positions,pool))
                self._pools[kind,branch]=pool
                old={"spatial":moments["spatial"]/5000**2,"area":4*moments["area"],"vertical":moments["vertical"] if branch!="area" else 0.}
                oldsum=sum(old.values())
                row={"city":city,"use_type":kind,"branch":branch,"reference_count":len(positions),
                     "scale_probe_count":len(probes),"geographic_pair_count":len(r2),"active_groups":active,
                     "r_rms_m":scales["spatial"],"log_area_rms":scales["area"],"log_vertical_rms":scales["vertical"],
                     "calibration_pair_scheme":";".join(f"{n}={schemes[n]}" for n in schemes),
                     "has_pooled_fallback":any(v=="pooled_random_pair_rms" for v in schemes.values())}
                for name in scales:
                    row[f"{name}_constant"]=constants[name]
                    row[f"{name}_scale_scheme"]=schemes[name]
                    row[f"old_expected_contribution_{name}"]=old[name]
                    row[f"old_expected_share_{name}"]=old[name]/oldsum if oldsum else 0.
                    row[f"new_expected_contribution_{name}"]=1/active if scales[name]>0 else 0.
                    row[f"new_geo_pair_contribution_{name}"]=moments[name]/scales[name]**2/active if scales[name]>0 else 0.
                self._scale_rows.append(row)
        # This separate pool supports a loose upper-cap height estimate only.
        # Its original heights never alter the feature vectors or unit pools.
        for kind in ["house","apartments","generic"]:
            area_pool=self._pools.get((kind,"area"))
            if area_pool is None:continue
            positions=area_pool["positions"]
            positions=positions[np.isfinite(arrays["height_m"][positions])&(arrays["height_m"][positions]>0)]
            if not len(positions):continue
            self._height_pools[kind]={"tree":cKDTree(self._features(arrays,positions,area_pool)),
                "heights":arrays["height_m"][positions].copy(),"identifiers":ids[positions].copy(),
                "source_groups":groups[positions].copy(),"area_pool":area_pool}
        self.reference_count_=int(valid.sum())
        return self

    def scales_table(self):return pd.DataFrame(self._scale_rows)

    def predict(self,target_df,estimate_missing_height=False):
        n=len(target_df)
        names=["neighbor_units","neighbor_log_units","neighbor_cost","neighbor_log_mad","neighbor_log_spread_90",
               "neighbor_log_q10","neighbor_log_q90","neighbor_count","neighbor_reference_count","neighbor_units_min",
               "neighbor_units_max","neighbor_median_distance_m","neighbor_nearest_distance_m","neighbor_height_p90"]
        out={name:np.full(n,np.nan) for name in names}
        out["neighbor_valid"]=np.zeros(n,bool);out["neighbor_branch"]=np.full(n,"unavailable",object)
        out["neighbor_reference_id"]=np.full(n,"",object);out["metric_constant_out_of_support"]=np.zeros(n,bool)
        out["neighbor_height_estimated"]=np.zeros(n,bool)
        arrays={key:numeric(target_df,key) for key in ["x","y","area_m2","height_m","levels"]}
        valid=np.isfinite(arrays["x"])&np.isfinite(arrays["y"])&np.isfinite(arrays["area_m2"])&(arrays["area_m2"]>0)
        types=use_types(target_df);ids,groups=identities(target_df)
        for kind in ["house","apartments","generic"]:
            branches=np.full(n,"area",object)
            if (kind,"height_m") in self._pools:
                branches[np.isfinite(arrays["height_m"])&(arrays["height_m"]>0)]="height_m"
            if (kind,"levels") in self._pools:
                branches[np.isfinite(arrays["levels"])&(arrays["levels"]>0)]="levels"
            for branch in ["area","height_m","levels"]:
                pool=self._pools.get((kind,branch))
                if pool is None:continue
                take=valid&(types==kind)&(branches==branch)
                compatible=np.ones(n,bool);s=pool["scales"]
                if s["spatial"]==0:compatible &= (arrays["x"]==pool["constant_x"])&(arrays["y"]==pool["constant_y"])
                if s["area"]==0:compatible &= np.log(arrays["area_m2"])==pool["constant_area"]
                if branch!="area" and s["vertical"]==0:compatible &= np.log(arrays[branch])==pool["constant_vertical"]
                out["metric_constant_out_of_support"] |= take&~compatible
                selected=np.flatnonzero(take&compatible)
                for start in range(0,len(selected),self.config.chunk_size):
                    ix=selected[start:start+self.config.chunk_size]
                    features=self._features(arrays,ix,pool)
                    distance,neighbor,count=query_independent(pool["tree"],features,self.config.k,
                        pool["identifiers"],pool["source_groups"],ids[ix],groups[ix])
                    usable=count>0
                    if not usable.any():continue
                    ix=ix[usable];distance=distance[usable];neighbor=neighbor[usable];count=count[usable]
                    weights=np.where(np.isfinite(distance),1/(1+distance)**2,0.)
                    logs=pool["log_units"][neighbor]
                    med=weighted_quantile(logs,weights,.5);q10=weighted_quantile(logs,weights,.1);q90=weighted_quantile(logs,weights,.9)
                    spatial=np.linalg.norm(pool["xy"][neighbor]-np.column_stack([arrays["x"][ix],arrays["y"][ix]])[:,None,:],axis=-1)
                    out["neighbor_units"][ix]=np.maximum(1,np.floor(np.exp(med)+.5))
                    out["neighbor_log_units"][ix]=med
                    out["neighbor_cost"][ix]=np.sum(np.where(np.isfinite(distance),distance,0.)*weights,axis=1)/weights.sum(axis=1)
                    out["neighbor_log_mad"][ix]=weighted_quantile(np.abs(logs-med[:,None]),weights,.5)
                    out["neighbor_log_spread_90"][ix]=np.maximum(0,q90-q10)
                    out["neighbor_log_q10"][ix]=q10;out["neighbor_log_q90"][ix]=q90
                    out["neighbor_count"][ix]=count;out["neighbor_reference_count"][ix]=len(pool["positions"])
                    out["neighbor_units_min"][ix]=np.exp(np.where(weights>0,logs,np.inf).min(axis=1))
                    out["neighbor_units_max"][ix]=np.exp(np.where(weights>0,logs,-np.inf).max(axis=1))
                    out["neighbor_median_distance_m"][ix]=weighted_quantile(spatial,weights,.5)
                    out["neighbor_nearest_distance_m"][ix]=np.where(weights>0,spatial,np.inf).min(axis=1)
                    out["neighbor_reference_id"][ix]=pool["identifiers"][neighbor[:,0]]
                    out["neighbor_branch"][ix]=branch;out["neighbor_valid"][ix]=True
        if estimate_missing_height:
            missing=(~(np.isfinite(arrays["height_m"])&(arrays["height_m"]>0))&
                     ~(np.isfinite(arrays["levels"])&(arrays["levels"]>0)))
            heights=self._height_p90(arrays,types,ids,groups,valid&missing)
            out["neighbor_height_p90"]=heights
            out["neighbor_height_estimated"]=np.isfinite(heights)
        return pd.DataFrame(out,index=target_df.index)

    def _height_p90(self,arrays,types,ids,groups,missing):
        heights=np.full(len(types),np.nan)
        for kind,reference in self._height_pools.items():
            pool=reference["area_pool"];scales=pool["scales"]
            compatible=np.ones(len(types),bool)
            if scales["spatial"]==0:
                compatible &= (arrays["x"]==pool["constant_x"])&(arrays["y"]==pool["constant_y"])
            if scales["area"]==0:compatible &= np.log(arrays["area_m2"])==pool["constant_area"]
            selected=np.flatnonzero(missing&(types==kind)&compatible)
            for start in range(0,len(selected),self.config.chunk_size):
                ix=selected[start:start+self.config.chunk_size]
                distance,neighbor,count=query_independent(reference["tree"],self._features(arrays,ix,pool),self.config.k,
                    reference["identifiers"],reference["source_groups"],ids[ix],groups[ix])
                usable=count>0
                if not usable.any():continue
                ix=ix[usable];distance=distance[usable];neighbor=neighbor[usable]
                weights=np.where(np.isfinite(distance),1/(1+distance)**2,0.)
                heights[ix]=weighted_quantile(reference["heights"][neighbor],weights,.9)
        return heights
