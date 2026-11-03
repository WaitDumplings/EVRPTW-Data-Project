"""Reference-only symmetric area ratio; no candidate ranking or source grouping."""
import numpy as np


def symmetric_area_ratio(left,right):
    left,right=np.broadcast_arrays(np.asarray(left,float),np.asarray(right,float))
    valid=np.isfinite(left)&np.isfinite(right)&(left>0)&(right>0)
    ratio=np.full(left.shape,np.inf)
    ratio[valid]=np.maximum(left[valid]/right[valid],right[valid]/left[valid])
    return ratio
