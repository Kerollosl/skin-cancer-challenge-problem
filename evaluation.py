"""Binary classification metrics with explicit positive-class semantics."""
import numpy as np

def metrics(actual, probabilities, threshold=0.5):
    actual=np.asarray(actual,dtype=int);scores=np.asarray(probabilities,dtype=float)
    if actual.ndim!=1 or actual.size==0 or actual.shape!=scores.shape:
        raise ValueError("Labels and scores must be nonempty matching vectors")
    if not np.isin(actual,[0,1]).all() or not np.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all() or not 0<=threshold<=1:
        raise ValueError("Expected binary labels and finite probabilities/threshold in [0,1]")
    predicted=(scores>=threshold).astype(int)
    tp=int(((actual==1)&(predicted==1)).sum());tn=int(((actual==0)&(predicted==0)).sum())
    fp=int(((actual==0)&(predicted==1)).sum());fn=int(((actual==1)&(predicted==0)).sum())
    return {"accuracy":float((actual==predicted).mean()),"sensitivity":tp/(tp+fn) if tp+fn else None,
            "specificity":tn/(tn+fp) if tn+fp else None,"tp":tp,"tn":tn,"fp":fp,"fn":fn}
