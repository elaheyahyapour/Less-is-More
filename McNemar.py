from math import comb
from scipy.stats import binomtest

def mcnemar_from_two_dfs(dfA, dfB, nameA, nameB):
    y = dfA["ground_truth"].tolist()
    a = (dfA["predicted"] == dfA["ground_truth"]).tolist()
    b = (dfB["predicted"] == dfB["ground_truth"]).tolist()
    b01 = sum(1 for ai, bi in zip(a, b) if ai and not bi)
    c10 = sum(1 for ai, bi in zip(a, b) if (not ai) and bi)
    n = b01 + c10
    if n == 0:
        return (nameA, nameB, 0, 0, 1.0)
    p = binomtest(k=min(b01, c10), n=n, p=0.5, alternative="two-sided").pvalue
    return (nameA, nameB, b01, c10, p)
