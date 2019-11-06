import time

import numpy as np


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


def search(subsets, n, i, path, K=np.inf):
  if i == n:
    if path:
      subsets.append(path)
    return
  if len(path) < K:
    search(subsets, n, i+1, path+[i], K)
  search(subsets, n, i+1, path, K)


def search_best_assortment(abspar, revenue, K=np.inf):
  # non-purchase is assumed to have abstraction par 1
  subsets = []
  search(subsets, len(abspar), 1, [], K)
  sorted_assort = sorted( [ (sum([abspar[prod]/
      (sum([abspar[prod] for prod in subset])+1)*revenue[prod]
      for prod in subset]), subset) for subset in subsets], key=lambda x:x[0] )
  return sorted_assort[-1]
