import time

import numpy as np


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


def search(subsets, n, i, path, K=np.inf):
  if i == (n+1):
    if path:
      subsets.append(path)
    return
  if len(path) < K:
    search(subsets, n, i+1, path+[i], K)
  search(subsets, n, i+1, path, K)


def search_best_assortment(abspar, revenue, K=np.inf):
  # abspar[0] and revenue[0] are reserved for non-purchase
  # products are numbered from 1
  subsets = []
  search(subsets, len(abspar)-1, 1, [], K)
  best_rev = -1
  best_assort = []
  for subset in subsets:
    denominator = sum([abspar[prod] for prod in subset])+abspar[0]
    rev = sum([abspar[prod]/denominator*revenue[prod] for prod in subset])
    if rev > best_rev:
      best_assort = subset
      best_rev = rev
  return best_rev, best_assort
