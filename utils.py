def search(subsets, n, i, path):
  if i == (n+1):
    if path:
      subsets.append(path)
    return
  search(subsets, n, i+1, path+[i])
  search(subsets, n, i+1, path)


def search_best_assortment(abspar, revenue):
  # products are numbered from 1
  subsets = []
  search(subsets, len(abspar)-1, 1, [])
  best_rev = -1
  best_assort = []
  for subset in subsets:
    denominator = sum([abspar[prod] for prod in subset])+1
    rev = sum([abspar[prod]/denominator*revenue[prod] for prod in subset])
    if rev > best_rev:
      best_assort = subset
      best_rev = rev
  return best_rev, best_assort
