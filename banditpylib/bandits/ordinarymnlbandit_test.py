from .ordinarymnlbandit import OrdinaryMNLBandit, search, search_best_assortment


class TestOrdinaryMNLBandit:
  """tests in ordinary mnl bandit"""

  def test_search_unrestricted(self):
    results = []
    search(results, 4, 1, [])
    assert results == [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3]]

  def test_search_restricted(self):
    results = []
    search(results, 4, 1, [], 1)
    assert results == [[1], [2], [3]]

  def test_search_best_assortment(self):
    best_rev, best_assort = search_best_assortment(
        [1, 1, 1, 1], [0, 1, 1, 1])
    assert best_assort == [1, 2, 3]
    assert best_rev == 0.75

  def test_ordinarymnlbandit(self):
    pars = {
        'abspar': [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
        'revenue': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'K': 4
    }
    bandit = OrdinaryMNLBandit(pars)
    bandit.reset()
    bandit.feed([1])
    bandit.feed([1])
    bandit.feed([1])
    assert getattr(bandit, '_'+type(bandit).__name__+'__regret')(0) == 2
