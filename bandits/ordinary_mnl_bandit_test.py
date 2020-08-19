from .ordinary_mnl_bandit import OrdinaryMNLBandit, \
    search, search_best_assortment


class TestOrdinaryMNLBandit:
  """Tests in ordinary mnl bandit"""

  def test_search_unrestricted(self):
    results = []
    search(results, 3, 1, [])
    assert results == [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3]]

  def test_search_restricted(self):
    results = []
    search(results, 3, 1, [], 1)
    assert results == [[1], [2], [3]]

  def test_search_best_assortment(self):
    best_revenue, best_assortment = search_best_assortment(
        [1, 1, 1, 1], [0, 1, 1, 1])
    assert best_assortment == [1, 2, 3]
    assert best_revenue == 0.75

  def test_regret(self):
    abstraction_params = [1.0, 1.0, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.5, 0.5]
    revenues = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    card_limit = 1
    bandit = OrdinaryMNLBandit(abstraction_params, revenues, card_limit)
    bandit.reset()
    bandit.feed([([1], 3)])
    assert bandit.regret(0) == 1.5

  def test_one_product(self):
    abstraction_params = [0.0]
    revenues = [1.0]
    bandit = OrdinaryMNLBandit(abstraction_params, revenues)
    bandit.reset()
    # always get no purchase
    assert set(bandit.feed([([1], 5)])[0][1]) == {0}
