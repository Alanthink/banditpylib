from absl.testing import absltest
from .ordinarymnlbandit import OrdinaryMNLBandit, search, search_best_assortment


class utilsTests(absltest.TestCase):

  def test_search_unrestricted(self):
    results = []
    search(results, 4, 1, [])
    self.assertEqual(results, [[1,2,3], [1, 2], [1, 3], [1], [2, 3], [2], [3]])

  def test_search_restricted(self):
    results = []
    search(results, 4, 1, [], 1)
    self.assertEqual(results, [[1], [2], [3]])

  def test_search_best_assortment(self):
    best_rev, best_assort = search_best_assortment([1,1,1,1], [0,1,1,1])
    self.assertEqual(best_assort, [1,2,3])
    self.assertEqual(best_rev, 0.75)


class banditTests(absltest.TestCase):

  def test_ordinarymnlbandit(self):
    abspar = [0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.5,0.5]
    revenue = [1,1,1,1,1,1,1,1,1,1]
    # upper bound of cardinality of every assortment
    K = 4
    bandit = OrdinaryMNLBandit(abspar, revenue, K)
    bandit.init()
    bandit.feed([1])
    bandit.feed([1])
    bandit.feed([1])
    self.assertEqual(bandit.regret(0), 2)


if __name__ == '__main__':
  absltest.main()
