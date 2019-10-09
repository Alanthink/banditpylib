from absl.testing import absltest

import utils


class UtilsTests(absltest.TestCase):

  def test_search(self):
    results = []
    utils.search(results, 3, 1, [])
    self.assertEqual(results, [[1,2,3], [1, 2], [1, 3], [1], [2, 3], [2], [3]])

  def test_search_best_assortment(self):
    best_rev, best_assort = utils.search_best_assortment([1,1,1,1], [0,1,1,1])
    self.assertEqual(best_assort, [1,2,3])
    self.assertEqual(best_rev, 0.75)


if __name__ == '__main__':
  absltest.main()
