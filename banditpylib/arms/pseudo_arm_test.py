from .bernoulli_arm import BernoulliArm
from .pseudo_arm import PseudoArm


class TestPseudoArm:
  """Test pseudo-arm"""
  def test_empirical_variance(self):
    pseudo_arm = PseudoArm()
    ber_arm = BernoulliArm(0.5)
    pseudo_arm.update(ber_arm.pull(100))
    em_var = pseudo_arm.em_var
    assert em_var <= 1, (
        'Empirical variance of Ber(0.5) is expected within [0.2, 0.3]. '
        'Got %.2f.' % em_var)
