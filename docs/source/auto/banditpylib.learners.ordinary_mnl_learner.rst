=============================================
``banditpylib.learners.ordinary_mnl_learner``
=============================================

.. automodule:: banditpylib.learners.ordinary_mnl_learner

   .. contents::
      :local:

.. currentmodule:: banditpylib.learners.ordinary_mnl_learner


Classes
=======

- :py:class:`OrdinaryMNLLearner`:
  Abstract class for learners playing with the ordinary mnl bandit

- :py:class:`UCB`:
  UCB policy :cite:`DBLP:journals/ior/AgrawalAGZ19`

- :py:class:`EpsGreedy`:
  Epsilon-Greedy policy

- :py:class:`ThompsonSampling`:
  Thompson sampling policy :cite:`DBLP:conf/colt/AgrawalAGZ17`


.. autoclass:: OrdinaryMNLLearner
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: OrdinaryMNLLearner
      :parts: 1

.. autoclass:: UCB
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: UCB
      :parts: 1

.. autoclass:: EpsGreedy
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: EpsGreedy
      :parts: 1

.. autoclass:: ThompsonSampling
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: ThompsonSampling
      :parts: 1
