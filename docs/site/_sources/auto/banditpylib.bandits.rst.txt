=======================
``banditpylib.bandits``
=======================

.. automodule:: banditpylib.bandits

   .. contents::
      :local:

.. currentmodule:: banditpylib.bandits


Functions
=========

- :py:func:`search_best_assortment`:
  Search assortment with the maximum reward

- :py:func:`local_search_best_assortment`:
  Local search assortment with the maximum reward


.. autofunction:: search_best_assortment

.. autofunction:: local_search_best_assortment


Classes
=======

- :py:class:`Bandit`:
  Abstract class for bandit environments

- :py:class:`MultiArmedBandit`:
  Multi-armed bandit

- :py:class:`LinearBandit`:
  Finite-armed linear bandit

- :py:class:`Reward`:
  General reward class

- :py:class:`MeanReward`:
  Mean reward

- :py:class:`CvarReward`:
  CVaR reward

- :py:class:`MNLBandit`:
  MNL bandit

- :py:class:`ThresholdingBandit`:
  Thresholding bandit environment

- :py:class:`ContextualBandit`:
  Finite-armed contextual bandit

- :py:class:`ContextGenerator`:
  Abstract context generator class

- :py:class:`RandomContextGenerator`:
  Random context generator


.. autoclass:: Bandit
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Bandit
      :parts: 1

.. autoclass:: MultiArmedBandit
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MultiArmedBandit
      :parts: 1

.. autoclass:: LinearBandit
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: LinearBandit
      :parts: 1

.. autoclass:: Reward
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Reward
      :parts: 1

.. autoclass:: MeanReward
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MeanReward
      :parts: 1

.. autoclass:: CvarReward
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: CvarReward
      :parts: 1

.. autoclass:: MNLBandit
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MNLBandit
      :parts: 1

.. autoclass:: ThresholdingBandit
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: ThresholdingBandit
      :parts: 1

.. autoclass:: ContextualBandit
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: ContextualBandit
      :parts: 1

.. autoclass:: ContextGenerator
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: ContextGenerator
      :parts: 1

.. autoclass:: RandomContextGenerator
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: RandomContextGenerator
      :parts: 1
