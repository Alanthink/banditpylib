========================
``banditpylib.learners``
========================

.. automodule:: banditpylib.learners

   .. contents::
      :local:


Submodules
==========

.. toctree::

   banditpylib.learners.linear_bandit_learner
   banditpylib.learners.ordinary_fbbai_learner
   banditpylib.learners.ordinary_fcbai_learner
   banditpylib.learners.ordinary_learner
   banditpylib.learners.ordinary_mnl_learner
   banditpylib.learners.thres_bandit_learner

.. currentmodule:: banditpylib.learners


Classes
=======

- :py:class:`Goal`:
  Abstract class for the goal of a learner

- :py:class:`IdentifyBestArm`:
  Best arm identification

- :py:class:`MaximizeTotalRewards`:
  Reward maximization

- :py:class:`MaximizeCorrectAnswers`:
  Maximize correct answers

- :py:class:`MakeAllAnswersCorrect`:
  Make all answers correct

- :py:class:`Learner`:
  Abstract class for learners


.. autoclass:: Goal
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Goal
      :parts: 1

.. autoclass:: IdentifyBestArm
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: IdentifyBestArm
      :parts: 1

.. autoclass:: MaximizeTotalRewards
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MaximizeTotalRewards
      :parts: 1

.. autoclass:: MaximizeCorrectAnswers
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MaximizeCorrectAnswers
      :parts: 1

.. autoclass:: MakeAllAnswersCorrect
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MakeAllAnswersCorrect
      :parts: 1

.. autoclass:: Learner
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Learner
      :parts: 1
