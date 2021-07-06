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
   banditpylib.learners.mab_collaborative_ftbai_learner
   banditpylib.learners.mab_fbbai_learner
   banditpylib.learners.mab_fcbai_learner
   banditpylib.learners.mab_learner
   banditpylib.learners.mnl_bandit_learner
   banditpylib.learners.thresholding_bandit_learner

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

- :py:class:`SinglePlayerLearner`:
  Abstract class for single player learners

- :py:class:`CollaborativeLearner`:
  Abstract class for collaborative learners

- :py:class:`CollaborativeAgent`:
  Abstract class for collaborative agents

- :py:class:`CollaborativeMaster`:
  Abstract class for collaborative masters that handle arm assignment and


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

.. autoclass:: SinglePlayerLearner
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: SinglePlayerLearner
      :parts: 1

.. autoclass:: CollaborativeLearner
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: CollaborativeLearner
      :parts: 1

.. autoclass:: CollaborativeAgent
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: CollaborativeAgent
      :parts: 1

.. autoclass:: CollaborativeMaster
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: CollaborativeMaster
      :parts: 1
