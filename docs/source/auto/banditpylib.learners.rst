========================
``banditpylib.learners``
========================

.. automodule:: banditpylib.learners

   .. contents::
      :local:


Submodules
==========

.. toctree::

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

- :py:class:`BestArmId`:
  Best arm identification

- :py:class:`MaxReward`:
  Reward maximization

- :py:class:`MaxCorrectAnswers`:
  Maximize correct answers

- :py:class:`AllCorrect`:
  Make all answers correct

- :py:class:`Learner`:
  Abstract class for learners


.. autoclass:: Goal
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Goal
      :parts: 1

.. autoclass:: BestArmId
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: BestArmId
      :parts: 1

.. autoclass:: MaxReward
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MaxReward
      :parts: 1

.. autoclass:: MaxCorrectAnswers
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: MaxCorrectAnswers
      :parts: 1

.. autoclass:: AllCorrect
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: AllCorrect
      :parts: 1

.. autoclass:: Learner
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Learner
      :parts: 1
