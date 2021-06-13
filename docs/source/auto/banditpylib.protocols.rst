=========================
``banditpylib.protocols``
=========================

.. automodule:: banditpylib.protocols

   .. contents::
      :local:

.. currentmodule:: banditpylib.protocols


Functions
=========

- :py:func:`parse_trials_data`:
  Retrieve trials data from bytes data

- :py:func:`trial_data_messages_to_dict`:
  Read file storing trials data and transform to pandas DataFrame


.. autofunction:: parse_trials_data

.. autofunction:: trial_data_messages_to_dict


Classes
=======

- :py:class:`Protocol`:
  Abstract class for a protocol which is used to coordinate the interactions

- :py:class:`SinglePlayerProtocol`:
  Single player protocol


.. autoclass:: Protocol
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Protocol
      :parts: 1

.. autoclass:: SinglePlayerProtocol
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: SinglePlayerProtocol
      :parts: 1
