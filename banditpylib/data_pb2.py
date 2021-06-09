# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='data.proto',
  package='banditpylib',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\ndata.proto\x12\x0b\x62\x61nditpylib\"\x1e\n\x03\x41rm\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0b\n\x03ids\x18\x02 \x03(\x05\"<\n\x0c\x41rmPullsPair\x12\x1d\n\x03\x61rm\x18\x01 \x01(\x0b\x32\x10.banditpylib.Arm\x12\r\n\x05pulls\x18\x02 \x01(\x05\"=\n\x07\x41\x63tions\x12\x32\n\x0f\x61rm_pulls_pairs\x18\x01 \x03(\x0b\x32\x19.banditpylib.ArmPullsPair\"\\\n\x0e\x41rmRewardsPair\x12\x1d\n\x03\x61rm\x18\x01 \x01(\x0b\x32\x10.banditpylib.Arm\x12\x0f\n\x07rewards\x18\x02 \x03(\x02\x12\x1a\n\x12\x63ustomer_feedbacks\x18\x03 \x03(\x05\"B\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12\x36\n\x11\x61rm_rewards_pairs\x18\x01 \x03(\x0b\x32\x1b.banditpylib.ArmRewardsPair\"P\n\x08\x44\x61taItem\x12\x0e\n\x06rounds\x18\x01 \x01(\x05\x12\x15\n\rtotal_actions\x18\x02 \x01(\x05\x12\x0e\n\x06regret\x18\x03 \x01(\x02\x12\r\n\x05other\x18\x04 \x01(\x02\"S\n\x05Trial\x12\x0e\n\x06\x62\x61ndit\x18\x01 \x01(\t\x12\x0f\n\x07learner\x18\x02 \x01(\t\x12)\n\ndata_items\x18\x03 \x03(\x0b\x32\x15.banditpylib.DataItemb\x06proto3'
)




_ARM = _descriptor.Descriptor(
  name='Arm',
  full_name='banditpylib.Arm',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='banditpylib.Arm.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ids', full_name='banditpylib.Arm.ids', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=27,
  serialized_end=57,
)


_ARMPULLSPAIR = _descriptor.Descriptor(
  name='ArmPullsPair',
  full_name='banditpylib.ArmPullsPair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='arm', full_name='banditpylib.ArmPullsPair.arm', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pulls', full_name='banditpylib.ArmPullsPair.pulls', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=119,
)


_ACTIONS = _descriptor.Descriptor(
  name='Actions',
  full_name='banditpylib.Actions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='arm_pulls_pairs', full_name='banditpylib.Actions.arm_pulls_pairs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=121,
  serialized_end=182,
)


_ARMREWARDSPAIR = _descriptor.Descriptor(
  name='ArmRewardsPair',
  full_name='banditpylib.ArmRewardsPair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='arm', full_name='banditpylib.ArmRewardsPair.arm', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rewards', full_name='banditpylib.ArmRewardsPair.rewards', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='customer_feedbacks', full_name='banditpylib.ArmRewardsPair.customer_feedbacks', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=184,
  serialized_end=276,
)


_FEEDBACK = _descriptor.Descriptor(
  name='Feedback',
  full_name='banditpylib.Feedback',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='arm_rewards_pairs', full_name='banditpylib.Feedback.arm_rewards_pairs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=278,
  serialized_end=344,
)


_DATAITEM = _descriptor.Descriptor(
  name='DataItem',
  full_name='banditpylib.DataItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='rounds', full_name='banditpylib.DataItem.rounds', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_actions', full_name='banditpylib.DataItem.total_actions', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='regret', full_name='banditpylib.DataItem.regret', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='other', full_name='banditpylib.DataItem.other', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=346,
  serialized_end=426,
)


_TRIAL = _descriptor.Descriptor(
  name='Trial',
  full_name='banditpylib.Trial',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='bandit', full_name='banditpylib.Trial.bandit', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='learner', full_name='banditpylib.Trial.learner', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_items', full_name='banditpylib.Trial.data_items', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=428,
  serialized_end=511,
)

_ARMPULLSPAIR.fields_by_name['arm'].message_type = _ARM
_ACTIONS.fields_by_name['arm_pulls_pairs'].message_type = _ARMPULLSPAIR
_ARMREWARDSPAIR.fields_by_name['arm'].message_type = _ARM
_FEEDBACK.fields_by_name['arm_rewards_pairs'].message_type = _ARMREWARDSPAIR
_TRIAL.fields_by_name['data_items'].message_type = _DATAITEM
DESCRIPTOR.message_types_by_name['Arm'] = _ARM
DESCRIPTOR.message_types_by_name['ArmPullsPair'] = _ARMPULLSPAIR
DESCRIPTOR.message_types_by_name['Actions'] = _ACTIONS
DESCRIPTOR.message_types_by_name['ArmRewardsPair'] = _ARMREWARDSPAIR
DESCRIPTOR.message_types_by_name['Feedback'] = _FEEDBACK
DESCRIPTOR.message_types_by_name['DataItem'] = _DATAITEM
DESCRIPTOR.message_types_by_name['Trial'] = _TRIAL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Arm = _reflection.GeneratedProtocolMessageType('Arm', (_message.Message,), {
  'DESCRIPTOR' : _ARM,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:banditpylib.Arm)
  })
_sym_db.RegisterMessage(Arm)

ArmPullsPair = _reflection.GeneratedProtocolMessageType('ArmPullsPair', (_message.Message,), {
  'DESCRIPTOR' : _ARMPULLSPAIR,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:banditpylib.ArmPullsPair)
  })
_sym_db.RegisterMessage(ArmPullsPair)

Actions = _reflection.GeneratedProtocolMessageType('Actions', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONS,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:banditpylib.Actions)
  })
_sym_db.RegisterMessage(Actions)

ArmRewardsPair = _reflection.GeneratedProtocolMessageType('ArmRewardsPair', (_message.Message,), {
  'DESCRIPTOR' : _ARMREWARDSPAIR,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:banditpylib.ArmRewardsPair)
  })
_sym_db.RegisterMessage(ArmRewardsPair)

Feedback = _reflection.GeneratedProtocolMessageType('Feedback', (_message.Message,), {
  'DESCRIPTOR' : _FEEDBACK,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:banditpylib.Feedback)
  })
_sym_db.RegisterMessage(Feedback)

DataItem = _reflection.GeneratedProtocolMessageType('DataItem', (_message.Message,), {
  'DESCRIPTOR' : _DATAITEM,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:banditpylib.DataItem)
  })
_sym_db.RegisterMessage(DataItem)

Trial = _reflection.GeneratedProtocolMessageType('Trial', (_message.Message,), {
  'DESCRIPTOR' : _TRIAL,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:banditpylib.Trial)
  })
_sym_db.RegisterMessage(Trial)


# @@protoc_insertion_point(module_scope)
