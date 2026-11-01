"""Checkpoint class order, shared without importing the model runtime.

Class zero is the CTC blank; these characters occupy classes one onward.
Changing this order invalidates existing checkpoints.
"""

VOCAB = (
    [chr(c) for c in range(0x11200, 0x11242) if c != 0x11212]
    + [chr(c) for c in range(0xAE6, 0xAF0)]
    + [" ", ".", ",", "(", ")", "\ufffd"]
)
