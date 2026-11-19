"""Control-frame operation codes the leader announces to every rank (engine.py) and the prefix cache issues."""

OP_STOP, OP_ATTACH, OP_DETACH, OP_FORWARD, OP_BATCH = 1, 2, 3, 4, 5
OP_SNAP, OP_RESTORE, OP_DROP, OP_COPY, OP_GROW = 6, 7, 8, 9, 10
