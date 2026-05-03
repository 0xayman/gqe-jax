"""Representation-learning module for GQE.

This folder uses a hyphen in its name (``representation-learning``) so it is
**not** importable as a Python package. Use one of the following patterns
instead:

    # Inside any sibling module / script in this folder
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import features, dataset  # noqa

    # From the command line
    cd representation-learning
    python -m unittest discover tests

The pipeline:

    1. ``dataset.py``   — generate brickwork-Haar unitaries plus
                          Qiskit-transpiler baselines (CNOT counts and angle
                          labels in the GQE token format).
    2. ``encoder.py``   — Flax MLP encoder ``U -> z``.
    3. ``budget_model.py`` — ``z -> P(success at F>=F_thr with <= k CNOTs)``.
    4. ``angle_init.py``   — ``(U, skeleton) -> initial angles``.
    5. ``train_budget.py`` / ``train_angle.py`` — Optax training loops.
    6. ``infer.py``     — load checkpoints; suggest budget + angle init.

The module re-implements the small bits of the parent project it needs
(brickwork-Haar generator, token vocabulary) so trained checkpoints can be
loaded later without importing the live training stack.
"""
