"""Evaluation harness.

Provides the episode runner, multiprocessing across episodes, post-hoc metrics (there is no
composite scalar reward), bootstrap confidence intervals, and figure generation. Sits above
``api`` and below ``agents``.
"""
