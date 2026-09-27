"""Scenario definitions and sampling.

Holds scenario templates, the rejection sampler that draws concrete anomaly instances from
those templates, the wiring of per-episode seed streams, and the train/validation/evaluation
splits (split by causal structure; compound faults are evaluation-only). Sits in the middle
layer alongside ``obs``, ``shield`` and ``ops``, may import ``plant``, and must never be
imported by ``shield``.
"""
