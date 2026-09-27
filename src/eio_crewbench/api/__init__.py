"""Environment API.

Defines ``EIOEnv``, decision epochs, the action schema, and the Gymnasium wrapper. This layer
composes ``obs``, ``shield``, ``ops``, ``scenarios`` and ``plant`` into a single environment
and is the only entry point that ``eval`` and ``agents`` use.
"""
