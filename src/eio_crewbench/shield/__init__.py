"""Safety shield.

Holds the safety invariants, the belief-robust forward simulation that screens proposed
actions against them, and fallback safing. May import ``plant`` to simulate forward, but
never imports ``scenarios``: the shield must not see a scenario's sampled fault.
"""
