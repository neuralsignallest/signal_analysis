"""Physical plant model.

Contains the compartment gas model, sorbent beds, leaks, the DC bus, the fixed-step RK4
integrator, and the application of sampled fault parameters to plant state. This is the
lowest layer: it imports nothing else from ``eio_crewbench`` except ``rng``. Every change here
ships with a conservation or balance test.
"""
