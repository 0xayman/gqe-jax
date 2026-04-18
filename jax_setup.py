"""Shared JAX runtime configuration for the GQE codebase."""

import jax

backend = jax.default_backend()
is_gpu = backend in ('gpu', 'cuda')

if is_gpu:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")
else:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
