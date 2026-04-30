"""Centralized JAX runtime defaults for training and refinement."""

import os

_BACKEND_HINT = os.environ.get("JAX_PLATFORMS", "")
if "cpu" not in _BACKEND_HINT.lower():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    _FLAGS = os.environ.get("XLA_FLAGS", "")
    _EXTRA = "--xla_gpu_enable_latency_hiding_scheduler=true"
    if _EXTRA not in _FLAGS:
        os.environ["XLA_FLAGS"] = (_FLAGS + " " + _EXTRA).strip()

import jax  # noqa: E402

backend = jax.default_backend()
is_gpu = backend in ('gpu', 'cuda')

_CACHE_DIR = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax-compilation-cache"),
)
try:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
except Exception:
    pass

if is_gpu:
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")
    try:
        jax.config.update("jax_default_device", jax.devices("gpu")[0])
    except Exception:
        pass
else:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
