"""Process-fidelity cost functions for quantum unitary compilation."""

from __future__ import annotations

from typing import Callable, List

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


def process_fidelity(u_target: np.ndarray, u_circuit: np.ndarray) -> float:
    d = u_target.shape[0]
    trace = np.trace(u_target.conj().T @ u_circuit)
    return float((abs(trace) ** 2) / (d**2))


def compilation_cost(gate_matrices: List[np.ndarray], u_target: np.ndarray) -> float:
    d = u_target.shape[0]
    u_circuit = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u_circuit = gate @ u_circuit
    return 1.0 - process_fidelity(u_target, u_circuit)


def build_cost_fn(u_target: np.ndarray) -> Callable[[List[np.ndarray]], float]:
    def cost_fn(gate_matrices: List[np.ndarray]) -> float:
        return compilation_cost(gate_matrices, u_target)

    return cost_fn


def process_fidelity_jax(u_target: jax.Array, u_circuit: jax.Array) -> jax.Array:
    d = u_target.shape[0]
    trace = jnp.trace(jnp.conjugate(u_target).T @ u_circuit)
    return (jnp.abs(trace) ** 2) / (d**2)


@jax.jit
def compose_unitary_batch(gate_batch: jax.Array) -> jax.Array:
    batch_size = gate_batch.shape[0]
    d = gate_batch.shape[-1]
    init = jnp.broadcast_to(jnp.eye(d, dtype=gate_batch.dtype), (batch_size, d, d))

    def step(unitary, gate):
        return gate @ unitary, None

    unitary, _ = jax.lax.scan(step, init, jnp.swapaxes(gate_batch, 0, 1))
    return unitary


@jax.jit
def compilation_cost_batch_jax(u_target: jax.Array, gate_batch: jax.Array) -> jax.Array:
    u_circuit = compose_unitary_batch(gate_batch)
    fidelity = jax.vmap(process_fidelity_jax, in_axes=(None, 0))(u_target, u_circuit)
    return 1.0 - fidelity
