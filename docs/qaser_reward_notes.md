# QASER Reward Notes for GQE Compilation

## Takeaway

QASER is worth borrowing, but not copying literally.

Its useful idea is:

* make circuit-efficiency rewards multiplicative rather than additive;
* let accuracy decide how much depth/CNOT improvements matter;
* avoid rewarding shallow circuits unless they are also accurate.

## What QASER Does

The paper's main reward couples:

* an accuracy term;
* a depth term;
* a gate-cost term;

using historical maxima for the structure costs. In effect, a circuit gets a strong reward only when it is **simultaneously** accurate and compact.

For compilation, the paper explicitly notes that energy can be replaced by fidelity.

## Why Not Copy It Directly

For unitary compilation, a literal energy-style ratio is awkward because:

* the ideal compilation error is `0`, so error-ratio forms become unstable;
* raw exponentials can be too aggressive for GRPO training;
* your current compiler optimizes whole circuits, not stepwise warm-start increments.

So the right move is to keep the **QASER principle** and change the exact formula.

## Recommended Objective

Use a fidelity-shaped score multiplied by depth/CNOT efficiency:

```math
\phi(F)
=
\mathrm{clip}\!\left(
-\log(1-F+\varepsilon),\,0,\,\phi_{\max}
\right)
```

```math
B(D,C)
=
\lambda_D \frac{D_{\mathrm{ref}}+1}{D+1}
+
\lambda_C \frac{C_{\mathrm{ref}}+1}{C+1},
\qquad
\lambda_D+\lambda_C=1
```

```math
\mathrm{score}
=
\phi(F)\,(\beta+\log B),
\qquad
\mathrm{cost}=-\mathrm{score}
```

## Why This Fits Better

* `-\log(1-F)` separates high-fidelity circuits much better than raw `F`.
* Depth and CNOT only help strongly when fidelity is already good.
* The objective is reference-based, not batch-relative, so it avoids the instability of z-scored penalties.
* Using `cost=-\mathrm{score}` is safer than optimizing a raw exponential directly.

## Practical Recommendations

* Use this instead of the current additive thresholded Pareto scalarization.
* Do **not** mix random weight vectors in replay if you use this objective; keep the reward stationary.
* Initialize `D_{\mathrm{ref}}` and `C_{\mathrm{ref}}` from a fidelity-only warmup, then freeze them or update them slowly.
* Set `\lambda_C > \lambda_D`, since CNOT cost is the more important hardware signal.
* Do not add total gate count yet; fidelity, depth, and CNOT are enough until variable-length generation exists.
