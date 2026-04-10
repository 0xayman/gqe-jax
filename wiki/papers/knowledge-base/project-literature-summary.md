# Literature Summary for GQE-Based Quantum Compilation

## Project Goal

The project goal is to adapt the **Generative Quantum Eigensolver (GQE)** idea from ground-state search to **quantum compilation**:

- Input: a large unitary or a quantum circuit.
- Output: the **shortest possible native circuit** for a target device.
- Constraints: native gate set, connectivity, hardware realism, and practical scalability.

The papers in `wiki/papers` suggest that this problem sits at the intersection of:

- autoregressive sequence modeling,
- reinforcement learning for sequential circuit construction,
- energy-based learning,
- reward engineering for multi-objective optimization,
- and continuous optimization for gate parameters.

The main theme across the papers is that compilation should be treated as a **structured search problem over circuit sequences**, where the model must jointly optimize:

- correctness with respect to the target unitary/circuit,
- circuit length and depth,
- two-qubit gate cost,
- and device constraints.

## Main Takeaways

### 1. GQE provides the most relevant high-level template

Paper: `The Generative Quantum Eigensolver (GQE) and It's Application for Ground State Search.pdf`

Direct ideas from the paper:

- GQE replaces the usual VQA setup with a **generative model over circuit sequences**.
- The transformer treats a circuit like a token sequence drawn from an **operator pool**.
- The model is trained so that sampled circuits are biased toward low-cost solutions.
- The paper introduces:
  - **logit matching**, which aligns model scores with the energy landscape,
  - an adaptive **inverse-temperature** mechanism for exploration vs exploitation,
  - and a **GRPO-style loss** to amplify lower-energy samples within each batch.
- GPT-QE can be **pretrained offline** from circuit/value pairs before quantum-device training.
- The transformer can accept **conditional information**, which the paper explicitly highlights as a future/useful direction.
- Unlike some RL approaches, GPT-QE does **not require intermediate measurements at every circuit-construction step**.

Inference for this project:

- Replace the chemistry energy objective with a **compilation objective** such as:
  - unitary mismatch,
  - process fidelity error,
  - gate count,
  - circuit depth,
  - two-qubit gate count,
  - connectivity violations or routing overhead.
- Use the GQE/GPT-QE framework to generate **native-gate circuit candidates** directly.
- Condition the model on:
  - device topology,
  - native gate set,
  - optimization budget,
  - target unitary or target circuit features.
- Pretrain the model on compiler-generated data, then fine-tune with GRPO-style or energy-based objectives.

Why this matters:

- This is the clearest paper-level bridge from generative modeling to circuit search.
- It suggests a route that is less brittle than pure step-by-step RL and more flexible than fixed-ansatz optimization.

### 2. Conditional-GQE is the strongest paper-level bridge from GQE to input-conditioned generation

Paper: `Generative quantum combinatorial optimization by means of a novelconditional generative quantum eigensolver.pdf`

Direct ideas from the paper:

- The paper extends GQE to **conditional-GQE**, where the circuit generator is explicitly conditioned on an input problem instance.
- It uses an **encoder-decoder Transformer** rather than the decoder-only GPT-QE setup.
- For combinatorial optimization, the input problem is represented as a **graph**, encoded by a **graph neural network**, and passed to the decoder.
- The resulting workflow, called **Generative Quantum Combinatorial Optimization (GQCO)**, generates quantum circuits conditioned on the problem instance.
- The paper uses a **preference-based training scheme** based on **DPO** rather than labeled supervised targets.
- It emphasizes that training only needs the **final measurement/energy comparison** of sampled circuits, not a labeled dataset and not intermediate-state supervision.
- It reports nearly perfect performance on held-out optimization problems up to 10 qubits and compares favorably with SA and QAOA in that regime.
- The paper argues that conditional-GQE is not specific to combinatorial optimization and can be extended to other observable-minimization tasks.

Inference for this project:

- This is the most relevant evidence in the folder that GQE can be turned into a **conditional compiler model** rather than a per-instance search procedure.
- For compilation, the encoder side could consume:
  - the target circuit,
  - a graph representation of the target unitary/circuit structure,
  - device topology,
  - native gate information,
  - and cost-model features.
- The paper strongly supports moving from "sample circuits for one fixed Hamiltonian" to "learn a reusable map from input problem to circuit".
- Its **dataset-free preference training** is especially relevant if optimal compilation labels are unavailable or too expensive to generate at scale.

Why this matters:

- The original GQE paper gives the generative template, but this paper shows how to make it **context-aware and reusable across instances**.
- For this project, that is closer to a real compiler: a model that takes an input description and emits a circuit, instead of retraining per target.

### 3. The IBM RL synthesis paper shows that hardware-aware synthesis is already learnable

Paper: `Practical and efficient quantum circuit synthesis and transpiling with Reinforcement Learning.pdf`

Direct ideas from the paper:

- Circuit synthesis is framed as a **sequential decision process**:
  - start from the target operator,
  - apply inverse decisions until the residual operator becomes identity,
  - then invert the chosen gate sequence.
- They train RL agents with **PPO**, custom Gym environments, and **curriculum learning**.
- Reward functions can depend on:
  - whether synthesis succeeded,
  - CNOT count,
  - total gate count,
  - depth,
  - and CNOT-layer depth.
- The method produces near-optimal synthesis for:
  - linear functions up to 9 qubits,
  - Clifford circuits up to 11 qubits,
  - permutation circuits up to 65 qubits,
  - and routing improvements up to 133 qubits.
- A major practical point is that the method works **directly with native gates and connectivity restrictions**.

Inference for this project:

- Compilation into native gates is not just a theoretical objective; it is already workable as a learning problem when the environment is designed carefully.
- Reward shaping around **two-qubit gates, depth, and exact success** is central.
- A GQE-style model could use this paper as the strongest evidence that the compilation target is meaningful and practically benchmarkable.
- This paper also supports a **hybrid design**:
  - a generative model proposes circuit skeletons,
  - a local compiler/optimizer simplifies or routes them,
  - the result is fed back as training signal.

Why this matters:

- It is the strongest direct evidence in the folder that learning-based compilation can beat or complement standard heuristics in real transpilation workflows.

### 4. The gradient-descent synthesis paper is an important baseline and a warning

Paper: `Gradient descent reliably finds depth- and gate-optimal circuits for generic unitaries (2601.03123v1).pdf`

Direct ideas from the paper:

- For continuous gate sets, simple gradient descent can reliably find **depth- and gate-optimal circuits for generic unitaries**.
- The key condition is choosing a circuit skeleton that is **adequately parameterized** and avoids **parameter-deficient** patterns.
- Underparameterized circuits plateau at poor precision.
- Restricted connectivity can still work if the skeleton is chosen correctly.
- Trying to mix discrete topology changes with gradient descent did not noticeably help in their experiments.

Inference for this project:

- Not all of compilation should be handed to a transformer or RL policy.
- A strong architecture is likely:
  - **discrete search** over circuit structure with GQE/RL,
  - **continuous optimization** over gate parameters with gradient methods.
- This paper also suggests that the discrete search space should be constrained to **good skeleton families**, not arbitrary raw circuit strings.

Why this matters:

- It gives a serious non-RL baseline.
- It also suggests that the model should not search blindly over all topologies if a principled skeleton prior is available.

### 5. The RL challenge paper explains why naive end-to-end RL is hard

Paper: `Challenges for Reinforcement Learning in Quantum Circuit Design.pdf`

Direct ideas from the paper:

- Quantum circuit design is an MDP, but current RL methods face major issues:
  - sparse rewards,
  - high-dimensional action spaces,
  - exponential state representations,
  - multi-objective tradeoffs,
  - hard exploration.
- They argue that smoother intermediate rewards are needed.
- They explicitly suggest that **attention mechanisms** or partial observability may help focus on relevant parts of the state.
- In their benchmarks, SAC often performs better than PPO/A2C/TD3 in the harder continuous-control-style settings.
- Their framework is generic across state preparation, unitary composition, and related tasks.

Inference for this project:

- Pure RL over one-gate-at-a-time compilation will likely become unstable quickly as qubit count grows.
- This strengthens the case for a **sequence model trained on sampled whole circuits** rather than an agent that must reason over every intermediate quantum state.
- It also supports using:
  - reward shaping,
  - hierarchical action spaces,
  - and attention-based representations.

Why this matters:

- It clarifies the failure modes that a GQE-style compiler should try to avoid from the start.

### 6. QASER is highly relevant for reward design

Paper: `QASER: Breaking the Depth vs Accuracy Trade-off for QAS (2511.16272v1).pdf`

Direct ideas from the paper:

- Existing RL-based quantum architecture search often optimizes one metric poorly at the expense of another.
- QASER introduces a reward that jointly handles:
  - accuracy,
  - depth,
  - and entangling-gate cost.
- The paper reports shallower circuits and better accuracy than prior RL methods.
- The reward design is meant to avoid reward hacking and stabilize learning.

Inference for this project:

- Compilation is also a multi-objective problem, so the reward cannot just be "match the target unitary".
- A useful compilation reward should likely include:
  - fidelity term,
  - gate-count penalty,
  - depth penalty,
  - two-qubit gate penalty,
  - and possibly routing penalty.
- QASER strongly supports **engineered rewards** rather than hoping a simple terminal signal is enough.

Why this matters:

- This paper gives one of the clearest concrete lessons for objective design in a learning-based compiler.

## Supporting Ideas from RL, Energy-Based Learning, and LLM Papers

### PPO is the practical baseline, but not obviously the endgame

Paper: `Proximal Policy Optimization Algorithms.pdf`

Direct ideas from the paper:

- PPO uses a clipped objective to stabilize policy updates.
- It is simple, robust, and widely used.
- The IBM transpilation paper uses PPO successfully.

Inference for this project:

- PPO is a reasonable baseline for any gate-by-gate RL formulation.
- But its on-policy nature may become expensive if evaluating a circuit candidate is costly.

### GRPO looks especially well matched to sampled-circuit training

Papers:

- `DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf`
- `REVISITING GROUP RELATIVE POLICY OPTIMIZATION: INSIGHTS INTO ON-POLICY AND OFF-POLICY TRAINING.pdf`
- also used directly inside the GQE paper

Direct ideas from the papers:

- GRPO avoids learning a critic by using **group-relative normalized rewards**.
- It is well suited to settings where many candidate outputs are sampled from the same prompt/input.
- Off-policy GRPO can be competitive with or better than on-policy GRPO while reducing overhead.

Inference for this project:

- Compilation naturally fits the GRPO pattern:
  - sample several candidate circuits for the same target,
  - score them,
  - normalize rewards within the group,
  - and push up the better ones.
- This may be more natural than vanilla PPO for a transformer that generates whole circuit strings.

Why this matters:

- GQE already moves in this direction, so these papers strengthen that design choice.

### Preference-based training is a serious alternative to RL fine-tuning

Paper:

- `Generative quantum combinatorial optimization by means of a novelconditional generative quantum eigensolver.pdf`

Direct ideas from the paper:

- The paper uses **direct preference optimization (DPO)** over sampled circuits.
- For each input instance, multiple candidate circuits are sampled and compared by their final objective values.
- The model is updated to increase the probability of better circuits and decrease the probability of worse ones.

Inference for this project:

- Compilation may be a good fit for **preference-style updates**:
  - sample several compilations for the same target,
  - score them by fidelity and cost,
  - then optimize the model to prefer the better ones.
- This sits between pure supervised learning and full RL.
- It may be especially useful if compiler-quality labels are scarce but pairwise or groupwise comparisons are easy to compute.

Why this matters:

- The new paper adds a concrete third training route alongside supervised pretraining and GRPO/PPO-style RL.

### Large discrete action spaces matter for native-gate compilation

Paper: `Deep Reinforcement Learning in Large Discrete Action Space.pdf`

Direct ideas from the paper:

- Large discrete action spaces can be handled by embedding actions and searching approximately in that embedding space.
- This allows sub-linear lookup and generalization across similar actions.

Inference for this project:

- If an action is defined as `(gate type, qubit(s), parameter bucket, placement context)`, the action space can become huge.
- Learned gate/action embeddings may help both RL policies and transformer output heads scale better.

### Energy-based learning is conceptually aligned with circuit scoring

Papers:

- `A Tutorial on Energy-Based Learning.pdf`
- `Reinforcement Learning with Deep Energy-Based Policies.pdf`

Direct ideas from the papers:

- EBMs assign low energy to desirable configurations and higher energy to undesirable ones.
- They do not require normalized probabilities.
- Learning can focus on the contrastive bad samples that matter most.
- In RL, energy-based policies and maximum-entropy objectives help with exploration and multi-modality.

Inference for this project:

- A compilation model does not need to represent an exact normalized probability over all circuits.
- It may be more natural to learn a **circuit energy** or score:
  - low for short, correct, native, hardware-friendly circuits,
  - high for long, incorrect, or hardware-hostile ones.
- The fact that many different circuits can implement the same target also makes **multi-modal policies** attractive.

Why this matters:

- These papers provide the cleanest conceptual bridge between "circuit scoring" and "generative search".

### Transformer/LLM papers justify the sequence-model backbone

Papers:

- `Attention Is All You Need.pdf`
- `DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf`

Direct ideas from the papers:

- Transformers are effective for long-range structured dependencies.
- Domain-specific pretraining matters a lot.
- GRPO-style RL can improve structured reasoning after pretraining.

Inference for this project:

- Circuits are naturally token sequences with long-range dependencies:
  - an early entangling choice affects many later simplifications,
  - hardware constraints propagate across the whole sequence.
- A compiler model should probably be:
  - **conditional**, with an encoder over the target problem and device context,
  - pretrained on large synthetic/compiler-generated circuit corpora,
  - then fine-tuned with task-specific reward optimization.

## How the Papers Fit Together

A coherent project architecture suggested by the papers is:

1. **Representation**
   - Use a conditional encoder-decoder transformer to represent the target problem/device context and generate native-gate circuit sequences.
   - Use a graph encoder when the target circuit or operator has a useful graph structure.

2. **Pretraining**
   - Build a dataset of target circuits/unitaries paired with reasonable compilations.
   - Pretrain autoregressively, as in GPT-style sequence learning.

3. **Scoring**
   - Define an energy/reward based on compilation quality:
     - correctness,
     - gate count,
     - depth,
     - two-qubit cost,
     - routing overhead,
     - device legality.

4. **Policy improvement**
   - Use GRPO-style groupwise updates, DPO-style preference optimization, or another sampled-output RL method to bias generation toward better compilations.

5. **Inner-loop optimization**
   - For parameterized gates, use gradient-based refinement on top of a proposed skeleton.

6. **Post-processing**
   - Apply deterministic circuit simplification, commutation, cancellation, and routing-aware cleanup.

That is the cleanest synthesis of the folder contents.

## Most Promising Research Directions for This Project

### 1. GQE-for-compilation rather than GQE-for-energy

Use the GQE training pattern, but define the target score as a compilation score instead of molecular energy.

Possible reward:

`reward = - unitary_error - lambda_1 * gate_count - lambda_2 * depth - lambda_3 * two_qubit_count - lambda_4 * routing_cost`

This is the most direct transfer of the GQE paper.

### 2. Supervised pretraining before RL

This is supported indirectly by:

- GPT-QE pretraining,
- conditional-GQE's argument for reusable input-conditioned models,
- DeepSeekMath domain pretraining,
- and the general difficulty of sparse-reward RL in the RL challenge paper.

Practical implication:

- Start from a model that imitates existing compilers or optimal small-instance solvers.
- Then fine-tune for shorter circuits and stricter device awareness.

### 3. Groupwise or preference-based sampled training instead of pure stepwise RL

This is supported by:

- GQE,
- conditional-GQE/DPO,
- GRPO,
- and the RL challenge paper's warnings.

Practical implication:

- For each target, sample many candidate compilations.
- Rank them by reward.
- Update the model relative to the group, or by pairwise/groupwise preferences.

This looks more scalable than forcing an RL agent to observe every intermediate residual state.

### 4. Hybrid discrete-continuous optimization

This is supported by:

- GQE for discrete sequence generation,
- gradient descent paper for parameter fitting,
- practical RL paper for routing/synthesis under hardware constraints.

Practical implication:

- Let the generative model choose the circuit skeleton.
- Let a continuous optimizer tune gate parameters.
- Let a simplifier/routing pass clean up the result.

### 5. Strong reward engineering is necessary

This is supported by:

- QASER,
- practical RL synthesis,
- and the RL challenge paper.

Practical implication:

- Do not rely on an exact-success-only reward.
- Include dense intermediate or surrogate signals whenever possible.

## Important Caveats

### 1. Not all papers are directly about full unitary compilation

Some papers focus on:

- ground-state search,
- state preparation,
- architecture search,
- or generic RL/LLM methods.

Their value here is mainly **methodological transfer**, not direct evidence for full-scale compilation.

### 2. Full-unitary compilation scales brutally

A likely bottleneck is not only search, but also **evaluation**.

For large qubit counts, exact comparison of full unitaries becomes intractable. This implies the project may need:

- blockwise compilation,
- approximate fidelities,
- tensor-network surrogates,
- compiler supervision from existing tools,
- or verification on sampled inputs / observables rather than exact dense matrices.

This point is an inference from the overall literature and the known scaling of synthesis/routing methods, not a single-paper claim.

### 3. Device-conditioning is probably essential

The papers consistently suggest that compilation quality depends strongly on:

- connectivity,
- native gates,
- and the two-qubit cost model.

So a general-purpose model that ignores device information is unlikely to be competitive.

## Bottom Line

The strongest reading of the folder is:

- **GQE** is the right conceptual backbone.
- **conditional-GQE/GQCO** shows how to turn GQE into a reusable, input-conditioned generator rather than a single-instance search routine.
- **IBM's RL synthesis/transpilation paper** is the strongest direct evidence that learning-based, device-aware compilation is practical.
- **QASER** and the RL challenge paper show that reward design is central.
- **GRPO** and **DPO-style preference optimization** are natural training methods for sampled circuit candidates.
- **Gradient-based unitary synthesis** should probably be used inside the system rather than treated as a competitor to sequence generation.

If this project succeeds, it will likely look less like a pure compiler and less like a pure language model, and more like a **conditional generative search system**:

- pretrained on compilations,
- conditioned on target circuit plus device,
- improved with groupwise reward optimization or preference optimization,
- and coupled to local continuous and symbolic optimization passes.
