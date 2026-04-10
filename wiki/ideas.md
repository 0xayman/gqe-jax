Razvan suggestions
- Try to do a supervised (imitation learning) pre-training first before the RL step
- Try reward shaping (if a gate cancels the previous one then give it negative reward), or other types of reward shaping (should explore a bit on this)
- Softmax temperature (I don't know what it is now but I should look into it)
- Something about evolution search 
- What if we can penalize circuits that have less than (alpha) entaglement gates? probably there is a heuristic for the number of CNOTS in a circuit based on some properties of it
- Razvan also mentioned X-LSTM as option for the model instead of transformer



=============================================


  ---
  Direction 1: Condition the model on the target unitary

  Right now the transformer is unconditional — it has no idea what target it's compiling for. If you encode u_target as a context vector and feed it as a prefix to the transformer
  (cross-attention or a learned embedding), the model could in principle learn "for this target, generate this structure." Inference becomes one forward pass over the sequence.

  You still need angles, but you could also predict them as additional output tokens interleaved with gate type tokens: [RZ, q0, θ=1.23, CNOT, q0, q1, ...]. This turns it into a
  conditional sequence model.

  ---
  Direction 2: Replace L-BFGS with a second network

  Train a separate "angle predictor" network that takes a gate structure and u_target and directly outputs the optimal rotation angles. No iterative optimization at inference — one
  forward pass gives you both the structure and the angles. The angle predictor can be trained with supervised loss against L-BFGS solutions found during GQE training.

  ---
  Direction 3: Diffusion or flow-matching over circuits

  Instead of autoregressive discrete sampling, model the full circuit as a continuous object and use score matching or flow matching to go from noise → circuit in one shot. This is an
  active research area for molecule and protein design, and circuits are structurally similar.

  ---
  Direction 4: Reinforcement learning with a richer action space

  Change the action space so each token is a (gate_type, qubit, angle) tuple — a continuous action. Use a policy gradient method like PPO with a Gaussian policy head for the angle
  component. The transformer becomes a full circuit generator in one autoregressive pass. The tradeoff is that continuous action spaces make RL much harder to train.
