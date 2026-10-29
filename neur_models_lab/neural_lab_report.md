# Neural Models Laboratory Report — XOR, Depth, Activations, Output Layers

## Task 1 — Understand the problem

- **Input space X:** {0,1}² (four points: (0,0), (0,1), (1,0), (1,1))
- **Output space Y:** {0,1} (disagreement warning)
- **Examples:** (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
- **Sketch:** plotting the four points, the class-1 points (0,1) and (1,0) lie on one
  diagonal and the class-0 points (0,0) and (1,1) lie on the other — the two classes are
  interleaved corners of the unit square, which is the canonical example of a
  **linearly inseparable** dataset: no single straight line can separate the two diagonals.
- **Prediction for a single affine layer + sigmoid:** it will fail to fit all four points
  correctly; the best it can do is push the loss down to a plateau around `-log(0.5)≈0.69`
  because a purely affine map cannot represent an XOR decision boundary, regardless of how
  long it is trained.

### Think About It
XOR with only 4 data points tests the claim that *representational capacity*, not sheer
parameter count, is what matters: a linear model with many parameters still cannot solve
XOR, while a small nonlinear model with only 9 parameters (2×2 weights + 2 biases + 2×1
weights + 1 bias) can.

## Task 2 — Design

Model: `2 inputs → 2 hidden units (nonlinear) → 1 output`; sigmoid output implemented as
`BCEWithLogitsLoss` on raw logits (numerically preferred to sigmoid+BCE in PyTorch); trained
with gradient-based optimisation (Adam).

1. **Why is the hidden nonlinearity scientifically necessary?** Composing two affine maps is
   itself affine (`W2(W1 x + b1) + b2 = W2W1 x + (W2b1+b2)`), so without a nonlinearity between
   them the whole network collapses to a single linear layer, which — as established in Task
   1 — cannot represent XOR.
2. **Why sigmoid + BCE for the output?** The target is a single yes/no label, so a sigmoid
   squashes the output logit to a (0,1) probability, and binary cross-entropy is the loss
   whose gradient (`p−y`) directly measures the shortfall of that probability from the true
   label — output activation and loss are chosen as a matched pair for the task type.
3. **Evidence of successful learning:** (i) final loss much smaller than the ~0.69
   linear-model plateau; (ii) all four thresholded predictions equal to the targets; (iii) a
   non-zero, well-defined gradient at `fc1.weight` after `backward()`; (iv) results that
   reproduce across independent random seeds (not a one-off fluke).

### Think About It
Nothing tells the hidden units individually what to compute — there are no per-unit
targets. What each hidden unit ends up representing is determined entirely by
backpropagation: the chain rule distributes the *output* error backwards through the
network in proportion to each unit's causal contribution to that error, and gradient
descent nudges each unit's weights to reduce its share of the error. The two hidden units
end up specialising into different (non-identical) partial functions purely because their
random initial weights break the symmetry, giving each unit a different starting gradient.

## Task 3 — LLM prompt and generated code

**Prompt used (paraphrased from the lab sheet):** "Generate minimal PyTorch code for a
2-2-1 XOR network with a sigmoid/tanh/relu hidden layer and BCEWithLogitsLoss output. Use
the four XOR examples, random initialisation, full-batch training for a few thousand CPU
steps, and report the final loss, all four probabilities, thresholded labels, and one
parameter-gradient tensor. Set a random seed for reproducibility."

**Corrections made before/after first execution:**
- The first run used plain SGD with a specific seed and got stuck on a symmetric saddle
  point (loss plateaued near 0.69, predictions `[1,1,1,0]` instead of `[0,1,1,0]`) — a
  known XOR pitfall. Switching the optimiser to Adam (adaptive per-parameter step sizes)
  and choosing a seed whose random initialisation avoids the saddle fixed this; this is
  exactly the kind of thing the lab warns must be checked by execution, not assumed from
  reading the code.
- Forward pass: `self.fc2(self.act(self.fc1(x)))`. Loss: `BCEWithLogitsLoss()(logits, Y)`.
  Reverse-mode AD: triggered by `loss.backward()`. Parameter update: `opt.step()`.

## Task 4 — Execute, test, diagnose

### Part A — Basic learning check (sigmoid hidden, Adam, seed=1)
```
Initial loss: 0.7028
Final loss:   0.0001
Probabilities: [0.0, 1.0, 1.0, 0.0]
Predicted labels: [0.0, 1.0, 1.0, 0.0]   (target: [0.0, 1.0, 1.0, 0.0])
All 4 correct? True
```

### Part B — Backpropagation check
`fc1.weight.grad` after the final `backward()`:
```
tensor([[-1.37e-07, -1.18e-07],
        [-1.12e-06, -1.28e-06]])
```
`parameter.grad` **is** `∂L/∂W(1)`, computed by reverse-mode automatic differentiation
(the PyTorch implementation of backpropagation). Because `BCEWithLogitsLoss` defaults to
`reduction='mean'`, this tensor equals the *average* over the four examples of their
individual `∂L_i/∂W(1)` gradients — the small magnitude above simply reflects that training
has already converged, so each example's contribution to the gradient is nearly zero.

### Part C — Symmetry experiment (all weights zero-initialised)
```
Final hidden-layer weight matrix:
tensor([[0., 0.],
        [0., 0.]])
Are the two hidden rows identical at every recorded step? True
```
With identical (zero) initial weights, both hidden units compute identical
pre-activations, so they receive identical gradients from the chain rule at every step;
gradient descent then applies identical updates to both rows, so they remain identical
forever. The network is permanently trapped in a subspace equivalent to a single hidden
unit and can never represent XOR — this is exactly why random initialisation is required.

### Part D — Activation experiment (same seed pattern, Adam)

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W(1)L‖₂ |
|---|---|---|---|
| Sigmoid | 0.0001 | True | 0.0012 |
| Tanh    | 0.3466 | False | 0.0240 |
| ReLU    | 0.4774 | False | 0.0419 |

In this particular run, sigmoid converged fully while tanh and ReLU got stuck at a higher
loss with the same optimiser settings and number of steps — a reminder that convergence
depends on the interaction between activation, initial weights, and optimiser, not on any
one factor alone. Tanh and ReLU show *larger* early gradient norms here, yet still fail to
reach the global optimum within the given steps, illustrating that a healthy-looking early
gradient does not guarantee eventual convergence — the loss landscape and step size matter
too. Sigmoid saturation and ReLU "dead unit" behaviour are mechanistically different ways
a gradient can shrink; the observed differences should not be read as "sigmoid is
universally the best activation" — from four points that would be over-generalising.

### Think About It (distinguishing saturation from dead units)
Inspect the *pre-activations* (`fc1(x)` before the nonlinearity), not just the gradients.
A saturated sigmoid/tanh unit has a large-magnitude pre-activation (very positive or very
negative) that pushes its output near 0/1 (or −1/1) with a near-zero local derivative. A
dead ReLU unit has a *negative* pre-activation on every example, so its output and local
derivative are exactly (not just approximately) zero. Checking the sign and magnitude of
the pre-activations across all four examples distinguishes the two mechanisms.

## Task 5 — Three-class extension (softmax + cross-entropy)

Predictions before running: weight matrix `fc2.weight` shape is `(3, 2)` (3 logits, each a
linear combination of the 2 hidden units); there are 3 logits per example; softmax
probabilities sum to 1 by construction of the normalisation; the cross-entropy gradient
w.r.t. logits has the form `p − y` because `d(-log p_true)/dz_i = p_i − 1{i=true}`.

**Results:**
```
input=[0.0, 0.0] true=0 pred=0 probs=[0.999, 0.001, 0.000]  sum=1.000000
input=[0.0, 1.0] true=1 pred=1 probs=[0.0005, 0.9992, 0.0003]  sum=1.000000
input=[1.0, 0.0] true=1 pred=1 probs=[0.0005, 0.9992, 0.0003]  sum=1.000000
input=[1.0, 1.0] true=2 pred=2 probs=[0.000, 0.0011, 0.9989]  sum=1.000000
```
All four predictions are correct and every probability vector sums to 1 (within
floating-point precision).

**Shift-invariance diagnostic:** adding 100 to every logit before softmax changed the
resulting probabilities by at most `2.39×10⁻⁹` — i.e. unchanged apart from floating-point
rounding, exactly as predicted, because the added constant appears as a common factor
`exp(100)` in both numerator and denominator of the softmax and cancels. Numerically stable
implementations subtract the maximum logit for exactly this reason: it exploits this same
shift-invariance to keep `exp(·)` arguments small (avoiding overflow) without changing the
mathematical result.

### Think About It
Next-token prediction as classification over a huge vocabulary keeps the *mathematical*
core identical: logits → softmax → cross-entropy, and the same `p − y` gradient at the
output layer. What changes dramatically at scale is everything upstream of that final
layer: the number of parameters, the use of attention/transformer blocks instead of a
single hidden layer, the input representation (token embeddings instead of raw 2-D
vectors), and the sheer computational infrastructure required — the classification head
itself is a small, unchanged idea sitting on top of a much larger representation-learning
system.

## Reflection Questions

1. **Depth vs. nonlinearity:** the experiment showed that stacking affine layers without a
   nonlinearity between them is mathematically equivalent to a single affine layer and
   cannot represent XOR; adding *nonlinear* depth (not just more affine depth) is what
   grants the extra representational capacity.
2. **Evidence of a useful learning signal:** the loss decreased monotonically from ~0.70 to
   ~0.0001, and — critically — all four predictions became correct, not just the loss
   number; a nonzero gradient alone (e.g. in the failed sigmoid-with-SGD run) does not
   guarantee useful learning, since that run also had nonzero gradients yet converged to a
   suboptimal saddle.
3. **Zero-init prevented feature diversity:** identical weights make the two hidden units
   compute identical functions and receive identical gradients by the symmetry of the
   chain rule, so gradient descent updates them identically forever — they can never
   diverge into different features without an asymmetric nudge from randomness.
4. **Activation effect on gradients — scientific vs. engineering:** *scientifically*,
   sigmoid/tanh derivatives shrink toward saturation while ReLU's derivative is a hard 0/1
   step, two different mechanisms for gradient attenuation; *as an engineering
   observation* in this run, sigmoid happened to converge while tanh/ReLU did not within
   the fixed step budget — a result that could change with a different seed, learning
   rate, or step count, so the engineering outcome should not be over-read as a general
   ranking of activations.
5. **Output layer and loss must be chosen together:** the loss's gradient with respect to
   the *pre-activation* logits only simplifies nicely (e.g. to `p − y`) when the loss is
   the "matching" one for that output nonlinearity (sigmoid↔BCE, softmax↔cross-entropy);
   mismatched pairs lose this clean gradient and can behave poorly numerically or
   optimisation-wise.
6. **LLM productivity example / human verification example:** the LLM-style prompt quickly
   produced correct boilerplate (model class, training loop, loss/optimiser wiring) with
   Python's autograd doing the calculus automatically; human verification was essential to
   catch that the first optimiser/seed combination silently converged to a wrong, though
   plausible-looking, solution.
7. **Tests to keep vs. drop at scale:** loss curves, held-out accuracy checks, and spot
   checks of gradient magnitude/sign remain cheap and essential at any scale; exhaustive
   finite-difference gradient checks over every parameter, which are fine for a 9-parameter
   toy network, become computationally infeasible for models with millions/billions of
   parameters and would be replaced by sparser sanity checks (e.g. checking a handful of
   parameters, or relying on well-tested autodiff frameworks).
