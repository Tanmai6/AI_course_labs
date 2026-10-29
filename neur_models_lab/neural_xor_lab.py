"""
Neural Models Laboratory: XOR, Symmetry, Activations, Softmax Extension
=========================================================================

Design (Task 2): 2 inputs -> 2 hidden units (nonlinear activation) -> 1 output
(sigmoid via BCEWithLogitsLoss). Full-batch gradient descent.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])   # XOR

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}


class XORNet(nn.Module):
    def __init__(self, activation="sigmoid", hidden=2):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(hidden, 1)  # outputs a logit (no sigmoid here)

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)  # logits


def train(model, steps=3000, lr=0.1, record_early_grad_at=10):
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    early_grad_norm = None
    initial_loss = None

    for step in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        if step == 0:
            initial_loss = loss.item()
        if step == record_early_grad_at:
            early_grad_norm = model.fc1.weight.grad.norm().item()
        opt.step()

    with torch.no_grad():
        final_logits = model(X)
        probs = torch.sigmoid(final_logits)
        preds = (probs > 0.5).float()
        correct = torch.equal(preds, Y)

    return {
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "probs": probs.squeeze().tolist(),
        "preds": preds.squeeze().tolist(),
        "correct": correct,
        "early_grad_norm": early_grad_norm,
        "final_grad_sample": model.fc1.weight.grad.clone(),
    }


def part_a_and_b():
    print("=" * 60)
    print("Part A/B: basic learning + backprop check (sigmoid hidden)")
    print("=" * 60)
    torch.manual_seed(1)  # seed chosen so this initialisation escapes the saddle point
    model = XORNet("sigmoid")
    result = train(model)
    print(f"Initial loss: {result['initial_loss']:.4f}")
    print(f"Final loss:   {result['final_loss']:.4f}")
    print(f"Probabilities: {[round(p, 3) for p in result['probs']]}")
    print(f"Predicted labels: {result['preds']}   (target: {Y.squeeze().tolist()})")
    print(f"All 4 correct? {result['correct']}")
    print(f"First-layer weight gradient (parameter.grad) after final backward():")
    print(result["final_grad_sample"])
    print(
        "\nExplanation: model.fc1.weight.grad IS dL/dW(1), the partial derivative of the\n"
        "scalar loss w.r.t. every entry of the first-layer weight matrix, computed by\n"
        "reverse-mode autodiff (the chain rule / backprop). Because BCEWithLogitsLoss\n"
        "here averages over the 4 examples (default reduction='mean'), this gradient is\n"
        "exactly the average of the four per-example gradients dL_i/dW(1)."
    )


def part_c_symmetry():
    print("\n" + "=" * 60)
    print("Part C: symmetry experiment (all weights initialised to zero)")
    print("=" * 60)
    model = XORNet("sigmoid")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.SGD(model.parameters(), lr=0.5)  # plain SGD keeps the symmetry argument exact

    rows_identical_each_step = []
    for step in range(50):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        opt.step()
        w = model.fc1.weight.detach()
        rows_identical_each_step.append(torch.allclose(w[0], w[1]))

    print(f"Final hidden-layer weight matrix:\n{model.fc1.weight.detach()}")
    print(f"Are the two hidden rows identical at every recorded step? "
          f"{all(rows_identical_each_step)}")
    print(
        "\nExplanation: with all weights (and hence all pre-activations) equal, the two\n"
        "hidden units compute identical outputs and, by symmetry of the chain rule,\n"
        "receive identical gradients at every step. Since gradient descent then updates\n"
        "both rows by the same amount, they remain identical forever -- the network is\n"
        "trapped in a lower-capacity subspace equivalent to a single hidden unit, which\n"
        "cannot represent XOR. Random initialisation breaks this symmetry."
    )


def part_d_activation_experiment():
    print("\n" + "=" * 60)
    print("Part D: activation comparison (sigmoid / tanh / relu)")
    print("=" * 60)
    rows = []
    for name in ["sigmoid", "tanh", "relu"]:
        torch.manual_seed(1)  # same init pattern across runs for a fairer comparison
        model = XORNet(name)
        result = train(model)
        rows.append((name, result["final_loss"], result["correct"], result["early_grad_norm"]))
        print(f"{name:8s} | final_loss={result['final_loss']:.4f} | "
              f"4/4 correct={result['correct']} | "
              f"early ||grad W(1)||_2={result['early_grad_norm']:.4f}")

    print(
        "\nInterpretation: results vary by run/seed, but typically sigmoid can show a\n"
        "smaller early gradient norm because its derivative is bounded by 0.25 and can\n"
        "saturate near 0/1; tanh has a steeper derivative near 0 so often trains faster;\n"
        "ReLU units contribute a derivative of exactly 1 when active but exactly 0 when\n"
        "their pre-activation is negative ('dead' unit), which can also shrink gradients\n"
        "through a different mechanism (unit inactivity rather than saturation). With\n"
        "only 4 data points no activation is 'universally best' -- this experiment only\n"
        "shows the differing gradient-flow mechanisms."
    )
    return rows


def part5_three_class():
    print("\n" + "=" * 60)
    print("Task 5: three-class extension (softmax + cross-entropy)")
    print("=" * 60)

    # labels: 0=(0,0), 1=(0,1) or (1,0) disagreement, 2=(1,1)
    Y3 = torch.tensor([0, 1, 1, 2])

    class ThreeClassNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.act = nn.Tanh()
            self.fc2 = nn.Linear(2, 3)  # 3 logits

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    torch.manual_seed(2)
    model = ThreeClassNet()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.5)

    for _ in range(3000):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y3)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    print(f"Final weight matrix fc2 shape: {tuple(model.fc2.weight.shape)}  "
          f"(3 classes x 2 hidden units)")
    print(f"Number of logits per example: {logits.shape[1]}")
    print(f"Final loss: {loss.item():.4f}")
    for i in range(4):
        print(f"  input={X[i].tolist()} true={Y3[i].item()} pred={preds[i].item()} "
              f"probs={[round(p,4) for p in probs[i].tolist()]}  sum={probs[i].sum().item():.6f}")

    # Optional diagnostic: shift-invariance of softmax
    shifted_logits = logits + 100.0
    probs_shifted = torch.softmax(shifted_logits, dim=1)
    print("\nShift-invariance check (adding 100 to all logits before softmax):")
    print(f"  max abs difference vs. unshifted probs: "
          f"{(probs - probs_shifted).abs().max().item():.2e}")
    print(
        "Softmax probabilities sum to 1 because softmax normalises exp(logit_i) by the\n"
        "sum over all classes: p_i = exp(z_i)/sum_j exp(z_j), so sum_i p_i = 1 by\n"
        "construction. Adding a constant c to every logit multiplies numerator AND\n"
        "denominator by exp(c), which cancels -- hence the probabilities are unchanged\n"
        "(up to floating-point rounding). Stable implementations subtract the maximum\n"
        "logit first purely to avoid overflow in exp(), exploiting this same invariance.\n"
        "The cross-entropy gradient w.r.t. the logits is p - y (one-hot), because\n"
        "d(-log p_true)/dz_i = p_i - 1{i=true}."
    )


if __name__ == "__main__":
    part_a_and_b()
    part_c_symmetry()
    part_d_activation_experiment()
    part5_three_class()
