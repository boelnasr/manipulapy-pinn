# manipulapy-pinn

Physics-informed neural networks for robot manipulators, built entirely on
[ManipulaPy](https://github.com/boelnasr/ManipulaPy)'s kinematics, dynamics,
and (as of v1.4) differentiable backend system. Three tasks:

| Task | What it learns | Physics signal |
|---|---|---|
| [`forward_dynamics`](src/manipulapy_pinn/forward_dynamics.py) | q̈ = f<sub>θ</sub>(q, q̇, τ) | Equation-of-motion residual against ManipulaPy's true M(q), C(q,q̇), g(q) |
| [`inverse_kinematics`](src/manipulapy_pinn/inverse_kinematics.py) | q = h<sub>θ</sub>(target pose) | Forward-kinematics residual, differentiated live through ManipulaPy's FK |
| [`trajectory`](src/manipulapy_pinn/trajectory.py) | q<sub>θ</sub>(t), a whole trajectory | EOM residual through a *trained* forward-dynamics surrogate, collocation-style |

ManipulaPy is the ground truth throughout — every physics term in every loss
function traces back to a ManipulaPy kinematics or dynamics call, not a
hand-derived approximation of one.

## Why this exists

Most of the hard problems in robot motion are optimization problems with a
physical constraint baked in: inverse kinematics minimizes pose error
subject to "the forward-kinematics map must hold"; trajectory generation
minimizes effort or time subject to "the equations of motion must hold".
Classical solvers handle the constraint by construction (an iterative
Jacobian solver, a recursive Newton-Euler dynamics pass). A **physics-informed
neural network** instead *learns* a function and is penalized whenever it
violates the constraint — trading a per-query iterative solve for a one-time
training cost and (thereafter) instant, differentiable inference.

ManipulaPy v1.4 made this practical to build cheaply: `forward_kinematics`,
`mass_matrix`, and friends now run under a PyTorch (or JAX) backend with a
tested autodiff contract, so "differentiate a physics residual through the
real kinematics/dynamics" is a few lines of code instead of a from-scratch
reimplementation.

## Install

```bash
git clone https://github.com/boelnasr/manipulapy-pinn.git
cd manipulapy-pinn
pip install -e ".[dev]"
```

Pulls in `ManipulaPy[pytorch]` — the default install has no CUDA
requirement; everything here runs on CPU (see [Performance](#performance)
for what that costs).

On Linux, `pip install torch` pulls the CUDA build and ~2.5 GB of `nvidia-*`
wheels this package never touches. To skip them:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### Without a local machine

The repo ships a [dev container](.devcontainer/devcontainer.json), so
**GitHub Codespaces** (Code → Codespaces → Create codespace) builds a ready
environment with no local setup — CPU-only torch, the package installed
editable, and a verification step that loads `panda` and prints the run
commands. The default 2-core machine is more than enough: a full training
cycle is under two minutes.

No GPU is needed anywhere. The networks are three 128-wide float64 layers,
and the inverse-kinematics loop is bounded by a per-sample Python call into
ManipulaPy's FK rather than by arithmetic — a GPU would not help and would
likely hurt.

## Quick start

```python
import numpy as np
from manipulapy_pinn import load_robot
from manipulapy_pinn.forward_dynamics import train

robot = load_robot("panda")               # any of the 25 robots ManipulaPy bundles
result = train(robot, n_samples=2000, iterations=1500)
print(f"held-out q̈ RMSE: {result.val_data_rmse:.3f} rad/s²")

# result.model is a plain torch.nn.Module — q̈ = model(q, qdot, tau)
```

Or from the command line, with plots:

```bash
python scripts/train_forward_dynamics.py --robot panda
python scripts/train_inverse_kinematics.py --robot panda
python scripts/train_trajectory.py --robot panda
python scripts/benchmark.py --robot panda   # vs. ManipulaPy's own classical solvers
```

Each `train_*.py` script saves a checkpoint and a figure under `runs/`.

## How each task is physics-informed

### Forward dynamics

```
loss = data_weight · MSE(q̈_pred, q̈_true)
     + physics_weight · mean[ (τ − (M(q) q̈_pred + C(q,q̇) + g(q)))² ]
```

M, C, g depend only on the sampled (q, q̇) — never on the network's own
output — so they're computed **once**, offline, through ManipulaPy's fast
NumPy dynamics, and reused as constants at every training step. Only q̈_pred
needs a gradient. This matters for tractability: differentiating
ManipulaPy's Coriolis term (`velocity_quadratic_forces`) live under the
torch backend costs on the order of 250ms/call on this package's reference
hardware (a 7-link recursive computation with many small tensor ops —
PyTorch's per-op eager dispatch overhead dominates at that granularity).
Precomputing sidesteps that entirely while staying exactly as physically
grounded.

### Inverse kinematics

```
loss = mean[ ‖ FK(h_θ(x)) − x ‖² ]
```

No stored (pose, q) table — the target `x` feeds the network, whose output
`q` feeds straight into ManipulaPy's `forward_kinematics` **live**, under
the torch backend, with autograd tracking through the FK call itself. This
is the one place in the package where ManipulaPy runs inside the training
loop rather than being precomputed, and it's tractable because FK is cheap
(~3ms/call here) — nothing like the dynamics Coriolis cost above. It also
sidesteps a real correctness trap at a redundant robot: supervising against
one arbitrarily-chosen reference q would penalize other, equally valid
solutions the network might reasonably find instead.

### Trajectory

The classic Raissi-style PINN-for-ODEs framing: represent the whole
trajectory as q<sub>θ</sub>(t), get q̇(t) and q̈(t) for free via
`torch.autograd.grad` (twice, w.r.t. t), and penalize departure from the
equation of motion at a set of collocation points:

```
loss = physics_weight · mean[ (f_φ(q, q̇, τ_θ(t)) − q̈)² ]     # f_φ = trained forward-dynamics PINN
     + effort_weight   · mean[ τ_θ(t)² ]                       # minimal control effort
     + boundary_weight · (‖q̇(0)‖² + ‖q̇(T)‖²)                  # start/end at rest
```

Position boundary conditions are satisfied **exactly**, for any network
weights, via the ansatz

```
q_θ(t) = q_start + t·(q_goal − q_start) + t·(1−t)·NN_θ(t)
```

— `t·(1−t)` vanishes at both endpoints, so there's nothing for the
optimizer to get slightly wrong there; only the velocity boundary condition
needs a soft penalty. The physics term uses a *trained*
`ForwardDynamicsPINN` in place of calling ManipulaPy's dynamics directly, for
the same tractability reason as above — see
[`trajectory.py`](src/manipulapy_pinn/trajectory.py)'s module docstring for
the full reasoning and the honest cost of that trade (the trajectory
optimizer is only as accurate as the surrogate it's built on).

## Adding a new robot

Anything in `ManipulaPy.ManipulaPy_data.list_robots()` works — add an entry
to `_ARM_DOF` in [`robots.py`](src/manipulapy_pinn/robots.py) mapping the
robot name to how many of its actuated joints count as "the arm" (excluding
grippers):

```python
_ARM_DOF = {
    "panda": 7,
    "ur5": 6,
    ...
    "my_new_robot": 6,
}
```

Worth a quick sanity check before trusting the defaults: some bundled URDFs
carry a near-degenerate mass-matrix entry on one joint (`xarm6`'s last joint
is ~8e-5, apparently a placeholder inertia tag) that blows sampled
accelerations up to unphysical magnitudes under the default torque range —
finite, so it silently passes shape/NaN checks, but it makes the forward-
dynamics dataset meaningless. Print `robot.dynamics.mass_matrix(q)` for a
few sampled configs and check nothing on the diagonal is many orders of
magnitude smaller than the rest before relying on this package's default
`tau_scale` for a new robot.

## Performance

Numbers below are from this package's reference hardware (CPU-only, no
CUDA) — reproduce on your own machine with `python scripts/benchmark.py`.
They're here to make the actual trade-off legible, not to claim the PINN
"wins":

| | ManipulaPy (classical) | PINN |
|---|---|---|
| Forward dynamics, 1 query | ~6 ms (exact) | a fraction of a ms (surrogate) |
| Inverse kinematics, 1 query | ~10 ms median (DLS, converges to tolerance) | a fraction of a ms (single forward pass, less accurate) |
| Trajectory (start→goal) | fast (kinematic-only quintic; no dynamics/effort term) | seconds (solves for torque + trajectory jointly, dynamics-consistent) |

The PINN numbers above don't include training time (data generation +
optimization): a few seconds for forward dynamics, one to two minutes for
inverse kinematics, and the same order for a trajectory solve — see each
`train_*.py` script's `--iterations` default and `scripts/benchmark.py`'s
output for exact figures on your hardware. That cost amortizes across every
query made after training; a single one-off query is almost always cheaper
with ManipulaPy's classical solvers directly.

## Known limitations

- **Trajectory: no velocity/torque limits.** The loss softly minimizes
  effort and pins the endpoints at rest, but nothing bounds the interior of
  the trajectory — unconstrained solves have produced peak joint speeds well
  past what a real robot could achieve. See the "Known limitation" note in
  [`trajectory.py`](src/manipulapy_pinn/trajectory.py) for the fix (a
  clipped-penalty term, the same pattern ManipulaPy's own
  `differentiable_reach_showcase.py` uses for obstacle clearance) — left out
  here to keep this a minimal example of the collocation pattern itself.
- **Inverse kinematics: less accurate than the classical solvers it's
  compared against**, by design — a single forward pass trades exactness for
  speed. A natural extension is a hybrid: use the PINN's output as the
  initial guess for ManipulaPy's `iterative_inverse_kinematics`, which
  typically converges faster from a good guess than from zero.
- **Forward dynamics: accuracy scales with training budget.** The default
  `train()` settings are a reasonable demo point (see `scripts/benchmark.py`
  output for actual numbers on your hardware), not a tuned, converged model
  — more samples, more iterations, and a wider network all still improve
  held-out RMSE at the point this was written.

## Choosing samples and iterations together

These two are not independent, and tuning one against the wrong value of the
other is the easiest way to get a worse model. Measured on `panda`, held-out
q̈ RMSE, two seeds per cell:

| samples | 1500 iters | 4000 iters | 8000 iters | 12000 iters |
|---:|---:|---:|---:|---:|
| 2000 | **3.15** | 3.26 | — | — |
| 8000 | — | 1.77 | 1.59 | **1.51** |

At 8000 samples the held-out curve is still descending at 12000 iterations, so
that column is a floor rather than an optimum — there is more to gain by
running longer.

Read it in both directions. At **2000 samples**, going from 1500 to 4000
iterations makes the model *worse* — held-out error bottoms around iteration
1000 and rises from there while the training loss keeps falling. At **8000
samples** the same extra iterations keep helping, and at 12000 the held-out
curve is still descending with a train/val gap of only 1.36×.

So the rule is: **raise samples first, then iterations.** More steps on too
little data is not a slower path to the same place — it is a worse model.

```bash
python scripts/train_forward_dynamics.py --samples 8000 --iterations 12000
```

Every run now plots held-out error next to the training loss and marks the
best iteration, so you can read the crossover for your own configuration
rather than trusting this table. If the best iteration lands well before the
end of your run, training prints a warning saying so.

### Depth

`--depth` above 4 switches from a plain stack to residual blocks
automatically. At 8000 samples, held-out RMSE with the train/val gap in
brackets:

| iterations | depth 3 | depth 5 |
|---:|---:|---:|
| 2000 | 2.042 (1.14×) | **1.778** (1.28×) |
| 4000 | 1.772 (1.20×) | 1.628 (1.44×) |
| 8000 | 1.591 (1.29×) | 1.541 (1.71×) |
| 12000 | **1.513** (1.36×) | 1.529 (2.02×) |

Depth buys **convergence speed, not final accuracy**. Depth 5 reaches in 2000
iterations roughly what depth 3 needs 4000 to reach — but by 12000 they meet,
and depth 3 edges ahead. Since depth 5 costs about twice as much per
iteration, that early advantage is spent paying for it.

The gap column shows why they meet: depth 5's train/val gap grows to 2.02×
while depth 3 stays at 1.36×. The deeper network spends its extra capacity
memorizing. Its held-out curve has already flattened by 12000 (1.541 → 1.529)
where depth 3's is still descending (1.591 → 1.513).

So forward dynamics defaults to depth 3. **Inverse kinematics defaults to
depth 5**, where the same comparison comes out the other way (92.7 mm → 57.0
mm at its default budget) — that task draws fresh targets every step, so
there is no fixed set to memorize and extra capacity goes into the map
itself. Same architecture, opposite verdict, decided by whether the data is
finite.

## Evaluating a trained model

Every `train_*.py` script scores its model after training and writes a
Markdown report next to the checkpoint and figure:

```
runs/forward_dynamics_panda.md
runs/inverse_kinematics_panda.md
runs/trajectory_panda.md
```

Each report records the full configuration and environment (so a run is
reproducible), the loss trajectory, the complete metric block, and a
**Review** section listing anything an automatic threshold check flagged.

The metrics ([`metrics.py`](src/manipulapy_pinn/metrics.py)) deliberately
measure what a training loss cannot:

| Task | Beyond the loss |
|---|---|
| Forward dynamics | per-joint and scale-free (normalized) RMSE, R², **skill score** against a constant-mean predictor, and the acceleration error mapped through the true mass matrix into N·m |
| Inverse kinematics | median/p95/max reach error, **success rate at 1/5/10/50 mm**, **joint-limit violation rate**, and the orientation error the position-only loss leaves unconstrained |
| Trajectory | torque error against ManipulaPy's **true `inverse_dynamics`** rather than the learned surrogate the solve optimized through, evaluated on a dense grid *and* on the training collocation points so the two can be compared |

Call them directly on any trained model:

```python
from manipulapy_pinn.metrics import forward_dynamics_metrics, format_metrics

print(format_metrics(forward_dynamics_metrics(result.model, robot), "Forward dynamics"))
```

Three of these check things nothing in training constrains — an IK network's
output is not bounded by the robot's joint limits, and a trajectory's
interior is bounded by neither velocity nor torque — so a model can score
well on its loss and still be unusable. The review section says so in
words.

## Testing

```bash
python -m pytest tests/ -q
```

The suite trains everything with deliberately tiny configurations (small
networks, ~15–60 iterations) — it checks correctness (shapes, finiteness,
boundary conditions, loss trending down, and the EOM/FK residuals actually
vanishing where they mathematically must) in a few seconds, not
convergence quality. For that, run the `scripts/` above with their full
defaults.

## Related work

[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) is a literature review of the
research this package draws on — physics-informed dynamics learning, learned
inverse kinematics, PINN trajectory optimization, and differentiable rigid-body
simulation — written to be useful rather than flattering. It includes a section
on [what the literature says the design here gets
wrong](docs/RELATED_WORK.md#what-the-literature-says-we-got-wrong) and a
[ranked list of follow-up work](docs/RELATED_WORK.md#ranked-next-steps).

Three findings from it are worth surfacing here, because they bear directly on
claims made above:

- **The forward-dynamics physics term is algebraically equivalent to the data
  term under a mass-matrix metric.** Because the labels and the `M, C, g`
  operators come from the same ManipulaPy model, the EOM residual reduces to
  `−M(q̈_pred − q̈_true)`, so `physics_loss = ‖MΔ‖²` where `data_loss = ‖Δ‖²`.
  Same minimizer, no independent signal. The term becomes genuinely informative
  once the labels come from somewhere else (measurements, noise, a different
  simulator).
- **The ~250 ms Coriolis-differentiation cost is a property of op-level eager
  autodiff, not of the derivative itself.** Analytical RNEA derivatives put the
  same quantity at ~3–5 µs for a 7-DoF arm (Carpentier & Mansard, RSS 2018;
  Le Lidec et al., 2024), and `torch.compile` alone is measured at ~4.7× on a
  comparable arm at batch 1.
- **The IK accuracy result is a reproduction, not a finding.** A published
  benchmark of 12 solvers on the Franka Panda measures a plain MLP at 0%
  success / ~10 mm — and shows warm-started refinement lifting learned solvers
  to 98.6–100%, converging from seeds up to 207 mm off. The warm-start
  extension proposed above is the empirically validated fix, and the network
  does not need to be accurate to be a useful seed.

## Relationship to ManipulaPy

This package is a downstream consumer of
[ManipulaPy](https://github.com/boelnasr/ManipulaPy) — it doesn't
reimplement any kinematics or dynamics, only trains neural surrogates
against ManipulaPy's own. If you're new to ManipulaPy itself, its
[`showcase/`](https://github.com/boelnasr/ManipulaPy/tree/main/showcase)
directory and
[`notebooks/12_differentiable_robotics.ipynb`](https://github.com/boelnasr/ManipulaPy/blob/main/notebooks/12_differentiable_robotics.ipynb)
are the place to start on the differentiable-backend contract this package
relies on throughout.

## License

[AGPL-3.0-or-later](LICENSE), matching ManipulaPy.
