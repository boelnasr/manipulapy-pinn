<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Related work

A literature review of the three things this package does — physics-informed
forward dynamics, FK-residual inverse kinematics, and PINN-collocation
trajectory optimization — and of the differentiable-physics substrate all
three sit on.

This document is written to be *useful rather than flattering*. Each section
ends with what the literature says about the specific choices in this
codebase, including where it says those choices are wrong or superseded. The
sharpest results are collected in [What the literature says we got
wrong](#what-the-literature-says-we-got-wrong), and the concrete follow-up
work is in [Ranked next steps](#ranked-next-steps).

> **On verification.** Every reference below was located by literature search
> and confirmed against an indexed publisher or arXiv landing page carrying a
> matching title and identifier. Direct fetching of arxiv.org,
> semanticscholar.org, doi.org and most publisher hosts is blocked by this
> environment's network egress policy, so the confirmation is
> "search index returned this exact title at this canonical URL," not "the
> abstract page was retrieved and read." Treat identifiers as high-confidence
> but verify a DOI before putting it in a camera-ready bibliography. Nothing
> here is reported from memory alone.

---

## Contents

- [1. PINN methodology — what this package inherits](#1-pinn-methodology--what-this-package-inherits)
- [2. Forward dynamics](#2-forward-dynamics)
- [3. Inverse kinematics](#3-inverse-kinematics)
- [4. Trajectory optimization](#4-trajectory-optimization)
- [5. Differentiable simulation and the 250 ms Coriolis cost](#5-differentiable-simulation-and-the-250-ms-coriolis-cost)
- [6. Is a PINN worth it at all?](#6-is-a-pinn-worth-it-at-all)
- [What the literature says we got wrong](#what-the-literature-says-we-got-wrong)
- [Ranked next steps](#ranked-next-steps)

---

## 1. PINN methodology — what this package inherits

### Foundations

| Work | Reference |
|---|---|
| Lagaris, Likas, Fotiadis. *Artificial Neural Networks for Solving Ordinary and Partial Differential Equations* | IEEE Trans. Neural Networks 9(5):987–1000, 1998 · [arXiv:physics/9705023](https://arxiv.org/abs/physics/9705023) |
| Raissi, Perdikaris, Karniadakis. *Physics-Informed Deep Learning (Part I)* | [arXiv:1711.10561](https://arxiv.org/abs/1711.10561) |
| Raissi, Perdikaris, Karniadakis. *Physics-informed neural networks* | J. Comput. Phys. 378:686–707, 2019 · DOI [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) |
| Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, Yang. *Physics-informed machine learning* | Nature Reviews Physics 3:422–440, 2021 · DOI [10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5) |
| Cuomo et al. *Scientific Machine Learning Through PINNs: Where we are and What's Next* | J. Sci. Comput. 92:88, 2022 · DOI [10.1007/s10915-022-01939-z](https://doi.org/10.1007/s10915-022-01939-z) |
| Wang, Sankaran, Wang, Perdikaris. *An Expert's Guide to Training PINNs* | [arXiv:2308.08468](https://arxiv.org/abs/2308.08468) |

Raissi et al. is the template `trajectory.py` follows directly. Lagaris et al.
matters more than its age suggests: it is the origin of the *trial solution*,
in which boundary conditions are satisfied by an algebraic construction rather
than a penalty — which is exactly the ansatz in
[`TrajectoryAnsatz`](../src/manipulapy_pinn/trajectory.py). Lagaris et al. had
no boundary loss term at all, and therefore no weight to tune against the
residual.

Karniadakis et al. supply a distinction worth adopting in this package's own
vocabulary: physics can enter as a **learning bias** (a loss term — what all
three modules here do) or as an **inductive bias** (structure in the
architecture — what Section 2's structured-dynamics literature does instead).
Cuomo et al. name the consequence: PINN training is *multi-task learning*, so
`data_weight` and `physics_weight` are a multi-objective scalarization, not
two independent knobs.

Wang et al.'s *Expert's Guide* is the most practically useful single reference
for anyone extending this package — it ablates which interventions
(non-dimensionalization, Fourier features, causal weighting, adaptive weights)
actually pay off.

### Loss weighting

`forward_dynamics.train()` uses `data_weight=1.0, physics_weight=1.0`, fixed
for the whole run. `trajectory.solve()` similarly fixes three weights. This is
the single most-criticized design choice in the PINN literature, and there are
three *independent* reasons it is wrong:

| Failure | Reference |
|---|---|
| **Magnitude imbalance, and it drifts.** Back-propagated gradients from different loss terms differ by orders of magnitude and the ratio changes during training, so any constant weight is correct for at most one moment of one run. Proposes online learning-rate annealing from gradient statistics. | Wang, Teng, Perdikaris. *Understanding and Mitigating Gradient Flow Pathologies in PINNs.* SIAM J. Sci. Comput. 43(5):A3055–A3081, 2021 · [arXiv:2001.04536](https://arxiv.org/abs/2001.04536) · [code](https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs) |
| **Convergence-rate mismatch.** The NTK's data-block and residual-block eigenvalues differ by orders of magnitude, so the two terms converge at completely different rates. Gives an NTK-trace recipe for the weights. | Wang, Yu, Perdikaris. *When and why PINNs fail to train: A neural tangent kernel perspective.* J. Comput. Phys. 449:110768, 2022 · [arXiv:2007.14527](https://arxiv.org/abs/2007.14527) |
| **Directional conflict.** The two gradients can point in *opposing* directions, in which case no scalar weighting produces an update that improves both. Constructs a provably conflict-free update. | Liu, Chu, Thuerey. *ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks.* ICLR 2025 (Spotlight) · [arXiv:2408.11104](https://arxiv.org/abs/2408.11104) |

Remedies, cheapest first:

- **ReLoBRaLo** — Bischof & Kraus, *Multi-Objective Loss Balancing for Physics-Informed Deep Learning*, [arXiv:2110.09813](https://arxiv.org/abs/2110.09813); CMAME 2025, DOI [10.1016/j.cma.2025.117914](https://doi.org/10.1016/j.cma.2025.117914). Needs only loss *values*, not per-term gradients, so it is nearly free to bolt onto the existing Adam loop. The paper also benchmarks GradNorm, SoftAdapt and learning-rate annealing.
- **Self-adaptive PINNs** — McClenny & Braga-Neto, J. Comput. Phys. 474:111722, 2023 · [arXiv:2009.04544](https://arxiv.org/abs/2009.04544). Per-collocation-point trainable weights via a minimax formulation; argues a correctly balanced *global* weight is still insufficient when difficulty is spatially concentrated.
- **Augmented Lagrangian** — Lu, Pestourie, Yao, Wang, Verdugo, Johnson, *hPINN*, SIAM J. Sci. Comput. 43(6):B1105, 2021 · [arXiv:2102.04626](https://arxiv.org/abs/2102.04626); and Son, Cho, Hwang, *AL-PINNs*, Neurocomputing 548:126424, 2023 · [arXiv:2205.01059](https://arxiv.org/abs/2205.01059). The most principled option: the weight becomes a multiplier *solved for* rather than tuned.

### Failure modes

| Work | Reference | Why it matters here |
|---|---|---|
| Krishnapriyan, Gholami, Zhe, Kirby, Mahoney. *Characterizing possible failure modes in physics-informed neural networks* | NeurIPS 2021 · [arXiv:2109.01050](https://arxiv.org/abs/2109.01050) · [code](https://github.com/a1k12/characterizing-pinns-failure-modes) | The definitive result: PINNs fail on mildly harder problems, and the failure is in *optimization*, not capacity — the true solution has lower loss than the one found, and bigger networks do not help. Raising `physics_weight` can worsen conditioning rather than help. |
| Wang, Sankaran, Perdikaris. *Respecting causality for training PINNs* | CMAME 421:116813, 2024 · [arXiv:2203.07404](https://arxiv.org/abs/2203.07404) · [code](https://github.com/PredictiveIntelligenceLab/CausalPINNs) | Weighting all collocation times equally trains the network to fit late times before early times are right. `trajectory.solve()` does exactly this. The fix is a cheap per-timestep discount by accumulated earlier residual, which doubles as a convergence diagnostic. |
| Daw, Bu, Wang, Perdikaris, Karpatne. *Mitigating Propagation Failures in PINNs using R3 Sampling* | ICML 2023, PMLR 202:7264–7302 · [arXiv:2207.02338](https://arxiv.org/abs/2207.02338) | Makes adaptive collocation a *correctness* issue: with fixed points, the residual can go low while initial-condition information never propagates into the interior, collapsing to a trivial solution. |
| Andersen & Matsubara. *PINNs Failure Modes are Overfitting* | [arXiv:2605.30910](https://arxiv.org/abs/2605.30910) | Reframes the above: the residual really is minimized *at* the collocation points but not *between* them. Reports state of the art on four failure-mode benchmarks with a vanilla architecture and up to 23× fewer points, using regularization plus double backprop over the residual. |
| Wang, Wang, Perdikaris. *On the eigenvector bias of Fourier feature networks* | CMAME 384:113938, 2021 · [arXiv:2012.10047](https://arxiv.org/abs/2012.10047) | Spectral bias: PINNs learn low frequencies first and high frequencies last or never. A Fourier-feature input layer is a smaller change than any weighting scheme with a larger payoff on oscillatory trajectories. |
| Ji, Qiu, Shi, Pan, Deng. *Stiff-PINN* | J. Phys. Chem. A 125(36):8098–8106, 2021 · [arXiv:2011.04520](https://arxiv.org/abs/2011.04520) | The clearest ODE-side evidence that vanilla PINNs fail on stiff systems, and that the fix is reformulating the governing equations, not increasing the residual weight. |

**Direct consequence for this package:** `trajectory.solve()` evaluates its
residual on `torch.linspace(0.0, 1.0, n_collocation)` — a *fixed, uniform,
never-resampled* grid, identical at every one of the 800 iterations. Daw et al.
and Andersen & Matsubara jointly say that a low reported residual on that grid
is not evidence of a physically correct trajectory between the points. The
package's `TrajectoryResult.sample(n=100)` evaluates on a denser grid, which is
the right instinct; what is missing is *checking the residual there* rather
than only plotting the result.

### Optimizer

Every module here trains with Adam alone.

- Rathore, Lei, Frangella, Lu, Udell. *Challenges in Training PINNs: A Loss Landscape Perspective.* ICML 2024 · [arXiv:2402.01868](https://arxiv.org/abs/2402.01868). Ill-conditioning from the differential operator makes first-order methods struggle; **Adam→L-BFGS beats either alone**, and their NysNewton-CG improves further. This is the cheapest available accuracy win — a few lines, no new hyperparameters.
- Müller & Zeinhofer. *Achieving High Accuracy with PINNs via Energy Natural Gradient Descent.* ICML 2023 · [arXiv:2302.13163](https://arxiv.org/abs/2302.13163). The Adam-to-second-order gap is measured in *orders of magnitude*, not percent.
- Wang, Bhartari, Li, Perdikaris. *Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective.* NeurIPS 2025 · [arXiv:2502.00604](https://arxiv.org/abs/2502.00604). The synthesis: the weighting problem and the optimizer problem are *the same problem*. A better optimizer (they recommend SOAP) partially dissolves the need for a weighting scheme.

### Collocation sampling

Wu, Zhu, Tan, Kartha, Lu. *A comprehensive study of non-adaptive and
residual-based adaptive sampling for physics-informed neural networks.* CMAME
403:115671, 2023 · [arXiv:2207.10289](https://arxiv.org/abs/2207.10289) ·
[code](https://github.com/lu-group/pinn-sampling). Benchmarks six non-adaptive
strategies against residual-based adaptive ones; RAD wins overall. The
low-effort takeaway for this package: even switching the fixed uniform grid to
a Sobol or Hammersley sequence, resampled per iteration, is a one-line change
with measurable benefit before any adaptive machinery.

---

## 2. Forward dynamics

`forward_dynamics.py` learns `q̈ = f_θ(q, q̇, τ)` from data, plus the EOM
residual `τ − (M q̈_pred + C + g)` with `M, C, g` precomputed offline as
constants.

### The structured alternative: learn the physics, don't penalize it

| Work | Reference |
|---|---|
| Lutter, Ritter, Peters. *Deep Lagrangian Networks (DeLaN)* | ICLR 2019 · [arXiv:1907.04490](https://arxiv.org/abs/1907.04490) · [code](https://github.com/milutter/deep_lagrangian_networks) |
| Lutter, Listmann, Peters. *DeLaN for End-to-End Learning of Energy-Based Control* | IROS 2019 · [arXiv:1907.04489](https://arxiv.org/abs/1907.04489) |
| Lutter & Peters. *Combining Physics and Deep Learning to Learn Continuous-Time Dynamics Models* | Int. J. Robotics Research 42(3), 2023 · [arXiv:2110.01894](https://arxiv.org/abs/2110.01894) · DOI [10.1177/02783649231169492](https://doi.org/10.1177/02783649231169492) |
| Greydanus, Dzamba, Yosinski. *Hamiltonian Neural Networks* | NeurIPS 2019 · [arXiv:1906.01563](https://arxiv.org/abs/1906.01563) |
| Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho. *Lagrangian Neural Networks* | ICLR 2020 Workshop · [arXiv:2003.04630](https://arxiv.org/abs/2003.04630) |
| Zhong, Dey, Chakraborty. *Symplectic ODE-Net* | ICLR 2020 · [arXiv:1909.12077](https://arxiv.org/abs/1909.12077) |
| Zhong, Dey, Chakraborty. *Benchmarking Energy-Conserving Neural Networks* | L4DC 2021, PMLR 144:1218–1229 · [arXiv:2012.02334](https://arxiv.org/abs/2012.02334) |
| Gupta, Menda, Manchester, Kochenderfer. *Structured Mechanical Models* | L4DC 2020, PMLR 120:328–337 · [arXiv:2004.10301](https://arxiv.org/abs/2004.10301) |

DeLaN is the canonical alternative to this package's design: rather than
injecting fixed `M, C, g` as a residual penalty, it *parameterizes* the
Lagrangian — one network emits a Cholesky factor of `M(q)` (so
positive-definiteness holds by construction), another the potential `V(q)` —
and obtains Coriolis terms by autodiff through Euler–Lagrange. Both forward
and inverse dynamics fall out of the same learned object, and it extrapolates
to `(q, q̇)` regions never visited at training time, where this package's
frozen constants have nothing to say.

Structured Mechanical Models is the closer design pattern to steal: black-box
approximators placed at the specific `M`, `C`, `g` slots of the Euler–Lagrange
equations — learnable and inspectable, rather than frozen offline.

Lagrangian Neural Networks matters because it drops HNN's canonical-coordinate
requirement, and joint coordinates `q` are precisely the non-canonical setting.

### Semi-parametric identification: analytical model + learned residual

| Work | Reference |
|---|---|
| Nguyen-Tuong & Peters. *Model Learning for Robot Control: A Survey* | Cognitive Processing 12(4):319–340, 2011 · DOI [10.1007/s10339-011-0404-1](https://doi.org/10.1007/s10339-011-0404-1) |
| Nguyen-Tuong & Peters. *Using Model Knowledge for Learning Inverse Dynamics* | ICRA 2010, pp. 2677–2682 |
| Camoriano, Traversaro, Rosasco, Metta, Nori. *Incremental Semiparametric Inverse Dynamics Learning* | ICRA 2016 · [arXiv:1601.04549](https://arxiv.org/abs/1601.04549) |
| Romeres, Zorzi, Camoriano, Chiuso. *Online Semi-Parametric Learning for Inverse Dynamics Modeling* | CDC 2016 · [arXiv:1603.05412](https://arxiv.org/abs/1603.05412) |
| Reuss, van Duijkeren, Krug, Becker, Shaj, Neumann. *End-to-End Learning of Hybrid Inverse Dynamics Models for Precise and Compliant Impedance Control* | RSS 2022 · [arXiv:2205.13804](https://arxiv.org/abs/2205.13804) |
| Giacomuzzo, Carli, Romeres, Dalla Libera. *A Black-Box Physics-Informed Estimator Based on GPR for Robot Inverse Dynamics Identification* | IEEE T-RO, 2024 · [arXiv:2310.06585](https://arxiv.org/abs/2310.06585) · [code](https://github.com/merlresearch/LIP4RobotInverseDynamics) |
| Liu, Borja, Della Santina. *PINNs to Model and Control Robots: A Theoretical and Experimental Investigation* | Advanced Intelligent Systems 6(5):2300385, 2024 · [arXiv:2305.05375](https://arxiv.org/abs/2305.05375) |

Nguyen-Tuong & Peters (ICRA 2010) is the founding semi-parametric result, and
it makes an argument that cuts against this package's design: the analytical
model should enter as the **prior/mean of the model**, not as a penalty on the
output. Reuss et al. is the strongest modern hardware demonstration — a rigid-
body prior plus a learned network for stick-slip friction and flexibility,
trained end-to-end and deployed at 1 kHz.

Giacomuzzo et al.'s LIP kernel is a third way: put the physics in the
*hypothesis space* (GPs modeling kinetic and potential energy) rather than in
the loss or the architecture.

Liu, Borja & Della Santina is the closest paper to this whole package's
framing, and the one to read first — it extends PINNs to non-conservative
robot dynamics and proves stability bounds when the learned model is dropped
into a model-based controller.

### Exact RBD — the baseline the surrogate has to beat

| Work | Reference |
|---|---|
| Carpentier & Mansard. *Analytical Derivatives of Rigid Body Dynamics Algorithms* | RSS 2018 · [PDF](https://www.roboticsproceedings.org/rss14/p38.pdf) · [HAL:hal-01790971](https://hal.science/hal-01790971) |
| Sutanto, Wang, Lin, Mukadam, Sukhatme, Rai, Meier. *Encoding Physical Constraints in Differentiable Newton-Euler Algorithm* | L4DC 2020, PMLR 120:804–813 · [arXiv:2001.08861](https://arxiv.org/abs/2001.08861) |
| Lutter, Silberbauer, Watson, Peters. *A Differentiable Newton–Euler Algorithm for Real-World Robotics* | [arXiv:2110.12422](https://arxiv.org/abs/2110.12422) · DOI [10.1007/978-3-031-37832-4_2](https://doi.org/10.1007/978-3-031-37832-4_2) |
| Dalla Libera, Giacomuzzo, Carli, Nikovski, Romeres. *Forward Dynamics Estimation from Data-Driven Inverse Dynamics Learning* | IFAC World Congress 2023, pp. 519–524 · [arXiv:2307.05093](https://arxiv.org/abs/2307.05093) |
| Habich, Mohammad, Ehlers, Bensch, Seel, Schappler. *Generalizable and Fast Surrogates: MPC of Articulated Soft Robots Using PINNs* | IEEE T-RO, 2025 · [arXiv:2502.01916](https://arxiv.org/abs/2502.01916) |
| Yadav, Ujjawal, Sun, Roy, Pan. *Physics-Aware Sparse Learning and Selective Online Adaptation for Euler–Lagrange Robot Dynamics* | [arXiv:2606.09640](https://arxiv.org/abs/2606.09640) |

Sutanto et al. and Lutter et al. (DiffNEA) make the data-efficiency argument:
keep the exact O(n) recursion and learn only its ~10n inertial parameters,
under constraints that keep them physically plausible. Habich et al. is the
clearest quantified case *for* a learned surrogate — up to 466× faster than the
first-principles model, enabling nonlinear MPC — but note the win comes from a
genuinely expensive soft/continuum model, which a 7-DoF rigid serial arm is not.

---

## 3. Inverse kinematics

`inverse_kinematics.py` learns `q = h_θ(x)` with loss `mean(‖FK(h_θ(x)) − x‖²)`,
no labels, differentiating live through ManipulaPy's FK.

### This training scheme has a name, and it is from 1992

**Jordan & Rumelhart. *Forward Models: Supervised Learning with a Distal
Teacher.* Cognitive Science 16(3):307–354, 1992 · DOI
[10.1207/s15516709cog1603_1](https://doi.org/10.1207/s15516709cog1603_1).**

When the target is specified in a *distal* space (end-effector pose) rather
than in the learner's own output space (joint angles), you compose the learner
with a differentiable forward model and backpropagate the distal error through
it. Jordan & Rumelhart show this resolves the classical ill-posedness of
learning inverse mappings for redundant systems — which is, word for word, the
argument in this package's README and in
[`inverse_kinematics.py`](../src/manipulapy_pinn/inverse_kinematics.py)'s
docstring. This belongs in the module docstring, not just in a related-work list.

### Modern self-supervised IK

| Work | Reference | Note |
|---|---|---|
| Tenhumberg, Mielke, Bäuml. *Efficient Learning of Fast Inverse Kinematics with Collision Avoidance* | Humanoids 2023 · [arXiv:2311.05938](https://arxiv.org/abs/2311.05938) | Closest published system to this module. 19-DoF humanoid, <10 ms, 1e-4 m accuracy, ~10× less training time than a supervised equivalent. |
| Qu, Liu, Wu, Luo. *EMSSL: Embodied Self-Supervised Learning … for Robot Arm Inverse Kinematics* | ICDL 2023 · [arXiv:2302.13346](https://arxiv.org/abs/2302.13346) | Uses FK to *generate labels* rather than to provide gradients — so FK need not be differentiable. The useful contrast for stating what this package's live-autograd choice buys and costs. |
| Qu, Liu, Du, Luo. *CEMSSL: Conditional Embodied Self-Supervised Learning …* | [arXiv:2306.12718](https://arxiv.org/abs/2306.12718) | Same no-labels premise, but wraps conditional generative models (CVAE/cGAN/cINN) in the loop. Reports 2–3 orders of magnitude precision improvement while *preserving* solution diversity. |
| Habekost, Strahl, Allgeuer, Kerzel, Wermter. *CycleIK: Neuro-inspired Inverse Kinematics* | ICANN 2023 · [arXiv:2307.11554](https://arxiv.org/abs/2307.11554) · [code](https://github.com/jangerritha/CycleIK) | FK-consistency training on 8-DoF redundant arms, plus a hybrid neuro-genetic pipeline that post-optimizes with SLSQP or a GA — i.e. this package's proposed warm-start extension, already built. |
| *MimicIK: Real-Time Generative Inverse Kinematics from Teleoperation with FK Consistency* | [arXiv:2606.15148](https://arxiv.org/abs/2606.15148) | Uses the differentiable-FK term as a *regularizer alongside* a data term, not as the sole objective — useful for framing pure-FK-loss training as one end of a spectrum. |

### The multimodality problem this module does not solve

The FK-residual loss removes the *bad-target* problem. It does not give the
network any way to *represent* the solution manifold: `h_θ` still emits one
deterministic `q` per pose.

- **Bishop. *Mixture Density Networks.* Aston NCRG/94/004, 1994** ([PDF](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)). The original statement of the failure mode, and its motivating example is robot IK: a least-squares network collapses to the invalid *average* of elbow-up and elbow-down.
- Ames, Morgan, Konidaris. *IKFlow: Generating Diverse Inverse Kinematics Solutions.* RA-L 7(3):7177–7184, 2022 · [arXiv:2111.08933](https://arxiv.org/abs/2111.08933). Conditional normalizing flow over the whole solution manifold; 2000 solutions in <10 ms at ~10 mm / 2°.
- Bensadoun, Gur, Blau, Shenkar, Wolf. *Neural Inverse Kinematics.* ICML 2022, PMLR 162 · [arXiv:2205.10837](https://arxiv.org/abs/2205.10837). Captures one-to-many ambiguity *while training on one-to-one data* — solving this package's stated problem through architecture rather than loss.
- Limoyo, Marić, Giamou, Alexson, Petrović, Kelly. *Generative Graphical Inverse Kinematics.* IEEE T-RO 41:1002–1018, 2025 · [arXiv:2209.08812](https://arxiv.org/abs/2209.08812). Distance-geometric graph representation; one trained model serves *multiple robots*.
- Zhang & Jiao. *IKDiffuser.* [arXiv:2506.13087](https://arxiv.org/abs/2506.13087). Conditional diffusion over configuration space for arbitrary kinematic trees.
- Park, Schwartz, Park. *NODE IK.* ICCAS 2022 · [arXiv:2209.00498](https://arxiv.org/abs/2209.00498). IKFlow-comparable accuracy with 93% fewer parameters, slower inference — a clean statement of the speed/accuracy/memory trilemma.

### Warm-starting a classical solver — the proposed extension, already validated

**Rudrasamudram & Malaichamee. *Singularity Avoidance in Inverse Kinematics: A
Unified Treatment of Classical and Learning-based Methods.*
[arXiv:2604.13405](https://arxiv.org/abs/2604.13405)** benchmarks 12 IK solvers
on a **Franka Panda, position-only** — this package's exact setting. Its
numbers settle two of this package's open questions at once:

- A plain MLP achieves **0% success and ~10 mm mean error**. The README's
  "less accurate than the classical solvers, by design" is therefore the
  *expected, published* result, and should be reported as a reproduction
  rather than as a finding.
- Under DLS refinement: **IKFlow 59% → 100%, CycleIK 0% → 98.6%, GGIK 0% →
  100%**, with DLS converging from initial errors as large as **207 mm**.

The last figure is the important one: **the network does not need to be
accurate to be a useful seed.** The warm-start extension should therefore be
evaluated on *iterations saved and success-rate lift*, not on the network's own
pose error.

Supporting work: Morgan, Ames, Konidaris et al., *CppFlow*, ICRA 2024 ·
[arXiv:2309.09102](https://arxiv.org/abs/2309.09102) (learned proposal →
classical refinement, 129× faster); *Explainable Graph Neural Networks Towards
Data-Driven Inverse Kinematics*, Electronics 15(14):3071, 2026 · DOI
[10.3390/electronics15143071](https://doi.org/10.3390/electronics15143071)
(explicitly positions learned IK as a warm-start initializer, not a
replacement); and Donti, Rolnick, Kolter, *DC3*, ICLR 2021 ·
[arXiv:2104.12225](https://arxiv.org/abs/2104.12225), which unrolls the
correction steps *inside* training so the network learns to emit seeds the
corrector can finish.

**Required ablation.** Yuan, Wan, Harada, *IKSel: Selecting Good Seed Joint
Values for Fast Numerical Inverse Kinematics Iterations*
([arXiv:2503.22234](https://arxiv.org/abs/2503.22234)) solves the same problem
with a KD-tree lookup over a precomputed table and no learning at all. Any
claim that the network earns its keep as a warm-start has to beat that.

### Classical baselines to compare against honestly

- Wampler. *Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods.* IEEE Trans. SMC 16(1), 1986; and Nakamura & Hanafusa. *Inverse kinematic solutions with singularity robustness…* ASME J. Dyn. Syst. Meas. Control 108:163–171, 1986. The two independent origins of DLS — cite both.
- Chiaverini, Siciliano, Egeland. *Overview of damped least-squares methods for inverse kinematics of robot manipulators.* J. Intell. Robot. Syst. · DOI [10.1007/BF01254007](https://doi.org/10.1007/BF01254007).
- Sugihara. *Solvability-Unconcerned Inverse Kinematics by the Levenberg–Marquardt Method.* IEEE T-RO 27(5):984–991, 2011 · DOI [10.1109/TRO.2011.2148230](https://doi.org/10.1109/TRO.2011.2148230). Adaptive damping; handles unreachable targets — which matters because a neural seed will sometimes propose one.
- Beeson & Ames. *TRAC-IK.* Humanoids 2015 · DOI [10.1109/HUMANOIDS.2015.7363472](https://doi.org/10.1109/HUMANOIDS.2015.7363472). The de-facto 7-DoF baseline.
- Starke, Hendrich, Krupke, Zhang. *Evolutionary multi-objective inverse kinematics (BioIK).* IROS 2017, pp. 6959–6966 · [code](https://github.com/sebastianstarke/BioIK).
- Kim, Yi, Choi, Ma, Goldberg, Kanazawa. *PyRoki: A Modular Toolkit for Robot Kinematic Optimization.* IROS 2025 · [arXiv:2505.03728](https://arxiv.org/abs/2505.03728). Reports 1.4–1.7× faster convergence to lower error than cuRobo. **This threatens the "faster than classical" claim** — batched GPU numerical IK erases much of a learned solver's speed advantage, so `scripts/benchmark.py` must state precisely which classical baseline it beat.

---

## 4. Trajectory optimization

`trajectory.py` represents `q_θ(t)` with a hard-constraint ansatz, differentiates
twice through `t`, and penalizes an EOM residual evaluated through a *trained
forward-dynamics surrogate*.

### The ansatz has a name and a generalization

`q_θ(t) = q_start + t(q_goal − q_start) + t(1−t)·NN_θ(t)` is precisely the
Lagaris et al. (1998) trial solution: a parameter-free BC-satisfying term plus
a network multiplied by a factor that vanishes on the boundary.

- Sukumar & Srivastava. *Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.* CMAME 389:114333, 2022 · [arXiv:2104.08426](https://arxiv.org/abs/2104.08426). The modern generalization — `t(1−t)` is the 1-D product distance function of this framework. The reference to reach for if BCs are ever needed at intermediate waypoints or on a non-unit time interval.
- Mortari. *The Theory of Functional Connections.* [arXiv:2105.08034](https://arxiv.org/abs/2105.08034) (survey); orig. *The Theory of Connections: Connecting Points*, Mathematics, 2017 · [arXiv:1702.06862](https://arxiv.org/abs/1702.06862). TFC analytically parameterizes the *entire family* of functions satisfying a set of linear constraints on a function **and/or its derivatives**. **This is the direct upgrade path for `boundary_velocity_weight`**: the zero-velocity endpoint condition can be folded into the ansatz exactly, deleting a soft penalty and a hand-tuned weight.

### PINNs for optimal control

| Work | Reference |
|---|---|
| Mowlavi & Nabi. *Optimal control of PDEs using physics-informed neural networks* | J. Comput. Phys. 473:111731, 2023 · [arXiv:2111.09880](https://arxiv.org/abs/2111.09880) |
| Barry-Straume, Sarshar, Popov, Sandu. *PINNs for PDE-Constrained Optimization and Control (Control PINNs)* | [arXiv:2205.03377](https://arxiv.org/abs/2205.03377) |
| Antonelo et al. *Physics-Informed Neural Nets for Control of Dynamical Systems (PINC)* | Neurocomputing, 2024 · [arXiv:2104.02556](https://arxiv.org/abs/2104.02556) |
| D'Ambrosio, Schiassi, Curti, Furfaro. *Pontryagin Neural Networks with Functional Interpolation* | Mathematics 9(9):996, 2021 |
| Tahimi & da Silva Junior. *Dual-Network PINNs for Optimal Control: A Reproducible Benchmark on the Mass–Spring–Damper System* | [arXiv:2606.15271](https://arxiv.org/abs/2606.15271) |

Mowlavi & Nabi is the closest methodological analogue to the `physics + effort`
loss here, and gives an honest head-to-head against mature adjoint-based
optimal control. Tahimi & da Silva Junior is the most on-point benchmark
available: same problem class, same exact-BC-ansatz idea, and a three-way
comparison against both single shooting and trapezoidal direct transcription,
in which the PINN reproduces the classical optimal cost to four significant
digits.

**PINC documents a failure mode this package should expect**: a monolithic
`q_θ(t)` degrades badly over long horizons. The fix — augmenting inputs with
the initial state and control so the model can be chained over segments — is
the structure to adopt if this ever needs to plan past a short start→goal move.

### Classical baselines, and what the PINN gives up

Kelly. *An Introduction to Trajectory Optimization: How to Do Your Own Direct
Collocation.* SIAM Review 59(4):849–904, 2017 · DOI
[10.1137/16M1062569](https://doi.org/10.1137/16M1062569).

The structural contrast is worth stating plainly in this package's own docs:
direct collocation discretizes into a sparse-Jacobian NLP with **explicit
algebraic bound constraints at every knot** — which is exactly why velocity and
torque limits are trivial there. The PINN replaces the knots with a continuous
`q_θ(t)` and the NLP with unconstrained SGD, buying mesh-freedom and
differentiability *at the cost of losing those bound constraints*. The
package's "Known limitation: no velocity/torque limits" is not an oversight in
the implementation; it is the structural price of the formulation.

Also relevant: Schulman et al., *TrajOpt*, IJRR 33(9):1251–1270, 2014 · DOI
[10.1177/0278364914528132](https://doi.org/10.1177/0278364914528132) — whose
straight-line initialization is precisely this package's ansatz at `NN = 0`;
and Ratliff, Zucker, Bagnell, Srinivasa, *CHOMP*, ICRA 2009, pp. 489–494, whose
covariant gradient is the classical answer to descending in *trajectory space*
rather than in raw waypoint coordinates (the smooth network basis here provides
implicitly what CHOMP builds in by hand).

### Fixing the missing velocity/torque limits

Ranked by directness:

1. **Kicki, Liu, Tateo, Bou-Ammar, Walas, Skrzypczyński, Peters. *Fast Kinodynamic Planning on the Constraint Manifold with Deep Neural Networks.* IEEE T-RO 40:277–297, 2024 · [arXiv:2301.04330](https://arxiv.org/abs/2301.04330) · [code](https://github.com/pkicki/cnp-b).** The closest published system to what this package is trying to be: a whole-trajectory neural representation in a constraint-manifold formulation that *includes the system dynamics*, so outputs satisfy kinodynamic and actuation limits, produced in constant time. Both the mechanism to borrow and the baseline to benchmark against.
2. **D'Ambrosio, Benedikter, Furfaro. *Physics-Informed Pontryagin Neural Networks for Path-Constrained Optimal Control Problems.* J. Guidance, Control, and Dynamics 48(8), 2025 · DOI [10.2514/1.G008854](https://doi.org/10.2514/1.G008854).** Enforces state and control inequality constraints via Fischer–Burmeister complementary-slackness residuals — smooth, autograd-friendly, evaluated at collocation points. **Smallest code change of the options.**
3. **hPINN / AL-PINNs** (above, §1) — augmented Lagrangian rather than a fixed penalty, which also fixes the hand-tuned weighting.
4. **TrajOpt's outer-loop penalty escalation** — start with a hinge penalty on `|q̇| − q̇_max` and raise its coefficient across restarts until violations vanish. The pragmatic minimum, and close to what the README already proposes.
5. **TFC** (above) — orthogonal but cheap: delete the velocity-boundary penalty by construction.
6. **The cautionary result.** Tassa, Mansard, Todorov. *Control-Limited Differential Dynamic Programming.* ICRA 2014, pp. 1168–1175. Generalizes DDP to box control constraints via a box-QP backward pass, and demonstrates empirically that **the simple heuristics normally used to enforce limits — clamping and penalizing — are not efficient in general.** This is the evidence that option 4 is a starting point, not a destination.

Broader map: Drgoňa, Nghiem et al. *Safe Physics-Informed Machine Learning for
Dynamics and Control.* ACC 2025 tutorial ·
[arXiv:2504.12952](https://arxiv.org/abs/2504.12952) — covers Lyapunov and
control-barrier functions, output projections (hard interior bounds as an
architectural layer rather than a penalty), and NN verification.

### Physics-informed neural motion planning

- Ni & Qureshi. *NTFields: Neural Time Fields for Physics-Informed Robot Motion Planning.* ICLR 2023 (oral) · [arXiv:2210.00120](https://arxiv.org/abs/2210.00120) · [code](https://github.com/ruiqini/NTFields). Learns an arrival-time field by minimizing the Eikonal residual, with no expert trajectories at all — the same data-free bet this package makes, on a different PDE.
- Ni & Qureshi. *Physics-Informed Neural Motion Planning on Constraint Manifolds.* ICRA 2024 · [arXiv:2403.05765](https://arxiv.org/abs/2403.05765). Extends it to constraints holding *along the whole trajectory*, the structural counterpart to this package's endpoint-only hard constraints.

### Does surrogate dynamics inside a trajectory optimizer hold up?

The README is candid that the trajectory solve is "only as accurate as the
surrogate." The literature both sharpens and partially answers that worry.

*Cautionary:*

- Ai, Tian, Shi, Wang, Pfaff, Tan, Christensen, Su, Wu, Li. *A review of learning-based dynamics models for robotic manipulation.* Science Robotics, 2025 · DOI [10.1126/scirobotics.adt1497](https://doi.org/10.1126/scirobotics.adt1497). Names the open problems directly: prediction errors accumulate and degrade planning, and identifying the *reliable region* of a learned model remains unsolved.
- Janner, Fu, Zhang, Levine. *When to Trust Your Model: Model-Based Policy Optimization.* NeurIPS 2019. The canonical bound tying achievable performance to model error and rollout horizon. **A nuance worth stating explicitly in this package's docs:** because `q_θ(t)` is global in time and does not autoregress, surrogate error does *not* compound multiplicatively along the horizon the way it does in step-by-step rollouts. But it is applied at every collocation point, and — worse — the optimizer is free to *exploit* surrogate inaccuracies to fake a low residual. That is a different risk, not a smaller one.

*Encouraging:*

- Sukhija, Köhler, Zamora, Zimmermann, Curi, Krause, Coros. *Gradient-Based Trajectory Optimization with Learned Dynamics.* ICRA 2023 · [arXiv:2204.04558](https://arxiv.org/abs/2204.04558). Gradient-based trajectory optimization straight through a learned differentiable model, working on a Boston Dynamics Spot and an RC car from ~25 minutes of interaction data.
- Liu, Borja, Della Santina (2024, above) supplies stability bounds *as a function of the PINN's learning error* — the closest thing to a guarantee currently available for this design.

*Mitigation:* make the surrogate structurally Lagrangian (DeLaN, §2). A
black-box surrogate is unconstrained exactly where the optimizer will push it;
a structured one is not.

---

## 5. Differentiable simulation and the 250 ms Coriolis cost

[`physics.py`](../src/manipulapy_pinn/physics.py) documents the measurement
that drives this package's whole architecture: `velocity_quadratic_forces`
costs ~250 ms/call under the torch backend, so dynamics are precomputed in
NumPy and only FK (~3 ms) runs live.

**The literature says that cost is avoidable, and by a very large factor.**
Three paths, ranked by payoff:

### Path 1 — analytical derivatives instead of autodiff (~10⁵×)

- **Carpentier & Mansard. *Analytical Derivatives of Rigid Body Dynamics Algorithms.* RSS 2018 · [PDF](https://www.roboticsproceedings.org/rss14/p38.pdf).** Exact first-order partials of RNEA and ABA, linear in the number of bodies: **3 µs for a 7-DoF arm**, 17 µs for a 36-DoF humanoid. Implemented in Pinocchio.
- **Le Lidec, Montaut, de Mont-Marin, Schramm, Carpentier. *End-to-End and Highly-Efficient Differentiable Simulation for Robotics.* [arXiv:2409.07107](https://arxiv.org/abs/2409.07107).** **5 µs for a 7-DoF manipulator**, and states explicitly that this is "at least two orders of magnitude" better than automatic differentiation — i.e. the penalty is structural to the AD approach, not merely to torch's eager dispatch.
- Singh, Russell, Wensing. *Efficient Analytical Derivatives of Rigid-Body Dynamics using Spatial Vector Algebra.* RA-L 7(2):1776–1783, 2022 · [arXiv:2105.05102](https://arxiv.org/abs/2105.05102). Refines exactly the needed quantity, up to 1.4× over Pinocchio.

**The concrete opening:** `C(q,q̇)q̇` is RNEA evaluated with `a = 0, g = 0`, so
`∂(Cq̇)/∂q` and `∂(Cq̇)/∂q̇` are exactly what `pinocchio.computeRNEADerivatives`
returns. **3 µs versus 250 ms is roughly a 10⁵× gap.** This is the single
highest-value experiment to run before treating the offline-precompute design
as permanent.

### Path 2 — compilation and fusion (~3–5× at batch 1)

- **pytorch_kinematics** ([UM-ARM-Lab](https://github.com/UM-ARM-Lab/pytorch_kinematics)) publishes a `torch.compile` benchmark on a 7-DoF Kuka IIWA (CPU): batch 1, **0.21 ms → 0.04 ms (4.7×)**; batch 1024, 1.13 → 0.51 ms (2.2×). The speedup *shrinks* with batch size — the signature of dispatch overhead being eliminated rather than arithmetic, which corroborates this package's diagnosis exactly. It also uses an analytical geometric Jacobian for the backward pass instead of taping the FK graph, a pattern worth copying.
- Morton & Pavone. *frax: Fast Robot Kinematics and Dynamics in JAX.* [arXiv:2604.04310](https://arxiv.org/abs/2604.04310) · [code](https://github.com/StanfordASL/frax). A jitted JAX/XLA dynamics stack in pure Python reaching **low-microsecond CPU times for a single 7-DoF arm**, with `jax.jit` applied at the top-level call. Suggests ManipulaPy's *JAX* backend, jitted, might make live Coriolis differentiation affordable. Note its warning that at batch 1 the CPU beats the GPU.
- Hu, Anderson, Li, Sun, Carr, Ragan-Kelley, Durand. *DiffTaichi.* ICLR 2020 · [arXiv:1910.00935](https://arxiv.org/abs/1910.00935). The earliest clean statement of this pathology: expressing fine-grained physics as a graph of framework-level ops is ~188× slower than compiling the same math into fused kernels.

### Path 3 — batching to amortize dispatch (large, but needs loop restructuring)

- Wang, Xu, Wu, Qiu, Li. *Batched Differentiable Rigid Body Dynamics in PyTorch for GPU-Accelerated Robot Learning (BARD).* ICANN 2026 · [arXiv:2605.31481](https://arxiv.org/abs/2605.31481) · [code](https://github.com/YueWang996/bard-pytorch-dynamics). Featherstone RNEA/CRBA/ABA written for batched GPU autodiff; up to 64× FK and 63× Jacobian throughput at batch 4096. **Caveat worth heeding:** compilation reliably fuses kinematics but can *degrade* dynamics performance on smaller GPUs through register spilling. Measure, don't assume.
- Plancher, Neuman, Ghosal, Kuindersma, Reddi. *GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients.* ICRA 2022 · [arXiv:2109.06976](https://arxiv.org/abs/2109.06976).
- Freeman, Frey, Raichuk, Girgin, Mordatch, Bachem. *Brax.* NeurIPS 2021 D&B · [arXiv:2106.13281](https://arxiv.org/abs/2106.13281); and MuJoCo XLA (MJX) / MuJoCo Warp — the whole-step-in-one-compiled-XLA-graph design.

### Simulators and gradient mechanisms worth knowing

| Work | Reference | Mechanism |
|---|---|---|
| Werling, Omens, Lee, Exarchos, Liu. *Nimble* | RSS 2021 · [arXiv:2103.16021](https://arxiv.org/abs/2103.16021) | Analytical LCP gradients; 87× over finite differencing |
| Howell, Le Cleac'h, Brüdigam, Kolter, Schwager, Manchester. *Dojo* | [arXiv:2203.00806](https://arxiv.org/abs/2203.00806) | Implicit function theorem — a derivative without autodiffing the computation that produced the value |
| Heiden, Millard, Coumans, Sheng, Sukhatme. *NeuralSim / Tiny Differentiable Simulator* | ICRA 2021 · [arXiv:2011.04217](https://arxiv.org/abs/2011.04217) | **Closest architectural precedent to this package**: analytic articulated-body core + spliced-in learned residual |
| Geilinger, Hahn, Zehnder, Thomaszewski, Coros. *ADD* | ACM TOG 39(6), SIGGRAPH Asia 2020 · [arXiv:2007.00987](https://arxiv.org/abs/2007.00987) | Adjoint sensitivity — one gradient pass at ~one forward pass |
| NVIDIA **Warp** / **Newton** | [warp](https://github.com/NVIDIA/warp) · [newton](https://github.com/newton-physics/newton) | Discrete adjoint at *kernel* granularity, not op granularity |

Surveys: Newbury, Collins, He, Pan, Posner, Howard, Cosgun. *A Review of
Differentiable Simulators.* IEEE Access, 2024 ·
[arXiv:2407.05560](https://arxiv.org/abs/2407.05560); and Le Lidec, Jallet,
Montaut, Laptev, Schmid, Carpentier. *Contact Models in Robotics: a Comparative
Analysis.* IEEE T-RO 40:3716–3733, 2024 ·
[arXiv:2304.06372](https://arxiv.org/abs/2304.06372).

### The argument *for* the current design

It is not one-sided. **Suh, Simchowitz, Zhang, Tedrake. *Do Differentiable
Simulators Give Better Policy Gradients?* ICML 2022 (Outstanding Paper) ·
[arXiv:2202.00817](https://arxiv.org/abs/2202.00817)** shows first-order
gradients from differentiable simulators are *not* unconditionally better than
zeroth-order ones — stiffness and discontinuity can make them high-variance or
biased. And Zhong, Han, Brikis, *Do Differentiable Simulations with Contacts
Have Correct Gradients…?* ([arXiv:2207.05060](https://arxiv.org/abs/2207.05060))
shows simulator gradients can be silently *wrong*. "More exact gradients
through physics" is not automatically worth its compute — but the 250 ms figure
reflects op-level eager AD specifically, not the intrinsic cost of the
derivative, and paths 1 and 2 are cheap enough to measure before the design is
locked in.

---

## 6. Is a PINN worth it at all?

This section exists because `scripts/benchmark.py` makes comparative claims,
and the literature has a well-documented problem with exactly those claims.

- **Grossmann, Komorowska, Latz, Schönlieb. *Can physics-informed neural networks beat the finite element method?* IMA J. Applied Mathematics 89(1):143–174, 2024 · [arXiv:2302.04107](https://arxiv.org/abs/2302.04107).** A careful, compute-controlled head-to-head. FEM dominates on low-dimensional forward problems in both accuracy and time; PINNs become competitive in higher dimensions and — importantly — on **inverse problems and data assimilation**, where classical solvers have no direct route.
- **McGreivy & Hakim. *Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations.* Nature Machine Intelligence 6(10):1256–1269, 2024 · [arXiv:2407.07218](https://arxiv.org/abs/2407.07218).** Of articles claiming to outperform a standard numerical method, **79% (60/76) compare against a weak baseline**, with documented outcome-reporting and publication bias on top. This is the standard `scripts/benchmark.py` should be held to: a *properly tuned* classical baseline, and negative results reported.
- **Chuang & Barba. *Experience report of physics-informed neural networks in fluid simulations: pitfalls and frustration.* SciPy 2022 · [arXiv:2205.14249](https://arxiv.org/abs/2205.14249).** The explicit "not-so-successful story": ~**32 hours of PINN training** to match a 16×16 finite-difference simulation that ran in **under 20 seconds**. Companion — *Predictive Limitations of PINNs in Vortex Shedding* ([arXiv:2306.00230](https://arxiv.org/abs/2306.00230)) — shows data-driven PINNs exhibit the dynamics *only while training data is available*, reverting to steady state the moment it stops. A pointed warning about extrapolation.
- **Karnakov, Litvinov, Koumoutsakos. *Solving inverse problems in physics by optimizing a discrete loss (ODIL).* PNAS Nexus 3(1):pgae005, 2024 · DOI [10.1093/pnasnexus/pgae005](https://doi.org/10.1093/pnasnexus/pgae005) · [code](https://github.com/cselab/odil).** Keeps the physics-residual loss and *discards the neural network*, using conventional discrete approximations instead — reporting **five orders of magnitude** speedup at equal or better accuracy, because the discrete loss is sparse and far better conditioned. The sharpest "is the network earning its keep?" challenge in the literature.
- Hao et al. *PINNacle: A Comprehensive Benchmark of PINNs for Solving PDEs.* NeurIPS 2024 D&B · [arXiv:2306.08827](https://arxiv.org/abs/2306.08827). Independently corroborates that loss reweighting is one of the two highest-leverage interventions.

**The honest framing this package should adopt.** A PINN should not be claimed
to beat a classical solver at forward integration of *known* dynamics — which
is what `forward_dynamics.py` does. The defensible claims are: amortized
inference (pay training once, query forever), differentiability of the
resulting model, and use as a component in a larger differentiable pipeline.
The README's Performance section already gestures at this ("that cost amortizes
across every query made after training; a single one-off query is almost always
cheaper with ManipulaPy's classical solvers directly") — the literature says
that framing is correct and should be the headline, not the caveat.

---

## What the literature says we got wrong

Collected here so it is hard to skip.

### 1. The forward-dynamics physics term is mathematically redundant

This is the sharpest finding, and it is checkable from the code rather than
from any paper.

`generate_forward_dynamics_dataset` labels each sample with
`robot.dynamics.forward_dynamics(q, q̇, τ, gravity, F_ext=0)` — i.e. the exact
`q̈* ` solving `M q̈* = τ − C − g`. `precompute_dynamics_operators` then computes
`M, C, g` at the *same* states with the *same* gravity. So

```
residual = τ − (M q̈_pred + C + g)
         = M q̈* − M q̈_pred
         = −M (q̈_pred − q̈*)
```

and therefore

```
physics_loss = ‖M Δ‖²  where Δ = q̈_pred − q̈*  is exactly the data-loss error
data_loss    = ‖Δ‖²
```

**The physics loss is the data loss under a mass-matrix metric** — `ΔᵀMᵀMΔ`
instead of `ΔᵀΔ`. It has the same minimizer, the same zero set, and adds no
information the label did not already carry. It is not *useless*: reweighting
the error into torque space is arguably more physically meaningful, since it
penalizes errors on high-inertia directions more. But the README's claim that
it is "the network's own consistency with the *true* dynamics operator … not
just its distance to a stored q̈ label" does not survive the algebra — the two
are the same objective in different metrics.

This holds as long as the labels and the operators come from the same model
and the same gravity vector, which is the case here: both trace to the same
ManipulaPy robot with `F_ext = 0` and `gravity = [0, 0, −9.81]`.

**Verified numerically** on `panda` with ManipulaPy 1.4.1, 64 samples and an
arbitrary perturbed `q̈_pred` (not a trained network):

```
max |residual − (−M Δ)|                     = 2.6e-14   (residual scale: 1.1e+01)
data_loss    = ‖Δ‖²                         = 0.230886
physics_loss = ‖MΔ‖²                        = 8.842534
  same quantity recomputed as ‖−MΔ‖²        = 8.842534
residual at q̈_pred = q̈_true, max            = 2.8e-14   (same zero set)
```

The identity holds to machine precision. `physics_loss` is `data_loss` in a
mass-matrix metric — same minimizer, same zero set, no independent signal.

It also cannot capture friction, backlash, or motor dynamics, which is
precisely the part the semi-parametric literature (Reuss et al. RSS 2022;
Trinh, Geist et al., *Newtonian and Lagrangian Neural Networks*, IFAC ROBOTICS
2025 · [arXiv:2506.17994](https://arxiv.org/abs/2506.17994); Yadav et al.
[arXiv:2606.09640](https://arxiv.org/abs/2606.09640)) identifies as worth
learning. The physics term becomes genuinely informative the moment the labels
come from anything other than the same analytical model — real measurements, a
different simulator, or a partial/noisy dataset.

### 2. Learning forward dynamics directly may be the wrong parameterization

Dalla Libera, Giacomuzzo, Carli, Nikovski, Romeres. *Forward Dynamics
Estimation from Data-Driven Inverse Dynamics Learning.* IFAC World Congress
2023 · [arXiv:2307.05093](https://arxiv.org/abs/2307.05093) argues empirically
that one should learn an *inverse*-dynamics model, extract inertial and
gravitational components analytically, and compute forward dynamics in closed
form — tested against direct forward-dynamics learning on a simulated Panda and
UR10, with the indirect route winning.

### 3. Physics structure in the loss may be worth less than assumed

Gruver, Finzi, Stanton, Wilson. *Deconstructing the Inductive Biases of
Hamiltonian Neural Networks.* ICLR 2022 ·
[arXiv:2202.04836](https://arxiv.org/abs/2202.04836) shows HNN/LNN gains come
from *modeling acceleration directly and avoiding coordinate complexity*, not
from symplectic structure — and that relaxing the physics biases matches
performance on conservative systems and dramatically beats them on
non-conservative ones (friction, contact: i.e. real robots). Combined with
finding 1, much of what this package attributes to its physics residual may
already be delivered by the data term.

### 4. The IK module cannot represent the solution manifold it was designed for

The FK-residual loss correctly avoids supervising against an arbitrary
reference `q`. But `h_θ` is a deterministic map ℝ³ → ℝⁿ, so at a redundant arm
it must still commit to one solution per pose, and nothing in the loss controls
*which*. Bishop (1994) named this failure mode using robot IK as the example;
IKFlow, GGIK, CEMSSL and IKDiffuser all argue for distributional outputs — and
CEMSSL specifically shows you can keep the no-labels premise *and* get
multimodality.

### 5. Fixed collocation points, uniformly weighted in time

`trajectory.solve()` re-creates the identical `torch.linspace(0, 1, 24)` grid
every iteration. Daw et al. (ICML 2023) make this a correctness issue, not an
efficiency one; Andersen & Matsubara show the residual can be genuinely low *at*
the points and not between them; Wang et al. (CMAME 2024) add that uniform
temporal weighting is acausal for what is an initial-value problem.

### 6. The comparative claims need a stronger baseline

Per McGreivy & Hakim, and per PyRoki's numbers against cuRobo: "faster than
classical" must name *which* classical implementation, and it should be a tuned
one. A single-threaded CPU DLS loop is the weak-baseline pattern that paper
documents.

### 7. The 250 ms measurement is real, but the conclusion drawn from it is too strong

The measurement stands. What does not follow is that live dynamics
differentiation is impractical *in general* — it is impractical *under
op-level eager PyTorch autodiff*. Analytical RNEA derivatives put the same
quantity at ~3–5 µs (Carpentier & Mansard; Le Lidec et al.), and
`torch.compile` alone buys ~4.7× at batch 1 on a comparable 7-DoF arm.

---

## Ranked next steps

Ordered by (value ÷ effort), each traceable to a citation above.

| # | Change | Why | Reference |
|---|---|---|---|
| 1 | **Re-state the forward-dynamics physics term honestly**, or make it informative by sourcing labels from something other than the same analytical model (noise, a held-out simulator, real data) | The current term is `‖MΔ‖²` vs the data term's `‖Δ‖²` — same minimizer | Derivation above; §2 semi-parametric line |
| 2 | **Add an L-BFGS refinement phase after Adam** in all three `train`/`solve` loops | Cheapest known PINN accuracy win, no new hyperparameters | Rathore et al., ICML 2024 |
| 3 | **Resample collocation points** each iteration (Sobol/Hammersley first, RAD later) and **report the residual on a denser held-out grid** | Fixed grids make a low residual meaningless | Wu et al., CMAME 2023; Daw et al., ICML 2023 |
| 4 | **Benchmark `pinocchio.computeRNEADerivatives` against the 250 ms figure**; also try `torch.compile(fullgraph=True)` on the FK path | ~3 µs vs 250 ms would invalidate the precompute design entirely | Carpentier & Mansard, RSS 2018; pytorch_kinematics benchmark |
| 5 | **Fold the zero-velocity endpoint condition into the trajectory ansatz** (TFC constrained expression) | Deletes `boundary_velocity_weight` and a soft penalty | Mortari, TFC; Sukumar & Srivastava, CMAME 2022 |
| 6 | **Add velocity/torque limits** — hinge penalty with outer-loop escalation first, Fischer–Burmeister complementary slackness properly | Closes the README's headline known limitation | D'Ambrosio et al., JGCD 2025; Schulman et al., IJRR 2014; *cf.* Tassa et al., ICRA 2014 |
| 7 | **Replace fixed loss weights with ReLoBRaLo** (loss-values only, cheap), or an augmented Lagrangian | Three independent results say no constant weight is right | Bischof & Kraus; Liu et al. (ConFIG); Lu et al. (hPINN) |
| 8 | **Implement the IK warm-start**, and evaluate it on *iterations saved and success-rate lift*, ablated against a KD-tree seed lookup | Published gains are large (0% → 98.6%); the ablation is what makes it a real result | arXiv:2604.13405; Yuan et al. (IKSel); Morgan et al. (CppFlow) |
| 9 | **Name the classical baselines precisely** in `scripts/benchmark.py`, and compare against a tuned one (TRAC-IK, or PyRoki) | Avoids the weak-baseline pattern | McGreivy & Hakim, Nat. Mach. Intell. 2024; Beeson & Ames; Kim et al. |
| 10 | **Consider a DeLaN-structured surrogate** for the trajectory module's dynamics model | Removes the optimizer's ability to exploit non-physical regions of a black-box surrogate | Lutter et al., ICLR 2019; Gupta et al., L4DC 2020 |
| 11 | **Add causal temporal weighting** to the trajectory residual | It is an initial-value problem trained acausally | Wang, Sankaran, Perdikaris, CMAME 2024 |
| 12 | **Cite Jordan & Rumelhart (1992) in `inverse_kinematics.py`'s docstring** | The module's core argument is theirs, and naming it is more convincing than restating it | Jordan & Rumelhart, Cognitive Science 1992 |
