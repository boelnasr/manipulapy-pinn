# Submitting ManipulaPy to an "awesome robotics" list

Prepared 2026-08-27. Everything below is copy-paste ready; the submissions
themselves have to be filed from an account with access to the target
repositories.

## Which list

| List | Curation status | Process | Verdict |
|---|---|---|---|
| [jslee02/awesome-robotics-libraries](https://github.com/jslee02/awesome-robotics-libraries) | active — entries carry 2026 last-commit metadata | **suggestion issue** — direct PRs are closed and redirected | **primary target** |
| [kiloreux/awesome-robotics](https://github.com/kiloreux/awesome-robotics) | last commit Sep 2024 | direct PR on `README.md` | secondary, low odds |
| [ahundt/awesome-robotics](https://github.com/ahundt/awesome-robotics) | no contributing guide, no recent activity | direct PR | skip |

`jslee02/awesome-robotics-libraries` is the only one that is genuinely
maintained, and it is the closest topical fit: its **Dynamics Simulation**
section is where pinocchio, RBDL, KDL, PyDy, Klampt and Robopy live.

Its `README.md` is generated from `data/*.yaml`, so it must not be edited by
hand, and `CONTRIBUTING.md` says new resources go through the
`suggest-resource.yml` issue template, not a pull request.

## Does ManipulaPy qualify

Must-haves (all four required) — all met: robotics-relevant, open source,
not a duplicate, has a real description.

Scoring rubric (needs ≥ 3 of 5):

| Criterion | Status |
|---|---|
| Popularity: ≥ 50 GitHub stars | ✗ — 23 stars (JOSS publication is the "equivalent adoption" argument) |
| Activity: a commit in the last 2 years | ✓ — v1.4.1 released 2026-08-14 |
| Documentation: README with usage examples or API docs | ✓ — [manipulapy.readthedocs.io](https://manipulapy.readthedocs.io/) |
| Maturity: ≥ 6 months old | ✓ — first PyPI release 2024-04-28 |
| Uniqueness: fills a gap | ✓ — see below |

4 of 5 → "✅ Accept" under their own stated rule.

The uniqueness argument, concretely: the Dynamics Simulation section has no
maintained general-purpose Python manipulator toolbox. Robopy (the Corke-style
Python toolbox listed there) is flagged 🔴 stale, pinocchio is C++ with
analytical derivatives, PyDy is symbolic multibody, and Brax/Newton are
general physics engines rather than manipulator libraries. ManipulaPy is a
maintained Python manipulator stack whose kinematics and dynamics run under
interchangeable NumPy, CuPy, PyTorch and JAX backends, so the same FK/mass
matrix calls are differentiable end-to-end.

Weak points to expect in review: 23 stars is under their popularity bar, and
AGPL-3.0-or-later is more restrictive than most entries in that section
(mostly MIT/BSD/Apache). Neither is disqualifying; disclose that you are the
author.

## Issue form answers — jslee02/awesome-robotics-libraries

Open <https://github.com/jslee02/awesome-robotics-libraries/issues/new?template=suggest-resource.yml> and fill in:

- **Resource Name**: `ManipulaPy`
- **URL**: `https://manipulapy.readthedocs.io/`
- **GitHub Repository**: `boelnasr/ManipulaPy`
- **Alternative Repository**: *(blank)*
- **Category**: `Dynamics Simulation`
- **Description**:

> Serial manipulator kinematics, dynamics, planning, and control in Python,
> with interchangeable NumPy, CuPy, PyTorch, and JAX backends. Covers forward
> and inverse kinematics (DLS/SQP/TRAC-IK), mass matrix, Coriolis and gravity
> terms, forward and inverse dynamics, quintic/cubic/LTS trajectory
> generation, PID/computed-torque/adaptive/robust control, manipulability and
> singularity analysis, and a dependency-free URDF parser. Ships 25 robot
> models and optional PyBullet simulation and stereo-vision modules.

- **Why should it be included?**:

> Disclosure: I am the author.
>
> The Dynamics Simulation section currently has no maintained
> general-purpose Python manipulator toolbox — Robopy is marked 🔴 stale,
> pinocchio is C++, and PyDy is symbolic multibody. ManipulaPy fills that
> slot and adds something none of the existing entries offer: the same
> kinematics and dynamics API runs on NumPy, CuPy, PyTorch, or JAX, so
> `forward_kinematics` and `mass_matrix` are differentiable and
> GPU-accelerated without a reimplementation. That makes it usable directly
> as the physics term in learning-based work.
>
> It is peer-reviewed and published in the Journal of Open Source Software
> (10(114):8490, https://doi.org/10.21105/joss.08490), has 15 PyPI releases
> since April 2024 (latest v1.4.1, 2026-08-14), full API docs at
> https://manipulapy.readthedocs.io/, and CI with coverage reporting.
>
> On the popularity criterion I would not claim the 50-star bar — the repo is
> at 23 stars. The JOSS review is the closest equivalent signal I can offer.

- **Must-have checklist**: check all four.
- **Scoring checklist**: check Activity, Documentation, Maturity, Uniqueness
  (leave Popularity unchecked — claiming it and being corrected costs more
  than conceding it).

### If a maintainer asks for the YAML

Add to `data/dynamics-simulation.yaml`, alphabetically between the
`LibrePilot` and `MARS` blocks (currently the `- name: MARS` line). Do not add
a `_meta:` block — CI generates `stars`, `last_commit` and `language`. Sign the
commit (`git commit --signoff`); their DCO check requires it.

```yaml
- name: ManipulaPy
  url: https://manipulapy.readthedocs.io/
  description: Serial manipulator kinematics, dynamics, planning, and control in Python with interchangeable NumPy, CuPy, PyTorch, and JAX backends.
  github: boelnasr/ManipulaPy
  license: AGPL-3.0-or-later
  languages:
  - Python
  models:
  - rigid
```

Field order and vocabulary follow the neighbouring entries: `models: rigid` is
the value used by the 25 other rigid-body entries in that file, and `languages`
uses bare `Python`.

Rendered, that becomes:

```
* 🟢 [ManipulaPy](https://manipulapy.readthedocs.io/) - Serial manipulator kinematics, dynamics, planning, and control in Python with interchangeable NumPy, CuPy, PyTorch, and JAX backends. [⭐ 23](https://github.com/boelnasr/ManipulaPy)
```

## Why Dynamics Simulation and not another section

Section names in that list are loose; what matters is what each one actually
contains.

- **Dynamics Simulation** — despite the name, this is the rigid-body
  kinematics-and-dynamics library bucket: pinocchio, KDL, RBDL, RBDyn, kindr,
  idyntree, PyDy, Klampt, Robopy. ManipulaPy is the same kind of object as
  those. **This is the fit.**
- **Inverse Kinematics** — a real secondary fit (ManipulaPy ships DLS, SQP and
  TRAC-IK solvers), and the section is thin: six entries, the smallest at 8
  stars. The issue form takes one category, so lead with Dynamics Simulation
  and offer this as a cross-listing in the "why" field. There is precedent —
  Bullet is listed under Dynamics Simulation and PyBullet under Simulators.
- **Motion Planning and Control** — poor fit. It is planners and optimal
  control (OMPL, MoveIt!, Crocoddyl, TOPP-RA, Ruckig). ManipulaPy's
  quintic/cubic time-scaling and PID/computed-torque controllers are features
  of a manipulator library, not the thing itself.
- **Robot Modeling** — no. That section is URDF/SDF formats plus model-authoring
  utilities (onshape-to-robot, phobos). The bundled URDF parser is not the
  library's identity.
- **Math** — no. Spatial algebra and Lie group libraries only (manif, Sophus,
  spatialmath-python).
- **Vision** — no. The optional stereo/YOLO module is not strong enough to
  stand as its own entry, and a thin second listing weakens the first.

## Secondary: kiloreux/awesome-robotics

Last commit September 2024, so a PR may sit indefinitely — but it costs one
edit. Fork, add to the **Software and Libraries** section in its existing
style (bold linked name, then a sentence), keep alphabetical placement:

```markdown
[**ManipulaPy**](https://github.com/boelnasr/ManipulaPy) A Python package for serial manipulator kinematics, dynamics, planning, and control, with NumPy, CuPy, PyTorch, and JAX backends.
```

PR title: `Add ManipulaPy to Software and Libraries`
PR body: one paragraph — what it does, the JOSS DOI, and an author disclosure.

## Before submitting either

Reviewers of a low-star entry look at the repo landing page first. Worth a
pass:

- README opens with what it is and a runnable 5-line example (it does).
- The GitHub repo's About blurb, topics, and website field are set to the
  readthedocs URL.
- CI badge is green on `main` at the moment you file.
