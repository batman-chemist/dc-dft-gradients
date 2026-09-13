# Limitations and TODO

An honest account of what this code does *not* do, what is not tested, and
what is known to be suboptimal. Written so that a future reader (or a future
version of the author) does not have to rediscover any of it.

The gradient formula itself is validated to 4×10⁻¹¹ (see
[`notes/REPORT.md`](notes/REPORT.md) §6). **Nothing below questions that
result.** Everything here is about coverage, robustness and engineering around
it.

## Priorities

| # | Item | Why it matters | Effort |
|---|---|---|---|
| 1 | Term-isolation tests are not in the repo | The tests that caught both real bugs exist only in scrollback | low |
| 2 | The energy is never validated | `E_KS[D_HF]` could be wrong and every gradient test would still pass | low |
| 3 | Open shell (UHF/UKS) not implemented | Excludes radicals — a main DC-DFT use case | high |
| 4 | Only H and O ever tested | No second-row element, no ECP, no diffuse basis | low |
| 5 | No hard-case validation | DC-DFT exists for anions/radicals/stretched bonds; none tested | medium |
| 6 | Grid built 3× per gradient | ~13% of runtime wasted | low |
| 7 | Z-vector crossover unmeasured past 15 atoms | Do not know where it actually pays off | low |
| 8 | No CI | Nothing catches regressions | low |

---

## 1. Not implemented

### Open shell (UHF/UKS)

Closed-shell RHF only. Every routine assumes doubly-occupied orbitals: the
density is built as `2·Σ_i C_i C_iᵀ`, the CPHF response carries the `*2` for
double occupancy, and the Z-vector operates on a single spin channel.

This is the most significant functional gap. **Radicals are one of the
principal cases DC-DFT is meant to fix**, and they are exactly what cannot be
run. Requires spin-resolved analogues throughout — `F_KS` per spin, UCPHF, and
a spin-summed response contraction.

### Periodic systems

Molecular only. Not planned.

---

## 2. Testing gaps

### 2.1 Term-isolation tests are missing — highest priority

`tests/validate.py` checks only the **total** gradient end-to-end. During
development, several checks were run interactively that are *not* in the
repository, and these are precisely the ones that caught the real bugs:

- finite-differencing the **direct term alone**, at fixed density matrix
  (caught bug 1 in REPORT §7.1)
- finite-differencing the **response term alone**, against the
  re-converged HF density response (caught bug 2, REPORT §7.2)
- verifying `U_oo = −½S⁽ᴬ⁾` to machine precision
- verifying the CPHF operator is symmetric and positive definite
- verifying `fx` is self-adjoint
- checking the Z-vector residual `‖Mz − L‖` (caught the stalled solver,
  REPORT §7.4)

A total gradient can look correct while two errors partially cancel — this is
not hypothetical, it is how bug 1 nearly escaped. **Without these, a future
refactor of `direct_exc_grad` or `zvector_response_grad` has no test that
localizes the failure.**

These should be ported into a `tests/test_terms.py`. They are cheap and
already written; they only need lifting out of the development history.

### 2.2 The energy is never validated

`tests/validate.py` never inspects `e_ks` or `e_total` — only gradients. Since
the finite-difference reference is built from the *same* `total_energy`
function the analytic gradient is compared against, **a systematic error in
the energy would cancel and go undetected**.

Suggested check: compare `ks_energy_on_density(mol, dm_scf)` against PySCF's
own `RKS.energy_tot()` for a *self-consistent* KS density, where the two must
agree exactly. That pins the energy expression independently.

### 2.3 No regression pinning

No reference values are stored. A change that shifts results slightly but
still passes the `1e-4` finite-difference threshold would go unnoticed.
Storing a few gradients to ~1e-10 would catch that.

---

## 3. Scientific validation gaps

### 3.1 Only water

Every test molecule is water — monomer (two geometries) or dimer (two
geometries). Consequences:

- **Only H and O.** No second-row element, no transition metal, no ECP.
- **No diffuse-heavy basis.** STO-3G and 6-31G in the suite; aug-cc-pVDZ has
  been run only ad hoc.
- **Closed-shell, neutral, equilibrium-ish.** Nothing strained.

This validates the *formula*. It says nothing about numerical robustness in
regimes with near-linear-dependent basis sets, sharp core densities, or
difficult SCF.

### 3.2 The hard cases — the ones DC-DFT exists for — are untested

DC-DFT's entire motivation is suppressing density-driven error, which is
largest for **anions, radicals, stretched bonds, transition states, and
halogen bonds**. None of these are tested. The water dimer is a well-behaved
closed-shell hydrogen bond where PBE's own density is already decent, so it
exercises the machinery but not the physics.

Worth noting a measured consequence: HF-PBE-D4 and HF-r2SCAN-D4 give
essentially identical water dimer interaction energies (−4.76 vs −4.77
kcal/mol at aug-cc-pVDZ). **The test set cannot currently distinguish the
functionals**, which means it also cannot detect a functional-specific error.

### 3.3 D4 damping parameters

Parameters are taken from the stock set matching the functional. HF-DFT/DC-DFT
composites are sometimes refit, and whether the stock values are appropriate
for a functional evaluated on an HF density has not been checked here.

Note this is a class of error **no gradient test can catch** (REPORT §7.3):
D4's analytic gradient is self-consistent with whatever parameters it is
handed. The gradient stays right; the energy it differentiates is wrong.

---

## 4. Performance and code quality

### 4.1 The quadrature grid is built three times per gradient

`total_gradient` triggers three independent `build_grids` calls — via
`direct_exc_grad`, `ks_fock_on_density`, and `ks_energy_on_density`. Measured
at ~13% of total gradient runtime (0.35 s per build on a 4-water chain,
360k points).

Fix: build once in `total_gradient` and thread the grid object through, the
way `grid_level` currently is.

### 4.2 `make_h1` dominates the response term

The derivative-Fock build is ~60% of the Z-vector path at 15 atoms and is
needed by both response implementations. It is the real bottleneck, not the
CPHF solve. Any future optimisation effort belongs here rather than on the
solve.

### 4.3 Unused imports

`gto` and `lib` are imported in `src/nonscf_grad.py` and never used.

### 4.4 Z-vector crossover unmeasured

Benchmarked only to 15 atoms on a single-threaded laptop, where the speedup is
~1.5×. Where it becomes decisive is unknown. `tests/compare_response.py` exists
to answer this and needs running on real hardware at larger sizes.

### 4.5 No density fitting

Conventional four-index integrals throughout. Fine for validation, limiting
for anything large.

---

## 5. Robustness / untested edge cases

None of the following are handled or tested; behaviour is unknown:

- **`mol.symmetry = True`** — PySCF may reorient the molecule, which would
  silently break the correspondence between analytic and finite-difference
  geometries.
- **Charged species** — should work, never run.
- **ECPs / effective core potentials** — the `hcore_generator` path is
  untested with them.
- **SCF non-convergence** — `run_hf` raises, but there is no fallback, no
  second-order solver, no alternative initial guess.
- **Linear dependence in the basis** — no canonical-orthogonalisation path.
- **Grid sensitivity** — meta-GGAs are more grid-sensitive than GGAs.
  `grid_level=5` is the default and was adequate for r2SCAN on water; no
  convergence study was done.

---

## 6. Infrastructure

- **No CI.** Nothing runs the suite automatically. The full suite is ~4
  minutes single-threaded, well within a free GitHub Actions runner.
- **No packaging.** Import works by `sys.path` manipulation rather than an
  installable package.
- **No API documentation** beyond docstrings.
- **Version sensitivity untracked.** Tested on PySCF 2.13.1 and 2.14.0. The
  code reaches into PySCF internals (`pyscf.grad.rks.get_vxc_full_response`,
  `pyscf.hessian.rhf.solve_mo1`, `gen_vind`), which are **not public API** and
  could change between releases without notice.

That last point deserves emphasis: several of the borrowed routines are
internal. A PySCF upgrade could break this code silently or loudly, and there
is no pinned upper bound in `requirements.txt`.

---

## 7. Toward TURBOMOLE

The Fortran port has not begun. See [`notes/REPORT.md`](notes/REPORT.md) §10
for the mapping of each term onto existing infrastructure, and the two traps
(§7.1, §7.2 there) that would otherwise have to be rediscovered.

The single most transferable warning: **do not assume a stock DFT gradient
routine fed a foreign density gives a valid answer.** It runs, it returns
plausible numbers, and it is wrong by more than the gradient itself.

### Scope boundary — the port does not belong in this repository

TURBOMOLE is licensed software. Its source, and material derived from reading
it, must not be published here. The port is developed separately.

The distinction is not only about code. Prose describing how its internals
work is derived material too:

| Publishable here | Stays out |
|---|---|
| The physics — derivation, CPHF structure, Z-vector; all textbook and literature | Source, in any quantity |
| This PySCF implementation | Subroutine signatures, common blocks, data structures read from the source |
| Program and module names from the published user manual | Source tree layout and file paths |
| Generic remarks such as "reuse the existing CPHF infrastructure" | Patches or diffs against it |
| | Notes describing how its internals actually behave |
| | Reference data shipped with the distribution |

The rows on the right stay private even when written from scratch: a
description of proprietary internals is derived from them regardless of who
typed it. The safe default, once working in that tree, is to treat everything
learned there as confidential unless it appears in the public manual.

The current contents of this repository were checked against this boundary and
contain nothing on the right-hand side.
