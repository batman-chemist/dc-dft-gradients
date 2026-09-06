# Analytic nuclear gradients for HF-DFT (density-corrected DFT) with D4

Reference implementation and finite-difference validation of analytic nuclear
gradients for

```
E_total(R) = E_KS[ D_HF(R) ] + E_D4(R)
```

that is, a **full Kohn-Sham energy functional evaluated on a converged
Hartree-Fock density**, plus D4 dispersion. The density is never relaxed under
the KS potential — it is HF's own. This is HF-DFT, also called
density-corrected DFT (DC-DFT).

Built with PySCF as a validated stepping stone toward a Fortran implementation
in TURBOMOLE.

## Why this is not a stock gradient

Every standard analytic gradient (HF, self-consistent KS-DFT) relies on the
density being **variationally stationary for the same energy expression being
differentiated**. That is what collapses the orbital-response (Pulay) force
into a single overlap-derivative term.

Here nothing is stationary: `D_HF` minimises `E_HF`, but the energy being
differentiated is `E_KS`. The orbital response survives, cannot be folded into
an energy-weighted density matrix, and must be obtained explicitly from the
coupled-perturbed Hartree-Fock (CPHF) equations.

**It is not a small correction.** In the validated tests the response term is
*larger* than the direct term. Omitting it does not give an approximate
gradient — it gives something that is not the derivative of any function, so
geometry optimisations converge to non-stationary points and dynamics does not
conserve energy.

Concretely, feeding a foreign density to a stock DFT gradient routine (PySCF's
`RKS.Gradients()` with HF orbitals injected) runs without complaint and returns
numbers that are wrong by ~0.087 Hartree/Bohr, **with sign errors** — an order
of magnitude larger than the true gradient components. There is no warning.

```
naive shortcut : 8.7e-02  error
this code      : 9.0e-06  error   (finite-difference floor, see below)
```

## Status

| | |
|---|---|
| Closed-shell RHF reference | done |
| GGA and meta-GGA functionals | done (`pbe`, `r2scan` validated) |
| D4 dispersion + gradient | done, damping parameters tied to the functional |
| Z-vector (one CPHF solve, not 3N) | done |
| Open shell (UHF/UKS) | **not implemented** |
| Fortran / TURBOMOLE port | not started |

Validated against brute-force finite differences: 16/16 cases pass
(symmetric + asymmetric geometries × STO-3G/6-31G × PBE/r2SCAN × D4 on/off).

Richardson-extrapolating the finite-difference reference to remove its own
`O(h²)` truncation error shows the analytic gradient is exact to **7e-11** —
i.e. limited by SCF convergence, not by the formula.

## Install

```bash
python -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
```

Use `python -m pip`, not bare `pip` — a shell alias shadowing the venv's pip is
an easy way to install into the wrong interpreter.

## Usage

```python
from pyscf import gto
import sys; sys.path.insert(0, "src")
import nonscf_grad as ng

mol = gto.M(atom="O 0 0 0; H 0.96 0.1 0.05; H -0.3 0.87 -0.2",
            basis="6-31g", unit="Angstrom")

res = ng.total_gradient(mol, xc="r2scan")
print(res["e_total"])     # E_KS[D_HF] + E_D4
print(res["g_total"])     # (natm, 3) analytic gradient
print(res["g_direct"], res["g_response"], res["g_d4"])   # the pieces
```

The functional is a runtime argument — any libxc string PySCF accepts, GGA or
meta-GGA. D4 damping parameters are derived from it automatically.

Useful keywords:

| keyword | default | meaning |
|---|---|---|
| `xc` | `"pbe"` | functional evaluated on the HF density |
| `with_d4` | `True` | include D4 dispersion |
| `d4_method` | from `xc` | override the D4 damping parameter set |
| `grid_level` | `5` | raise for meta-GGAs if gradients look noisy |
| `response` | `"zvector"` | or `"per-atom"` for the reference path |

## Testing

```bash
cd tests
python validate.py                    # 16-case finite-difference suite
python compare_response.py 5 6-31g    # z-vector vs per-atom CPHF, + scaling
```

### On a cluster

SLURM submit scripts are in `scripts/`. Put your site's environment setup in an
untracked `scripts/env.sh`, then submit with your own partition:

```bash
cd scripts
cp env.sh.example env.sh    # edit: conda activate / module load / venv
sbatch --partition=<your-partition> validate.slurm
sbatch --partition=<your-partition> compare_response.slurm
```

The scripts pin BLAS to one thread inside PySCF's OpenMP threads. Without that
the two layers oversubscribe the cores — measured at **2× wall time and 32×
system time** on a 24-core node, for bit-identical results.

Note that PySCF here is **OpenMP only**: it uses many cores on one node and
cannot span nodes, so requesting more than one node just leaves them idle.
These systems are also small enough that parallel efficiency saturates early
(~40% of 24 cores) — for bulk reference-data generation, a job array of
few-thread tasks scales far better than one wide job.

## Layout

```
src/nonscf_grad.py        implementation
tests/validate.py         finite-difference validation (the ground truth)
tests/compare_response.py z-vector vs per-atom cross-check and benchmark
scripts/*.slurm           cluster submit scripts
notes/REPORT.md           full technical record: derivation, results, bugs
notes/theory.md           the derivation, from first principles
notes/derivation.md       implementation notes and pitfalls
```

Start with [`notes/REPORT.md`](notes/REPORT.md) for the complete account —
method, derivation, every measured number, the bugs found and how, performance,
and the route to a Fortran implementation. `notes/theory.md` is the pedagogical
version of the derivation alone.

## Two traps worth knowing

Both were found only by finite-difference checking, not by inspection. They are
documented in full in [`notes/theory.md`](notes/theory.md).

**The direct XC term is two pieces.** A GGA/meta-GGA gradient quadrature yields
a grid-weight-response piece and a basis-function-derivative piece. Using only
the first silently gives a *near-zero* result rather than an obviously broken
one, because the two largely cancel.

**Nuclear-coordinate CPHF is not field-perturbation CPHF.** The naive
right-hand side `B_ai = F_ai⁽ᴬ⁾ − ε_i S_ai⁽ᴬ⁾`, correct for electric-field
perturbations, omits terms specific to a basis that moves with the nuclei: the
occupied-occupied rotation block is pinned at `−½S⁽ᴬ⁾` and feeds back into the
virtual-occupied equations. That block is *symmetric* (a metric
renormalisation, not a unitary rotation), so it genuinely changes the density —
it carries roughly half the response norm. Getting it wrong is a ~100% error.

## A note on the Z-vector speedup

The textbook framing is "3N solves become 1". Measured, the overall speedup is
more modest (~1.5× at 15 atoms) for two honest reasons:

- the derivative-Fock build (`make_h1`) is needed by both paths and is
  irreducible — it dominated at the sizes tested;
- PySCF's per-atom path already batches all perturbations into one Krylov
  space, so it was never 3N independent solves.

The solve component itself dropped ~4×. The advantage should widen with system
size, and the memory argument (one vector rather than 3N) is unconditional.
`tests/compare_response.py` is there to measure where the crossover actually
falls on real hardware.

## Requirements

Python 3.9+, PySCF (2.13/2.14 tested), NumPy, SciPy, and `dftd4` for the
dispersion term. Everything except D4 runs without `dftd4` (`with_d4=False`).

## License

MIT — see [LICENSE](LICENSE).
