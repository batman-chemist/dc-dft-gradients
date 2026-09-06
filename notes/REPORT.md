# Analytic nuclear gradients for density-corrected DFT with D4 dispersion

**Complete technical record.** Method, derivation, implementation, validation,
measured results, and the bugs found along the way.

Reference implementation: <https://github.com/batman-chemist/dc-dft-gradients>

---

## 1. Summary

Analytic nuclear gradients were derived and implemented for

```
E_total(R) = E_KS[ D_HF(R) ] + E_D4(R)
```

a full Kohn–Sham energy functional evaluated non-self-consistently on a
converged Hartree–Fock density (HF-DFT / density-corrected DFT), plus D4
dispersion.

**Result: the gradient is exact.** Against a Richardson-extrapolated
finite-difference reference the analytic gradient agrees to **4×10⁻¹¹**, i.e.
limited by SCF convergence rather than by the formula. Thirty-two validation
cases pass — water monomer and hydrogen-bonded dimer, symmetric and
symmetry-broken, two basis sets, a GGA and a meta-GGA, with and without
dispersion.

**Why it is not trivial.** The density minimises `E_HF` while the energy being
differentiated is `E_KS`. The variational-stationarity argument that lets stock
gradient code skip the orbital response does not apply, and the surviving
response term turns out to be *larger than* the direct term. A gradient without
it is not approximate — it is not the derivative of any function.

Implemented in Python/PySCF as a validated reference for a subsequent Fortran
implementation in TURBOMOLE.

---

## 2. The method

```
E_KS[D] = Tr[D h] + ½ Tr[D J[D]] + E_xc[ρ_D] + E_nn
```

evaluated at `D = D_HF`. No `E_HF` term, no exact exchange — the functional's
own exchange is the only exchange present.

**Motivation.** The error of an approximate functional on its own
self-consistent density decomposes as

```
E_approx[n_approx] − E_exact[n_exact]
    = ( E_approx[n_approx] − E_approx[n_exact] )   density-driven error
    + ( E_approx[n_exact]  − E_exact[n_exact]  )   functional-driven error
```

Semilocal functionals suffer delocalisation/self-interaction error that
corrupts the density itself — worst for anions, radicals, stretched bonds and
transition states. HF densities err in the opposite direction (over-localised)
but are frequently closer to exact, so substituting them suppresses the
density-driven term while leaving the functional-driven term untouched. D4 is
added because semilocal functionals lack dispersion.

The structural consequence that drives everything below: **the density and the
energy functional come from different theories.**

---

## 3. Why the gradient is hard

### 3.1 The general problem

With atom-centred basis functions the energy depends on nuclear position twice
over — explicitly through integrals, and implicitly through parameters
optimised at that geometry:

```
dE/dR_A = (dE/dR_A)|_{D fixed}  +  Σ_μν (dE/dD_μν)(dD_μν/dR_A)
          └──── direct ────┘        └────── response ──────┘
```

### 3.2 Why variational methods escape the response term

For HF or self-consistent KS-DFT, build the orthonormality-constrained
Lagrangian

```
L = E[C] − Σ_pq ε_pq (CᵀSC − 1)_pq
```

The orbitals make `L` stationary, so every term proportional to `dC/dR`
multiplies zero. Only the constraint's own `R`-dependence survives, giving the
familiar overlap-derivative term:

```
dE_HF/dR_A = Tr[h⁽ᴬ⁾D] + (two-electron derivatives) − Tr[S⁽ᴬ⁾W] + dE_nn/dR_A
             W = 2 Σ_i ε_i C_i C_iᵀ
```

No CPHF, no `dD/dR`. **This works only because the energy being differentiated
is the one the orbitals minimised.**

### 3.3 Why it fails here

`D_HF` minimises `E_HF`; we differentiate `E_KS`. Hence `dE_KS/dC ≠ 0`, the
response term survives, and it cannot be compressed into an energy-weighted
density matrix. The same structural situation as MP2/CI/CC gradients: a
non-variational energy on a variational reference.

**Magnitude.** Measured on water/6-31G with r2SCAN:

```
‖direct‖   ≈ 0.21
‖response‖ ≈ 0.35        ← the response term is the larger one
```

### 3.4 The trap this creates

Feeding a foreign density to a stock DFT gradient routine — construct an `RKS`
object, inject HF orbitals, mark it converged, call `.Gradients()` — runs
without error or warning and returns plausible numbers:

```
naive shortcut : [ 0.0493,  0.1172, −0.0204, …]
ground truth   : [−0.0058,  0.0303, −0.0066, …]

error, naive   : 8.7×10⁻²      ← sign errors; larger than the gradient itself
error, correct : 9.0×10⁻⁶      ← finite-difference floor
```

`grad/rks.py` assumes stationarity: it applies the `−Tr[S⁽ᴬ⁾W]` shortcut and
omits the CPHF response entirely. **PySCF has no DC-DFT gradient support** — a
search of the package for `density-corrected` / `dc-dft` / `hf-dft` returns
nothing, and every gradient module is either self-consistent DFT or a
correlated-wavefunction method.

---

## 4. Derivation

```
dE_total/dR_A = (dE_KS/dR_A)|_{D fixed}  +  Tr[F_KS · dD_HF/dR_A]  +  dE_D4/dR_A
```

### 4.1 What multiplies the response

```
F_KS = dE_KS/dD = h + J[D_HF] + V_xc[D_HF]
```

The **full Kohn–Sham Fock matrix**, not merely `V_xc`. Had the energy been
`E_HF + E_xc[D_HF]`, HF's stationarity would have annihilated the `h + J − K/2`
part, leaving only `Tr[V_xc dD/dR]`. With `E_KS[D_HF]` there is no `E_HF` term
to be stationary, so every part of the functional feels the density response.

### 4.2 Direct term

Density matrix elements held fixed as numbers; differentiate the moving basis
functions, moving quadrature grid and moving nuclei:

```
(dE_KS/dR_A)|_{D fixed} = Tr[h⁽ᴬ⁾D] + Coulomb derivative
                          + (dE_xc/dR_A)|_{D fixed} + dE_nn/dR_A
```

Deliberately **no** overlap/energy-weighted-density term: that is a response
effect, handled explicitly in §4.3 rather than smuggled in through the
constraint.

The XC piece has **two** parts, because both the integrand and the quadrature
scheme depend on nuclear position:

1. **Basis-function derivative** — `ρ` is built from atom-centred Gaussians.
2. **Grid response** — the quadrature grid is itself atom-centred; each atom's
   points ride along with it, and Becke partition weights depend on *all*
   internuclear distances.

### 4.3 Response term: CPHF

Expand perturbed orbitals in the unperturbed set, `dC_p/dR_A = Σ_q U_qp C_q`.
Differentiating `CᵀSC = 1` gives

```
U_pq + U_qp = −S⁽ᴬ⁾_pq            (MO basis)
```

so the **symmetric part of `U` is fixed for free** by the overlap derivative;
for the occupied-occupied block, `U_ij = −½S⁽ᴬ⁾_ij`. Only the
occupied-virtual block requires solving:

```
(ε_a − ε_i) U_ai + Σ_jb A_{ai,jb} U_bj = −B_ai
```

### 4.4 A trap: the occupied-occupied block is not negligible

It is tempting to argue that occupied-occupied rotations leave `D` invariant,
since `D` is invariant under unitary rotations within the occupied space.
**This is false.** Unitary rotations have *antisymmetric* generators; this
block is `−½S⁽ᴬ⁾`, which is **symmetric** — a metric renormalisation forced by
the changing basis overlap, not a rotation.

Verified numerically (water/STO-3G, atom 0, y):

```
max| U_ij − (−½S⁽ᴬ⁾_ij) |                    = 2.5×10⁻¹⁶   identity exact
‖dD/dR‖                                      = 1.077
‖dD/dR from occupied-virtual block only‖     = 1.073
‖difference, i.e. occ-occ contribution‖      = 0.511       ← ~half the norm
```

This is why **nuclear-coordinate CPHF is not field-perturbation CPHF.** Under
an electric field the basis does not move, `S⁽ᴬ⁾ = 0`, the symmetric part
vanishes, and the simpler field-independent form is correct. Under nuclear
displacement none of that holds.

The correct right-hand side, read off PySCF's `solve_withs1` rather than
re-derived:

```
B_ai = (h1 − s1·ε_i)_ai + [fx(U_oo)]_ai ,      U_oo = −½S⁽ᴬ⁾_oo
```

The second term — the coupled response to the *fixed* occupied-occupied block
— is exactly what a naive derivation omits.

### 4.5 Z-vector reduction

Solving CPHF once per perturbation is `3·N_atom` solves. Since only a
contraction of the solution with a fixed Lagrangian `L = CᵀF_KS C` is ever
needed, and the CPHF operator `M = diag(ε_a − ε_i) + A` is self-adjoint:

```
Σ_ai L_ai U^A_ai = −Σ_ai z_ai B^A_ai ,        M z = L
```

One solve serves every perturbation. Equivalently: add HF's stationarity
conditions to the energy with multipliers `z` and choose `z` to restore
stationarity — manufacturing the variational property the expression lacked.
(Structurally identical to reverse-mode automatic differentiation.)

`fx` is *also* self-adjoint on this space (verified to 2.4×10⁻¹⁵), so the same
interchange applies a second time, moving the `fx` evaluation out of the
per-atom loop:

```
⟨z, fx(U_oo^A)⟩ = ⟨fx(z), U_oo^A⟩
```

### 4.6 Dispersion

In the standard D4 model the dispersion energy depends on geometry alone (EEQ
charges from geometry, not from the wavefunction), so it carries no density
response and contributes no CPHF. Its analytic gradient is taken from the
`dftd4` library.

---

## 5. Implementation

| Theory | Function in `src/nonscf_grad.py` |
|---|---|
| `E_KS[D]` §2 | `ks_energy_on_density` |
| `F_KS` §4.1 | `ks_fock_on_density` |
| direct term §4.2 | `direct_ks_grad` |
| XC part, both pieces §4.2 | `direct_exc_grad` |
| per-atom CPHF §4.3 | `hf_density_response` / `response_grad` |
| Z-vector §4.5 | `zvector_response_grad` |
| D4 §4.6 | `d4_energy_and_grad` |

The functional is a runtime parameter (any libxc string; GGA and meta-GGA both
work unchanged). **D4 damping parameters are derived from it** — see §7.3.

Both response paths are retained so either can validate the other.

---

## 6. Validation

### 6.1 Methodology

Analytic gradients are easy to get subtly wrong and easy to check. The
reference path shares **no intermediates** with the analytic path: at every
displaced geometry it re-converges HF from scratch and re-evaluates the whole
functional plus D4 on that geometry's own density.

Two methodological points earned the hard way:

- **Test asymmetric geometries.** Symmetry zeroes whole components and can mask
  a wrong term.
- **Validate terms in isolation, not just the total.** Both bugs in §7 were
  found by finite-differencing an individual term; a total can look plausible
  while two errors partially cancel.

### 6.2 The suite: 32/32 pass

Water monomer (C2v experimental, and a symmetry-broken variant) and water
dimer (S22 hydrogen-bonded geometry, O-O 2.910 A, and a symmetry-broken
variant) × STO-3G and 6-31G × PBE and r2SCAN × D4 on/off. Monomer results:

```
water (C2v)/sto-3g/pbe          3.649e-06     water (asym)/sto-3g/pbe      9.965e-06
water (C2v)/sto-3g/r2scan       3.704e-06     water (asym)/sto-3g/r2scan   9.959e-06
water (C2v)/6-31g/pbe           3.443e-06     water (asym)/6-31g/pbe       9.032e-06
water (C2v)/6-31g/r2scan        3.483e-06     water (asym)/6-31g/r2scan    9.030e-06
```

Reproduced bit-for-bit across **PySCF 2.13.1 vs 2.14.0, x86-64 vs arm64,
OpenBLAS vs Accelerate, 24 threads vs 1** — the result is not an artefact of
one machine.

### 6.3 The residual is the reference, not the gradient

Error tracks **geometry only** — not basis, not functional — which is the
signature of finite-difference truncation (dependent on the third derivative, a
property of the geometry). A fault in the XC or response code would key on
functional or basis instead.

Confirmed by step-size scaling:

| `h` (Bohr) | max\|analytic − FD\| | ratio |
|---|---|---|
| 2.0×10⁻² | 1.595×10⁻⁴ | — |
| 1.0×10⁻² | 3.986×10⁻⁵ | 4.00× |
| 5.0×10⁻³ | 9.965×10⁻⁶ | 4.00× |
| 2.5×10⁻³ | 2.491×10⁻⁶ | 4.00× |
| 1.25×10⁻³ | 6.228×10⁻⁷ | 4.00× |

Exact `h²` scaling over a 16× range. An error in the analytic gradient would
sit as a constant offset and refuse to shrink.

Richardson-extrapolating the reference to cancel the `h²` term:

```
vs plain FD (h = 5×10⁻³)     9.965×10⁻⁶
vs plain FD (h = 2.5×10⁻³)   2.491×10⁻⁶
vs Richardson-extrapolated   4.383×10⁻¹¹   ← true analytic error
```

Eleven digits: SCF convergence, not physics.

### 6.4 Z-vector agreement

```
per-atom : vs Richardson-extrapolated FD = 7.654×10⁻¹⁰
zvector  : vs Richardson-extrapolated FD = 7.247×10⁻¹¹
```

The Z-vector is *more* accurate, because conjugate gradient converges tighter
than the Krylov solver inside PySCF's `solve_mo1`.

---

## 7. Bugs found, and how

None were visible by inspection. All three were caught by numerical checking.

### 7.1 The direct XC term is two pieces

Only the grid-weight-response piece was included; the shell-sliced
`2·Tr[V_xc,A · D]` basis-derivative contraction was missing. **Failure mode:
the two pieces largely cancel**, so the result came out at ~10⁻⁷ — looking
like a legitimately near-zero term — while the true value was ~0.32.

### 7.2 Nuclear CPHF ≠ field-perturbation CPHF

The naive `B_ai = F⁽ᴬ⁾_ai − ε_i S⁽ᴬ⁾_ai` omits the occupied-occupied feedback
of §4.4. Error ~100%, with sign errors — yet the direct term and every
density-only sanity check still looked correct in isolation. Only
finite-differencing the true re-converged HF density response localised it.

### 7.3 A bug class that validation *cannot* catch

D4 damping parameters were initially hardcoded to PBE's, independent of the
functional:

```
xc=pbe     E_D4 = −0.0001950780
xc=r2scan  E_D4 = −0.0000466845
r2scan with PBE's parameters: −0.0001950780      ← ~4× too much dispersion
```

**Every gradient test still passes** with mismatched parameters, because D4's
analytic gradient is perfectly self-consistent with whatever parameters it was
handed. The gradient is right; the energy it is the gradient *of* is wrong.
The parameter set is now derived from the functional, and an XC string that
cannot be reduced to a single functional name raises rather than guessing.

### 7.4 A solver that silently under-converges

PySCF's `cphf.solve` stalls at ~10⁻⁶ on the Z-vector right-hand side
regardless of the tolerance requested. Building the operator explicitly showed
it symmetric to 1.3×10⁻¹⁵, positive definite (eigenvalues 0.36–21.6,
condition number ~60), with a **direct solve giving residual 1.6×10⁻¹⁷**.
Replaced with matrix-free conjugate gradient.

**Recurring lesson:** reuse validated machinery rather than re-deriving it —
but verify that it converges on *your* right-hand side.

---

## 8. Performance

### 8.1 Where the time goes

Water chains, 6-31G, r2SCAN:

| natm | nao | SCF | direct | Fock | CPHF | CPHF share |
|---|---|---|---|---|---|---|
| 3 | 13 | 0.08 | 0.28 | 0.11 | 0.02 | 3% |
| 6 | 26 | 0.09 | 0.73 | 0.29 | 0.14 | 12% |
| 9 | 39 | 0.24 | 1.71 | 0.54 | 0.58 | 19% |
| 12 | 52 | 0.34 | 5.98 | 1.65 | 2.65 | 25% |

CPHF grows ~`N^3.5` against the direct term's ~`N^2.2` — negligible at
validation scale, dominant at production scale (crossover ~20–30 atoms).

### 8.2 Z-vector: the honest number

The textbook framing "3N solves become 1" **overstates the benefit**. Measured
at 15 atoms:

```
make_h1 (derivative Fock)  1.58 s   ← needed by BOTH paths; irreducible
CG solve                   0.58 s
remainder                  0.47 s
                           ------
z-vector total             2.63 s   vs per-atom 3.97 s   →  1.5× overall
```

Two reasons: `make_h1` dominates and is common to both; and PySCF's per-atom
path already batches all perturbations into a single Krylov space, so it was
never `3N` independent solves. The solve component itself dropped ~4×.

The Z-vector remains correct to keep — its advantage widens with system size,
its memory advantage (one vector rather than `3N`) is unconditional, and
production Fortran needs it regardless.

### 8.3 Threading

PySCF is **OpenMP only**: many cores on one node, never across nodes. Letting
both PySCF and BLAS thread oversubscribes the cores. Pinning BLAS to one thread
inside PySCF's threads, on a 24-core node:

| | before | after | change |
|---|---|---|---|
| wall | 154.9 s | 77.6 s | **2.0× faster** |
| user | 1843 s | 804 s | 2.3× less |
| **sys** | **624 s** | **19 s** | **32× less** |

Identical results — the 624 s of system time was purely threads contending.

Parallel efficiency then saturates near 43% of 24 cores: these molecules are
too small to occupy that many. For bulk reference-data generation, a job array
of few-thread tasks scales far better than one wide job.

---

## 9. Status

**Done and validated:** closed-shell RHF reference; GGA and meta-GGA (`pbe`,
`r2scan`); D4 with functional-matched damping; Z-vector reduction;
finite-difference validation to 4×10⁻¹¹.

**Not done:**

- **Open shell (UHF/UKS)** — spin-resolved analogues needed throughout.
- **Scaling crossover** — where the Z-vector actually starts to pay is
  unmeasured beyond 15 atoms; `tests/compare_response.py` exists to settle it
  on cluster hardware.
- **Realistic systems** — water in small bases validates the *formula*, and
  says nothing about production basis-set quality or the dispersion-bound
  systems D4 exists for.
- **TURBOMOLE Fortran port.**

---

## 10. Path to TURBOMOLE

| Piece | Route |
|---|---|
| Energy `E_KS[D_HF]` | evaluate the functional on a supplied density — straightforward |
| Direct term | existing KS gradient quadrature on the frozen HF density, **omitting** the overlap/energy-weighted term |
| D4 | existing `disp`/`dftd4` interface, unchanged |
| **Response term** | **the actual work** |

**Reuse the existing CPHF/CPKS infrastructure** (analytic Hessians, MP2
gradients) — specifically its *nuclear-perturbation right-hand side*, not a
from-scratch field-independent form (§4.4, §7.2) — with `F_KS` built from the
HF density as the Z-vector Lagrangian in place of an MP2 Lagrangian.

**On the RPA parallel.** An existing RPA-on-PBE-density gradient path is the
same structural class, and anything written generically there is reusable. The
pieces swap favourably:

| | RPA@PBE | this method |
|---|---|---|
| Response equations | CP-**KS** | CP-**HF** |
| XC kernel `f_xc` in the response? | yes | **no** — just `2J − K` |
| Lagrangian | RPA amplitudes, frequency integration | `h + J + V_xc` |
| Two-particle density matrix | effectively yes | **no** — 1-RDM only |

Strictly cheaper on both axes than the RPA path.

**A warning that transfers.** Do not assume a stock DFT gradient routine fed a
foreign density gives a valid answer (§3.4). Check against finite differences
before trusting anything.

**A practical note on what to transfer.** The HF *density* is not sufficient:
`D` is a projector onto the occupied space (rank = `n_occ`), so it contains
neither the virtual orbitals nor the orbital energies that CPHF needs. The
converged SCF solution is required, not just its density.

---

## 11. Reproducing

```bash
git clone git@github.com:batman-chemist/dc-dft-gradients.git
cd dc-dft-gradients
python -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
cd tests && python validate.py            # 32-case suite, ~4 min
python compare_response.py 5 6-31g        # z-vector vs per-atom + scaling
```
