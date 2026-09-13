# Theory: analytic gradients for PBE-D4 on an HF density

An explanation of what the code in `src/` does and why, starting from
why nuclear gradients are non-trivial at all.

Notation throughout:

```
  mu, nu, la, si   AO (basis function) indices
  i, j             occupied MO indices
  a, b             virtual MO indices
  p, q             any MO index
  R_A              coordinates of nucleus A
  X^(A)            derivative of quantity X w.r.t. R_A
  C                MO coefficient matrix,  D = 2 sum_i C_i C_i^T  (closed shell)
  S, h, J, K       overlap, core Hamiltonian, Coulomb, exchange matrices
```

---

## 1. The method being differentiated

```
  E_total(R) = E_PBE[ D_HF(R) ] + E_D4(R)

  E_PBE[D]   = Tr[D h] + 1/2 Tr[D J[D]] + E_xc^PBE[rho_D] + E_nn
```

Converge Hartree-Fock, take its density, and evaluate the **entire** PBE
Kohn-Sham energy functional on that density. No HF energy term survives, and
no exact exchange appears — PBE exchange is the only exchange.

**Why anyone does this.** The error of an approximate functional evaluated on
its own self-consistent density splits into two parts:

```
  E_approx[n_approx] - E_exact[n_exact]
      = ( E_approx[n_approx] - E_approx[n_exact] )   <- density-driven error
      + ( E_approx[n_exact]  - E_exact[n_exact]  )   <- functional-driven error
```

Semilocal functionals like PBE suffer from self-interaction/delocalization
error, which corrupts the *density* itself — most visibly for anions,
radicals, stretched bonds, transition states. HF densities have the opposite
bias (over-localized) but are frequently closer to exact. So substituting the
HF density kills the density-driven term while leaving the functional-driven
term alone. This is HF-DFT, or density-corrected DFT (DC-DFT). D4 is bolted
on because PBE has no dispersion.

The important structural consequence for us: **the density and the energy
functional come from different theories.**

**A note on PBE.** PBE is the running example throughout this document, but
nothing in the derivation depends on it. The functional enters only through
`E_xc[D]`, `V_xc = dE_xc/dD`, and the XC quadrature gradient — so any libxc
functional works, and the code takes it as a runtime argument
(`total_gradient(mol, xc="r2scan")`). Both a GGA (PBE) and a meta-GGA
(r2SCAN) are validated. A meta-GGA additionally depends on the kinetic energy
density `tau = 1/2 sum_i |grad phi_i|^2`, which adds terms to `V_xc` and to
the quadrature gradient, but `tau` is computable from the same one-particle
density matrix, so the structure of §5 is untouched.

---

## 2. Why nuclear gradients are subtle at all

We want `dE/dR_A` — the force on each nucleus — for geometry optimization
and dynamics.

The naive hope is the **Hellmann-Feynman theorem**: for an exact eigenstate,
`dE/dR = <psi| dH/dR |psi>`, so you only differentiate the Hamiltonian, not
the wavefunction. Convenient, and wrong in practice, for two reasons:

1. **We don't have exact eigenstates**, only variational approximations in a
   finite basis.
2. **The basis functions are glued to the nuclei.** Gaussians sit on atoms,
   so moving atom A changes the basis set itself. The energy depends on `R_A`
   both through the Hamiltonian *and* through the functions used to expand
   the wavefunction.

That second effect is the **Pulay force**. Any correct gradient with
atom-centered basis functions has to account for it.

Formally, write the energy as depending on `R` in two ways — explicitly
through integrals, and implicitly through whatever parameters (MO
coefficients, density matrix) were optimized at that geometry:

```
  dE/dR_A  =  (dE/dR_A)|_{D fixed}   +   sum_{mu,nu} (dE/dD_mu,nu) (dD_mu,nu/dR_A)
              \_______________/           \____________________________________/
                 "direct"                            "response"
```

Everything below is about those two terms.

---

## 3. The easy case: variational methods

For HF or self-consistent KS-DFT, the response term almost entirely vanishes,
and this is why stock gradient code is fast and simple.

The argument: build a Lagrangian enforcing orbital orthonormality,

```
  L = E[C] - sum_pq eps_pq ( C^T S C - 1 )_pq
```

HF/KS orbitals make `L` stationary with respect to `C`. So when we
differentiate the *total* energy, all the terms proportional to `dC/dR`
multiply something that is zero — except the piece coming from the
constraint, because the constraint itself is `R`-dependent (`S` changes as
atoms move). What survives is:

```
  dE_HF/dR_A =   Tr[ h^(A) D ]                        <- core Hamiltonian integrals
               + (two-electron integral derivatives)
               - Tr[ S^(A) W ]                        <- Pulay / constraint term
               + dE_nn/dR_A

  with  W = 2 sum_i eps_i C_i C_i^T                   <- energy-weighted density
```

**No CPHF. No `dD/dR` anywhere.** The orbital response is entirely absorbed
into that one overlap-derivative term. This only works because the energy
being differentiated is the *same* energy the orbitals minimized.

---

## 4. Our case: the stationarity argument collapses

`D_HF` minimizes `E_HF`. We are differentiating `E_PBE`. These are different
functionals, so:

```
  dE_PBE/dC  !=  0     at the HF solution
```

The response term no longer dies, and it cannot be compressed into an
energy-weighted density matrix. We need `dD_HF/dR_A` explicitly.

This is the same structural situation as MP2, CI, and coupled-cluster
gradients: a non-variational energy expression built on top of a variational
reference. The machinery below is the standard machinery for that class.

**How big is the term we would be dropping?** In the validated water tests,
the response term is *larger* than the direct term — roughly 0.21 vs 0.09
Hartree/Bohr on individual components. This is not a small correction. A
gradient without it is not "slightly approximate"; it is not the derivative
of anything, so geometry optimizations would converge to points that are not
stationary and dynamics would not conserve energy.

---

## 5. The derivation

### 5.1 Chain rule

```
  dE_total/dR_A  =  (dE_PBE/dR_A)|_{D fixed}       (direct)
                 +  Tr[ F_KS . dD_HF/dR_A ]        (response)
                 +  dE_D4/dR_A                     (dispersion)
```

### 5.2 What multiplies the density response

```
  F_KS = dE_PBE/dD = h + J[D_HF] + V_xc[D_HF]
```

This is the **full Kohn-Sham Fock matrix built from the HF density** — not
just `V_xc`.

This is worth dwelling on, because it is where the corrected functional
differs most from a superficially similar alternative. Had the energy been
`E_HF + E_xc^PBE[D_HF]`, then HF's own stationarity would have annihilated
the `h + J - K/2` part of the response, leaving only `Tr[V_xc dD/dR]`. With
`E_PBE[D_HF]` there is no `E_HF` term to be stationary, so *every* part of
the functional feels the density response.

### 5.3 The direct term

Hold the density matrix elements fixed **as numbers** and differentiate
everything else — moving basis functions, moving quadrature grid, moving
nuclei:

```
  (dE_PBE/dR_A)|_{D fixed} =   Tr[ h^(A) D ]              core Hamiltonian
                             + Coulomb integral derivative
                             + (dE_xc/dR_A)|_{D fixed}    XC quadrature
                             + dE_nn/dR_A                 nuclear repulsion
```

Every one of these is an integral derivative a KS-DFT gradient code already
computes. Note there is **no overlap/energy-weighted-density term here** —
that term is a *response* effect, and in this scheme response is handled
explicitly in 5.4 rather than smuggled in through the constraint.

**The XC piece has two parts, and this trips people up.** The XC energy is a
numerical quadrature,

```
  E_xc = sum_g  w_g  f( rho(r_g), grad rho(r_g) )
```

and *both* the integrand and the quadrature scheme depend on nuclear
positions:

1. **Basis-function derivative** — `rho` is built from atom-centered
   Gaussians, so moving atom A changes `rho` at every point.
2. **Grid response** — the quadrature grid is itself atom-centered. Atom A's
   grid points ride along with it, and the Becke partition weights `w_g`
   depend on *all* internuclear distances, so every weight in the molecule
   shifts when any atom moves.

Both must be included. See §9 for what happens if you include only one.

### 5.4 The response term: CPHF

`dD_HF/dR_A` comes from differentiating the HF equations themselves — the
coupled-perturbed Hartree-Fock (CPHF) equations. Expand the perturbed MO
coefficients in the unperturbed MOs:

```
  dC_p/dR_A = sum_q U_qp C_q
```

so the whole problem is finding the matrix `U`.

**Orthonormality fixes part of U for free.** Differentiating `C^T S C = 1`:

```
  U_pq + U_qp = - S^(A)_pq          (in the MO basis)
```

The *symmetric* part of `U` is therefore determined entirely by the overlap
derivative — no equation to solve. In particular, for the occupied-occupied
block the standard choice is

```
  U_ij = - (1/2) S^(A)_ij
```

**Only the occupied-virtual block requires real work**, and it obeys the
CPHF equation

```
  (eps_a - eps_i) U_ai  +  sum_jb A_{ai,jb} U_bj  =  - B_ai
```

where `A` is the coupled-HF orbital Hessian (the usual `4(ai|jb) - (ab|ij) -
(aj|ib)` for closed-shell RHF) and `B_ai` is a right-hand side built from
derivative integrals.

Finally the density response is assembled from `U`:

```
  dD/dR_A = 2 sum_i [ (dC_i/dR_A) C_i^T + C_i (dC_i/dR_A)^T ]
```

### 5.5 A trap in 5.4 worth stating explicitly

It is tempting to argue: *"the occupied-occupied block only rotates occupied
orbitals among themselves, and `D` is invariant under unitary rotations
within the occupied space, so it cannot affect `dD/dR`."*

**This is false, and it is an easy trap to fall into.**

`D` is invariant under *unitary* rotations, whose generator is
**antisymmetric**. But the occupied-occupied block here is `U_ij = -½S^(A)_ij`,
which is **symmetric**. It is not a rotation at all — it is a renormalization
forced by the fact that the basis metric `S` is changing as the atoms move.
Symmetric generators do change `D`.

Verified numerically for water/STO-3G (atom 0, y-direction):

```
  max| U_ij  -  ( -1/2 S^(A)_ij ) |            = 2.5e-16     (identity holds exactly)
  || dD/dR ||                                  = 1.077
  || dD/dR from occupied-virtual block only ||  = 1.073
  || difference, i.e. occ-occ contribution ||   = 0.511       <- about half the norm
```

So the occ-occ block carries roughly half the density response. Dropping it,
or building a right-hand side that ignores its feedback into the two-electron
terms, produces an answer that is wrong by order 100% — not by a little.

This is precisely why nuclear-coordinate CPHF is *not* the same as
field-perturbation CPHF (electric field, polarizabilities). Under an electric
field the basis functions do not move, `S^(A) = 0`, the symmetric part of `U`
vanishes, and the simpler "field-independent" form is correct. Under nuclear
displacement none of that holds.

---

## 6. The Z-vector method (what this *should* become)

Solving CPHF once per nuclear coordinate means `3*N_atoms` linear solves.
That is what the current code does, and it does not scale.

The **Handy-Schaefer Z-vector** trick reduces it to **one** solve, regardless
of system size. Two equivalent ways to see it:

**As linear algebra (interchange theorem).** Write the CPHF operator as
`M = diag(eps_a - eps_i) + A`, which is self-adjoint. The per-perturbation
solution is `U(A) = -M^-1 B(A)`. What we actually want is never `U` itself,
only its contraction with a fixed Lagrangian `L_ai = (C^T F_KS C)_ai`:

```
  sum_ai U_ai(A) L_ai  =  -( M^-1 B(A) )^T L
                       =  - B(A)^T ( M^-1 L )      <- M symmetric
                       =  sum_ai z_ai B_ai(A)      with  M z = -L
```

So solve `M z = -L` **once**, then contract the resulting `z` against each
perturbation's `B(A)` — and `B(A)` is built from ordinary derivative
integrals a gradient code already forms.

**As a Lagrangian.** Add the HF stationarity conditions to the energy with
multipliers `z`:

```
  L = E_PBE[D] + sum_ai z_ai F_ai[D] + (orthonormality constraints)
```

and choose `z` to make `L` stationary with respect to orbital rotations. Once
it is stationary, the §3 argument applies again: the response term disappears
and the gradient becomes a purely "direct" derivative of `L`. The Z-vector
equation is exactly the condition that makes this happen. In effect, we
*manufacture* the variational property that the original expression lacked.

This is the standard route for MP2/CC/CI gradients, and it is what a
production implementation should use. It is on the TODO list — mathematically
equivalent to what is implemented, just `3N` times cheaper.

---

## 7. D4 dispersion

Dispersion is the least interesting term here. In the standard D4 model the
dispersion energy is a function of geometry alone — pairwise `C6/R^6`-type
terms with coordination-number-dependent coefficients and EEQ charges derived
from geometry, *not* from the wavefunction. So it carries no density
response, contributes no CPHF, and its analytic gradient is already
implemented in the `dftd4` library. We call it and add the result.

One thing that does need care: **damping parameters must match the
functional.** The code derives them from the XC choice (`d4_method_for_xc`)
and refuses to guess when a string cannot be reduced to a single functional
name.

This is worth being strict about, because it is a bug class that validation
cannot catch. Pairing r2SCAN with PBE's D4 parameters overestimates the
dispersion energy by ~4x on the water test — yet every finite-difference
gradient check still passes, because D4's analytic gradient is perfectly
self-consistent with whatever parameters it was handed. The gradient is
right; the energy it is the gradient *of* is the wrong one.

Separately, HF-DFT/DC-DFT composites are sometimes refit, so confirm whether
stock functional parameters are what you want.

---

## 8. Theory-to-code map

| Theory | Function in `src/nonscf_grad.py` |
|---|---|
| `E_PBE[D]`, §1 | `ks_energy_on_density` |
| `F_KS = h + J + V_xc`, §5.2 | `ks_fock_on_density` |
| Direct term, §5.3 | `direct_ks_grad` |
| XC part of direct term (both pieces), §5.3 | `direct_exc_grad` |
| `dD_HF/dR_A` via CPHF, §5.4 | `hf_density_response` |
| `Tr[F_KS dD/dR]`, §5.2 | `response_grad` |
| D4, §7 | `d4_energy_and_grad` |
| Assembly | `total_gradient` |

`hf_density_response` delegates to PySCF's `pyscf.hessian.rhf.solve_mo1` —
the same nuclear-coordinate CPHF solver its analytic Hessians rely on —
rather than re-deriving the right-hand side, for the reasons in §5.5.

---

## 9. How we know it is right

Analytic gradients are unusually easy to get subtly wrong and unusually easy
to check: differentiate the energy numerically and compare.

```
  dE/dR_A,x  ~  [ E(R + h e_Ax) - E(R - h e_Ax) ] / 2h
```

`tests/validate.py` does this with **an entirely independent code path**: at
each displaced geometry it re-converges HF from scratch and re-evaluates the
whole PBE functional plus D4 on that geometry's own density. It shares no
intermediates with the analytic path — no reused Fock matrices, no reused
integrals, no reused grids.

Current status: **32/32 passing to ~1e-5** — water monomer (symmetric and
symmetry-broken) and water dimer (S22 hydrogen-bonded geometry and a
symmetry-broken variant), STO-3G and 6-31G, PBE (GGA) and r2SCAN (meta-GGA),
D4 on and off. The dimer is what genuinely exercises D4: its dispersion energy
is ~6x the monomer's. Central differences are
`O(h^2)`-accurate so ~1e-5 agreement at `h = 5e-3` is consistent with an
exact analytic gradient.

Two methodological notes:

- **Test asymmetric geometries.** A symmetric molecule zeroes out whole
  components by symmetry, which can mask a wrong term.
- **Validate terms in isolation, not just the total.** Both real bugs below
  were found by finite-differencing an *individual* term. A total gradient
  can look plausible while two errors partially cancel.

---

## 10. Two bugs, as case studies

Both were caught by finite differences, neither by inspection.

**Bug 1 — the direct XC term is two pieces, and only one was included.**
An early version took only the grid-weight-response piece, omitting the shell-sliced
`2*Tr[V_xc,A . D]` basis-derivative contraction. The failure mode was nasty:
the two pieces *largely cancel*, so the result was ~1e-7 — indistinguishable
from a legitimate near-zero term — while the true value was ~0.32. It looked
like a converged-to-zero physical result rather than a missing term. Finite
differences exposed it instantly.

**Bug 2 — nuclear CPHF is not field-independent CPHF.** The CPHF
right-hand side was first built as `B_ai = F_ai^(A) - eps_i S_ai^(A)`, which is the correct
form for field-type perturbations. For nuclear displacements it omits the
feedback of the constrained occupied-occupied block (§5.5) through the
two-electron terms. The response term came out wrong by ~100%, with the sign
wrong on some components — yet the direct term and every density-only sanity
check still looked fine in isolation. Only finite-differencing the *true
re-converged HF density response* pinned it down.

The fix in both cases was the same: stop re-deriving machinery that a mature
code already has correct, and call PySCF's own validated routines. **The same
advice applies to any production port** — a mature code's CPHF/CPKS
infrastructure (analytic Hessians, MP2 gradients) already contains the correct
nuclear-perturbation right-hand side. Reuse that construction; do not rebuild
it from the textbook field-independent formula.

---

## 11. What is not done yet

- **Z-vector reduction** (§6) — currently `3N` CPHF solves instead of 1.
- **Open shell** — closed-shell RHF only; UHF/UKS needs the spin-resolved
  analogues throughout.
- **D4 damping parameters** (§7) — now derived from the functional; still
  worth confirming whether a DC-DFT composite wants refit values.
- **Realistic test systems** — water in small bases validates the *formula*;
  it says nothing about behavior at production basis-set quality, or on the
  dispersion-bound systems D4 actually exists for.
