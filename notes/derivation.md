# Gradients for PBE-D4 evaluated on an HF density (HF-DFT / DC-DFT)

## Scheme being differentiated

Converge a standard HF wavefunction, then evaluate the **full PBE Kohn-Sham
energy functional** on that frozen HF density, plus D4 dispersion. The
density is never relaxed under the PBE potential — it is HF's own density:

    E_total(R) = E_PBE[ D_HF(R) ] + E_D4(R)

    E_PBE[D] = Tr[D h] + 1/2 Tr[D J[D]] + E_xc^PBE[rho_D] + E_nn

This is HF-DFT, a.k.a. density-corrected DFT (DC-DFT): the HF density stands
in for the self-consistent KS density to suppress density-driven error. Note
there is **no** `E_HF` term and **no** exact-exchange term — PBE exchange is
the only exchange present.

## Why this needs more than a stock gradient

Every standard analytic gradient in a quantum chemistry code (HF, KS-DFT)
leans on the density being *variationally stationary for the same energy
expression being differentiated*. That is what makes the orbital-response
(Pulay) force collapse into a simple overlap-derivative term via the
energy-weighted density matrix.

Here nothing is stationary. `D_HF` minimizes `E_HF`, but the energy being
differentiated is `E_PBE`. So when a nucleus moves, `D_HF(R)` responds
through the **HF** equations, while the energy responds through the **PBE**
functional — and the mismatch is a genuine, large gradient term. It cannot
be folded into an energy-weighted density matrix, and dropping it gives a
non-conservative (formally wrong) gradient.

How large: in the validated water tests below, the response term is *bigger
than* the direct term (~0.21 vs ~0.09 Hartree/Bohr on individual
components). This is not a small correction that can be neglected.

## Term-by-term breakdown

    dE_total/dR_A = dE_PBE/dR_A |_direct  +  dE_PBE/dR_A |_response  +  dE_D4/dR_A

Since `E_PBE` depends on `R` both explicitly (atom-centered basis functions
and quadrature grid move; nuclei move) and implicitly (through `D_HF(R)`),
the chain rule splits it exactly two ways:

    dE_PBE/dR_A = (dE_PBE/dR_A)|_{D fixed}          ... "direct"
                + Tr[ F_KS . dD_HF/dR_A ]           ... "response"

**1. Direct term** — the derivative through moving basis functions, grid and
nuclei, holding the density matrix elements fixed as numbers. Four pieces:
hcore (kinetic + nuclear attraction, including Hellmann-Feynman on the
nuclei), Coulomb, XC quadrature, and nuclear repulsion. All of these are
integrals a KS-DFT gradient code already computes. Deliberately **no**
overlap/energy-weighted-density (Pulay) term appears here — orbital response
is not being smuggled in through the back door; it is handled explicitly in
term 2.

**2. Response term** — `F_KS = dE_PBE/dD = h + J[D_HF] + V_xc[D_HF]`, the KS
Fock matrix built from the HF density, contracted with the CPHF response
`dD_HF/dR_A` of the HF density matrix to nuclear displacement.

Note this is the **full KS Fock matrix**, not just `V_xc`. (If the energy
expression had been `E_HF + E_xc^PBE`, HF's own stationarity would kill
everything except `V_xc` here. It isn't, so it doesn't.)

`dD_HF/dR_A` comes from the ordinary nuclear-coordinate CPHF equations —
the same ones a code solves for analytic Hessians.

**3. D4 term** — standard D4 analytic gradient, a function of geometry (and
EEQ charges derived from geometry, not from the wavefunction, in the standard
D4 model). Already solved by the `dftd4` library. Just add it on.

## Validation

[`nonscf_grad.py`](../src/nonscf_grad.py) builds each term;
[`validate.py`](../tests/validate.py) checks the sum against a brute-force
central-difference derivative of the *actual* `E_total(R)`, re-converging HF
independently and re-evaluating the whole PBE functional plus D4 at every
displaced geometry — no intermediates shared with the analytic code path.

Currently passing to ~1e-5 across 32 cases: water monomer (symmetric and
water geometries × STO-3G and 6-31G × PBE (GGA) and r2SCAN (meta-GGA) ×
with and without D4.

**The functional is a runtime parameter, not baked in.** `total_gradient(mol,
xc="r2scan")` works exactly as `xc="pbe"` does, with no code change — the
derivation never depended on the choice. The only functional-specific inputs
are `E_xc[D]`, `V_xc = dE_xc/dD`, and the XC quadrature gradient, all of
which the numerical-integration layer supplies per functional type
(including the τ-dependent terms a meta-GGA adds).

## Two implementation pitfalls, worth knowing before touching the Fortran

**The direct XC term is two pieces, not one.** A GGA gradient quadrature
routine (`get_vxc_full_response` in PySCF; the equivalent GGA gradient loop in
any code) yields a grid-weight-response piece and a
basis-function-derivative piece. Using only the grid-weight piece and
forgetting the shell-sliced `2·Tr[V_xc,A · D]` contraction (the standard
"nabla-on-bra, ×2 for the ket partner" convention) silently produces a
*near-zero* direct XC term rather than an obviously broken one — the two
pieces largely cancel. Easy to miss without a finite-difference check.

**Nuclear-coordinate CPHF is not field-independent CPHF.** A naive
`B_ai = F_ai^(R) − ε_i S_ai^(R)` — correct for field-type perturbations like
electric field/polarizability CPHF — is *missing* terms specific to
AO-following (nuclear) perturbations, where the occupied-occupied
orbital-rotation block is pinned by `−½S^(R)` rather than free, and feeds
back into the virtual-occupied equations. Using the naive right-hand side
gave a response term wrong by ~100%, while the direct term and every
density-only sanity check still looked fine in isolation. It only surfaced
by finite-differencing the true re-converged HF density response.

The current code sidesteps re-deriving this by reusing PySCF's own validated
nuclear-coordinate CPHF solver (`pyscf.hessian.rhf.solve_mo1`, the machinery
its analytic Hessians depend on) to get `dD_HF/dR_A` directly.

**Lesson for any port:** a mature code's existing CPHF/CPKS infrastructure
(analytic Hessians, MP2 gradients) already has the correct
nuclear-perturbation right-hand side. Reuse *that* construction rather than
re-deriving from the textbook field-independent CPHF formula.

## Status

Validated: direct term + CPHF response term + D4 gradient, against
finite differences, to ~1e-5 over the 32 cases listed above, for both a GGA
and a meta-GGA.

**Open items before this is production-ready:**

- **Z-vector reduction.** The response term currently solves CPHF once per
  atom (3·N_atoms solves). The Handy–Schäfer Z-vector trick collapses this
  to a *single* solve independent of N_atoms, via the interchange theorem
  ( Σ_ai U_ai(R) L_ai = Σ_ai z_ai B_ai(R) ), with `L = C^T F_KS C` in the
  occupied-virtual block as the Lagrangian. Mathematically equivalent, but
  essential for this to scale past toy systems.
- **D4 damping parameters** — *resolved*: the parameter set is now derived
  from the functional (`d4_method_for_xc`), and an XC string that cannot be
  reduced to a single functional name raises rather than guessing. This
  mattered: pairing r2SCAN with PBE's D4 parameters overestimates the
  dispersion energy by a factor of ~4 on the water test, and *no gradient
  check would catch it* — D4's own gradient is perfectly self-consistent
  with whatever parameters it was handed. Still worth confirming whether an
  HF-DFT/DC-DFT composite should use refit parameters rather than the
  stock ones for the functional.
- **Open-shell (UHF/UKS).** Closed-shell RHF only so far.
- **Larger, dispersion-relevant test cases.** Only water in small bases has
  been tested — enough to validate the formula, not to judge numerical
  behavior at production quality.

## Porting to a production code

- **Direct term** → adapt the existing KS gradient quadrature, called on the
  frozen HF density instead of a self-consistent KS density, with the
  overlap/energy-weighted-density term omitted.
- **Response term** → reuse the existing CPHF infrastructure (whatever drives
  analytic Hessians or MP2 gradients), specifically its nuclear-perturbation
  right-hand-side construction, with the KS Fock matrix `F_KS` built from the
  HF density as the Z-vector Lagrangian (in place of an MP2 Lagrangian).
- **D4** → link the existing dispersion interface, unchanged.

The response term is the only piece that is genuinely new work; the rest is
existing machinery called on a different density.
