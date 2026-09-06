"""
Analytic gradient for   E_total(R) = E_xc-functional[ D_HF(R) ] + E_D4(R)

i.e. a full Kohn-Sham energy functional evaluated on a converged
Hartree-Fock density (HF-DFT / density-corrected DFT), plus D4 dispersion.
The density is NOT relaxed under the KS potential; it is HF's own density.

    E_KS[D] = Tr[D h] + 1/2 Tr[D J[D]] + E_xc[rho_D] + E_nn

The functional is a free parameter everywhere: any libxc string PySCF
accepts works, GGA or meta-GGA. Validated for `pbe` and `r2scan`. The
derivation is functional-agnostic -- the only functional-specific inputs are
E_xc[D], V_xc = dE_xc/dD, and the XC quadrature gradient, all of which the
numerical-integration layer supplies per functional type.

See ../notes/theory.md for the derivation and ../notes/derivation.md for
implementation notes.
"""
import numpy
from pyscf import gto, scf, dft, lib
from pyscf.grad import rhf as rhf_grad  # noqa: F401  (registers RHF.Gradients)

DEFAULT_XC = "pbe"
DEFAULT_GRID_LEVEL = 5


# --------------------------------------------------------------------------
# functional / dispersion bookkeeping
# --------------------------------------------------------------------------

def d4_method_for_xc(xc):
    """
    Map a PySCF/libxc XC string onto the name of the matching D4 damping
    parameter set ('pbe,pbe' -> 'pbe', 'r2scan' -> 'r2scan').

    Raises rather than guessing when the string cannot be reduced to a single
    functional name (e.g. a custom hybrid or a mismatched exchange/correlation
    pair) -- pairing a functional with another functional's dispersion
    parameters is a silent correctness bug, so it must be made explicit.
    """
    key = xc.strip().lower().replace(" ", "")
    if "," in key:
        x_part, c_part = key.split(",", 1)
        if x_part != c_part:
            raise ValueError(
                f"cannot infer D4 damping parameters from xc={xc!r} "
                f"(exchange {x_part!r} != correlation {c_part!r}); "
                f"pass d4_method='...' explicitly"
            )
        key = x_part
    if not key:
        raise ValueError(f"cannot infer D4 damping parameters from xc={xc!r}")
    return key


def d4_energy_and_grad(mol, xc=None, d4_method=None, param=None):
    """
    D4 dispersion energy and its analytic gradient.

    The damping parameters must match the functional being evaluated.
    Precedence: an explicit `param` wins; else `d4_method` names the
    parameter set; else it is derived from `xc`. One of the three must be
    given -- there is deliberately no fallback default, since silently
    pairing a functional with the wrong dispersion parameters is a bug that
    would not show up in a gradient validation (D4's own gradient is
    self-consistent with whatever parameters it was handed).
    """
    from dftd4.interface import DispersionModel, DampingParam

    if param is None:
        if d4_method is None:
            if xc is None:
                raise ValueError("need one of param, d4_method, or xc")
            d4_method = d4_method_for_xc(xc)
        param = DampingParam(method=d4_method)

    numbers = numpy.array(mol.atom_charges())
    positions = mol.atom_coords()  # already in Bohr for a pyscf Mole
    disp = DispersionModel(numbers=numbers, positions=positions)
    res = disp.get_dispersion(param, grad=True)
    return res["energy"], res["gradient"]


# --------------------------------------------------------------------------
# reference wavefunction and energy-functional evaluation
# --------------------------------------------------------------------------

def run_hf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("HF did not converge")
    return mf


def build_grids(mol, grid_level=DEFAULT_GRID_LEVEL):
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    return grids


def exc_on_density(mol, dm, xc=DEFAULT_XC, grid_level=DEFAULT_GRID_LEVEL):
    """E_xc[dm] and the XC potential matrix V_xc, at fixed geometry/dm."""
    ni = dft.numint.NumInt()
    n, exc, vxc = ni.nr_rks(mol, build_grids(mol, grid_level), xc, dm)
    return exc, vxc


def ks_energy_on_density(mol, dm, xc=DEFAULT_XC,
                         grid_level=DEFAULT_GRID_LEVEL):
    """
    E_KS[dm]: the full KS-DFT energy functional evaluated on an arbitrary
    (here: HF) density matrix. Built explicitly rather than via
    RKS.energy_tot so the quadrature grid matches the gradient code exactly
    and pyscf's density-based grid pruning cannot introduce inconsistencies.
    """
    hcore = scf.hf.get_hcore(mol)
    vj = scf.hf.get_jk(mol, dm)[0]
    exc, _vxc = exc_on_density(mol, dm, xc, grid_level)
    e1 = numpy.einsum('ij,ij->', dm, hcore)
    ecoul = 0.5 * numpy.einsum('ij,ij->', dm, vj)
    return e1 + ecoul + exc + mol.energy_nuc()


def ks_fock_on_density(mol, dm, xc=DEFAULT_XC, grid_level=DEFAULT_GRID_LEVEL):
    """
    dE_KS/dD = h + J[D] + V_xc[D], the KS Fock matrix built from the HF
    density. This is what the density response gets contracted against --
    note it is the *full* KS Fock matrix, not just V_xc: because there is no
    separate E_HF term in this energy expression, nothing is stationary and
    every part of E_KS feels the HF density's response.
    """
    hcore = scf.hf.get_hcore(mol)
    vj = scf.hf.get_jk(mol, dm)[0]
    _exc, vxc = exc_on_density(mol, dm, xc, grid_level)
    return hcore + vj + vxc


# --------------------------------------------------------------------------
# direct term:  (dE_KS/dR)|_{D fixed}
# --------------------------------------------------------------------------

def direct_exc_grad(mol, dm, xc=DEFAULT_XC, grid_level=DEFAULT_GRID_LEVEL):
    """
    (d Exc/dR)|_{dm fixed}, atom-resolved. Sum of two pieces, combined
    exactly as pyscf's own RKS gradient code combines them in
    grad.rhf.grad_elec:

      1. the shell-sliced Vxc-matrix contraction, 2*Tr[Vxc_A . dm], keeping
         only bra AO rows belonging to atom A (the "nabla-on-bra, x2 for the
         ket partner" convention shared by every two-electron gradient term),
      2. the grid-weight-response ("Pulay grid") term, which is what
         `extra_force`/exc1_grid supplies in the real RKS gradient kernel.

    Both come from the *same* get_vxc_full_response call so the grid
    partitioning is self-consistent between them. Handles LDA/GGA/meta-GGA
    alike: for a meta-GGA the tau-dependent contributions are included by
    that routine's MGGA branch.
    """
    from pyscf.grad import rks as rks_grad

    ni = dft.numint.NumInt()
    excsum, vxc_mat = rks_grad.get_vxc_full_response(
        ni, mol, build_grids(mol, grid_level), xc, dm)

    aoslices = mol.aoslice_by_atom()
    g = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        g[ia] = 2 * numpy.einsum('xij,ij->x', vxc_mat[:, p0:p1], dm[p0:p1])
    g += excsum
    return g


def direct_ks_grad(mol, mf, dm, xc=DEFAULT_XC, grid_level=DEFAULT_GRID_LEVEL):
    """
    (d E_KS/dR)|_{D fixed}: the derivative of the whole KS energy functional
    through the moving basis functions / moving grid / moving nuclei, with
    the density matrix elements held fixed as numbers. Four pieces:

      hcore (kinetic + nuclear attraction, incl. Hellmann-Feynman on nuclei),
      Coulomb, XC (see direct_exc_grad), and nuclear repulsion.

    There is deliberately no overlap/energy-weighted-density (Pulay) term
    here: that term is an orbital-response effect, which this scheme handles
    explicitly through dD_HF/dR instead (see hf_density_response).
    """
    mf_grad = mf.Gradients()
    hcore_deriv = mf_grad.hcore_generator(mol)
    vj = mf_grad.get_j(mol, dm)
    aoslices = mol.aoslice_by_atom()

    g = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        g[ia] += numpy.einsum('xij,ij->x', hcore_deriv(ia), dm)
        g[ia] += 2 * numpy.einsum('xij,ij->x', vj[:, p0:p1], dm[p0:p1])

    g += direct_exc_grad(mol, dm, xc, grid_level)
    g += mf_grad.grad_nuc(mol)
    return g


# --------------------------------------------------------------------------
# response term:  Tr[ F_KS . dD_HF/dR ]
# --------------------------------------------------------------------------

def hf_density_response(mf):
    """
    Solve the ordinary nuclear-coordinate CPHF equations for the HF density's
    own response, dD_HF/dR_A for every atom A, by reusing PySCF's validated
    coupled-perturbed-HF machinery from the analytic-Hessian code
    (pyscf.hessian.rhf.solve_mo1 / Hessian.make_h1) rather than re-deriving
    the nuclear-perturbation right-hand side (which, unlike a
    field-independent/electric-field CPHF, carries extra overlap-derivative
    (S1) terms beyond a naive "F1_ai - eps_i*S1_ai"; getting those subtly
    wrong was a real bug caught only by finite-difference validation -- see
    notes/theory.md section 5.5).

    Independent of the XC functional: the reference is HF, so this is plain
    CPHF with no XC kernel, whatever functional is evaluated on top.

    Returns a list of (3, nao, nao) arrays, dD/dR_A for each atom A.

    NOTE: solves CPHF once per atom (3*natm total; cheap for small systems).
    The Handy-Schaefer Z-vector trick collapses this to a single solve
    independent of natm via the interchange theorem; that optimization is
    mathematically equivalent and is what the TURBOMOLE port should use, but
    is not needed to validate the formula.
    """
    from pyscf.hessian import rhf as rhf_hess

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nocc = int(numpy.sum(mo_occ > 0))
    mocc = mo_coeff[:, :nocc]

    hobj = mf.Hessian()
    h1ao = hobj.make_h1(mo_coeff, mo_occ)
    mo1s, _e1s = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao)

    dD = []
    for ia in range(mf.mol.natm):
        dC = mo1s[ia]  # (3, nao, nocc)
        dD_ia = 2 * (numpy.einsum('xpi,qi->xpq', dC, mocc)
                     + numpy.einsum('pi,xqi->xpq', mocc, dC))
        dD.append(dD_ia)
    return dD


def response_grad(mf, fock_ks):
    """
    dE_KS/dR_A|_{response} = Tr[ F_KS . dD_HF/dR_A ], with F_KS the KS Fock
    matrix built from the HF density and dD_HF/dR from CPHF.

    Reference implementation: solves CPHF once per atom. Correct but O(3*natm)
    solves. See zvector_response_grad for the production route.
    """
    dD_list = hf_density_response(mf)
    g = numpy.zeros((mf.mol.natm, 3))
    for ia in range(mf.mol.natm):
        g[ia] = numpy.einsum('ij,xij->x', fock_ks, dD_list[ia])
    return g


def _cphf_pieces(mf):
    """
    Per-atom ingredients of the nuclear-coordinate CPHF equations, in the MO
    basis with shape (3, nmo, nocc) each:

        h1mo[A]  derivative Fock matrix   (from pyscf's Hessian.make_h1)
        s1mo[A]  derivative overlap

    plus `fx`, the coupled-response operator (nmo,nocc) -> (nmo,nocc).

    Taken from pyscf's own hessian code so the conventions match the
    validated per-atom path exactly.
    """
    from pyscf.hessian import rhf as rhf_hess

    mol = mf.mol
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    nocc = int(numpy.sum(mo_occ > 0))
    mocc = mo_coeff[:, :nocc]

    h1ao = mf.Hessian().make_h1(mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    nao = mol.nao

    h1mo, s1mo = [], []
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        s1ao = numpy.zeros((3, nao, nao))
        s1ao[:, p0:p1] += s1a[:, p0:p1]
        s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
        s1mo.append(numpy.einsum('pq,xqr,ri->xpi', mo_coeff.T, s1ao, mocc))
        h1mo.append(numpy.einsum('pq,xqr,ri->xpi', mo_coeff.T,
                                 numpy.asarray(h1ao[ia]), mocc))

    fx = rhf_hess.gen_vind(mf, mo_coeff, mo_occ)
    return h1mo, s1mo, fx


def zvector_response_grad(mf, fock_ks, tol=1e-12, max_cycle=60):
    """
    Same quantity as response_grad, via the Handy-Schaefer Z-vector method:
    ONE linear solve regardless of the number of atoms, instead of 3*natm.

    Derivation. The response gradient is

        dE/dR_A|_resp = 4 * sum_{p,i} L_pi U^A_pi ,
        L = C^T F_KS C   (columns restricted to occupied)

    where U^A is the CPHF orbital-rotation matrix for perturbation A. Split it
    by block:

      * occupied-occupied: U^A_ji = -1/2 S1^A_ji, fixed in closed form by
        orbital orthonormality -- no solve needed. (Note this block is
        SYMMETRIC, a metric renormalization rather than a unitary rotation,
        so it genuinely changes the density; see notes/theory.md 5.5.)

      * virtual-occupied: obeys  M U^A = -B^A  with
            M      = diag(eps_a - eps_i) + A_coupled       (self-adjoint)
            B^A_ai = (h1 - s1*eps_i)_ai + [fx(U^A_oo)]_ai

        That last term -- the coupled response to the *fixed* occ-occ block --
        is the piece a naive field-perturbation CPHF right-hand side omits,
        and getting it wrong is a ~100% error, not a small one. It is read
        straight off pyscf's solve_withs1 rather than re-derived.

    Since M is self-adjoint, for the fixed L we can interchange:

        sum_ai L_ai U^A_ai = -sum_ai z_ai B^A_ai ,     M z = L

    so one solve for z serves every perturbation.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    mol = mf.mol
    mo_coeff, mo_energy, mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    nocc = int(numpy.sum(mo_occ > 0))
    mocc = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    nvir = orbv.shape[1]
    eps_o = mo_energy[:nocc]

    # Lagrangian L = C^T F_KS C, occupied columns
    L = mo_coeff.T @ fock_ks @ mocc          # (nmo, nocc)

    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    def M_apply(u):
        """
        The CPHF operator M = diag(eps_a - eps_i) + A_coupled restricted to
        the virtual-occupied block. Mirrors pyscf.hessian.rhf.gen_vind
        (including its *2 for double occupancy), projected onto vir-occ.
        """
        u = u.reshape(nvir, nocc)
        dm = orbv @ (u * 2) @ mocc.T
        v1 = vresp(dm + dm.T)
        return ((mo_energy[nocc:, None] - eps_o) * u
                + orbv.T @ v1 @ mocc).ravel()

    # --- the single solve:  M z = L.
    #
    # Solved with conjugate gradient rather than pyscf's cphf.solve: M is
    # symmetric (verified to 1e-15) and positive definite (the electronic
    # Hessian of a stable HF solution), which is exactly CG's case, whereas
    # pyscf's Krylov routine is tuned for the CPHF right-hand sides that
    # arise from nuclear perturbations and stalls around 1e-6 on this
    # Lagrangian regardless of the tolerance requested -- enough to degrade
    # the gradient by ~100x. CG is matrix-free, so this still never forms M.
    n = nvir * nocc
    op = LinearOperator((n, n), matvec=M_apply, dtype=float)
    z, info = cg(op, L[nocc:].ravel(), rtol=tol, atol=0.0, maxiter=max_cycle * 20)
    if info != 0:
        raise RuntimeError(f"Z-vector CG did not converge (info={info})")
    z = z.reshape(nvir, nocc)

    h1mo, s1mo, fx = _cphf_pieces(mf)
    nmo = mo_coeff.shape[1]

    # The per-atom right-hand side contains  fx(U_oo^A), i.e. one coupled
    # response (a J/K build) per perturbation -- 3*natm of them, which would
    # throw away most of the benefit of the single solve. But fx is
    # self-adjoint on this space (verified to 1e-15), so the interchange
    # trick applies a SECOND time:
    #
    #     <z, fx(U_oo^A)>  =  <fx(z), U_oo^A>
    #
    # Apply fx once to z, then every perturbation is a cheap contraction.
    z_full = numpy.zeros((nmo, nocc))
    z_full[nocc:] = z
    W = fx(z_full.reshape(1, nmo, nocc)).reshape(nmo, nocc)

    g = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        for x in range(3):
            s1 = s1mo[ia][x]                       # (nmo, nocc)
            # occupied-occupied block is fixed in closed form: U_oo = -1/2 S1_oo
            u_oo = -0.5 * s1[:nocc]

            # -<z, B^A> with B^A = (h1 - s1*eps)_vo + fx(U_oo^A)_vo,
            # the second piece evaluated via <fx(z), U_oo^A> instead.
            resp_vo = -(numpy.einsum('ai,ai->', z, (h1mo[ia][x] - s1 * eps_o)[nocc:])
                        + numpy.einsum('ji,ji->', W[:nocc], u_oo))
            resp_oo = numpy.einsum('ji,ji->', L[:nocc], u_oo)
            g[ia, x] = 4 * (resp_vo + resp_oo)
    return g


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def total_energy(mol, xc=DEFAULT_XC, with_d4=True, d4_method=None,
                 d4_param=None, grid_level=DEFAULT_GRID_LEVEL):
    """E_total = E_KS[D_HF] + E_D4, with an independent HF solve."""
    mf = run_hf(mol)
    e = ks_energy_on_density(mol, mf.make_rdm1(), xc, grid_level)
    if with_d4:
        e += d4_energy_and_grad(mol, xc=xc, d4_method=d4_method,
                                param=d4_param)[0]
    return e


def total_gradient(mol, xc=DEFAULT_XC, with_d4=True, d4_method=None,
                   d4_param=None, grid_level=DEFAULT_GRID_LEVEL,
                   response="zvector"):
    """
    Analytic gradient of E_total = E_KS[D_HF] + E_D4.

    Args:
        xc:         any libxc functional string PySCF accepts, GGA or
                    meta-GGA (e.g. 'pbe', 'r2scan').
        with_d4:    include the D4 dispersion term.
        d4_method:  D4 damping parameter set; defaults to the one matching
                    `xc` (see d4_method_for_xc).
        d4_param:   an explicit dftd4 DampingParam, overriding d4_method.
        grid_level: PySCF grid level. Meta-GGAs are more grid-sensitive than
                    GGAs; raise this if gradients look noisy.

    Returns a dict of the individual gradient pieces plus their sum (natm, 3),
    and the corresponding energies.
    """
    mf = run_hf(mol)
    dm = mf.make_rdm1()

    g_direct = direct_ks_grad(mol, mf, dm, xc, grid_level)
    fock_ks = ks_fock_on_density(mol, dm, xc, grid_level)
    if response == "zvector":
        g_response = zvector_response_grad(mf, fock_ks)
    elif response == "per-atom":
        g_response = response_grad(mf, fock_ks)
    else:
        raise ValueError(f"response must be 'zvector' or 'per-atom', got {response!r}")

    result = {
        "xc": xc,
        "e_ks": ks_energy_on_density(mol, dm, xc, grid_level),
        "g_direct": g_direct,
        "g_response": g_response,
    }

    g_total = g_direct + g_response
    e_total = result["e_ks"]
    if with_d4:
        e_d4, g_d4 = d4_energy_and_grad(mol, xc=xc, d4_method=d4_method,
                                        param=d4_param)
        result["e_d4"] = e_d4
        result["g_d4"] = g_d4
        g_total = g_total + g_d4
        e_total = e_total + e_d4

    result["g_total"] = g_total
    result["e_total"] = e_total
    return result
