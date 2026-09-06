"""
Finite-difference validation of the analytic gradient in
../src/nonscf_grad.py for  E_total = E_PBE[D_HF] + E_D4
(HF-DFT / density-corrected DFT: full PBE KS functional on the HF density).

The reference path shares no intermediates with the analytic path: at every
displaced geometry it re-converges HF independently and re-evaluates the
whole PBE functional plus D4 on that geometry's own density.

Run: source ../venv/bin/activate && python3 validate.py
"""
import sys
import os
import numpy
from pyscf import gto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import nonscf_grad as ng  # noqa: E402

MOLECULES = {
    "water (C2v)": """
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
    """,
    "water (no symmetry)": """
        O   0.000000  0.000000  0.000000
        H   0.960000  0.100000  0.050000
        H  -0.300000  0.870000 -0.200000
    """,
}

BASES = ["sto-3g", "6-31g"]

# The derivation is functional-agnostic; both a GGA and a meta-GGA are
# checked, since meta-GGAs add a kinetic-energy-density (tau) dependence
# that shows up in both V_xc and the XC quadrature gradient.
FUNCTIONALS = ["pbe", "r2scan"]


def finite_diff_gradient(mol0, with_d4, xc, h=5e-3):
    coords0 = mol0.atom_coords()
    g_fd = numpy.zeros_like(coords0)
    for ia in range(mol0.natm):
        for x in range(3):
            vals = {}
            for sign in (+1, -1):
                coords = coords0.copy()
                coords[ia, x] += sign * h
                molp = mol0.copy()
                molp.set_geom_(coords, unit="Bohr")
                molp.build()
                vals[sign] = ng.total_energy(molp, xc=xc, with_d4=with_d4)
            g_fd[ia, x] = (vals[1] - vals[-1]) / (2 * h)
    return g_fd


def check_d4_pairing():
    """The D4 parameter set must follow the functional, not a stale default."""
    print("D4 damping parameter selection:")
    for xc in FUNCTIONALS + ["pbe,pbe", "r2scan,r2scan"]:
        print(f"  xc={xc!r:18s} -> D4 params {ng.d4_method_for_xc(xc)!r}")
    try:
        ng.d4_method_for_xc("b3lyp,pbe")
    except ValueError:
        print("  mismatched exchange/correlation -> raises (as intended)")
    else:
        print("  [FAIL] mismatched xc string silently accepted")
    print()


def main():
    check_d4_pairing()
    failures = 0
    for name, atom in MOLECULES.items():
        for basis in BASES:
            for xc in FUNCTIONALS:
                for with_d4 in (False, True):
                    mol = gto.M(atom=atom, basis=basis, unit="Angstrom",
                                verbose=0)
                    res = ng.total_gradient(mol, xc=xc, with_d4=with_d4)
                    g_fd = finite_diff_gradient(mol, with_d4, xc)
                    err = numpy.max(numpy.abs(res["g_total"] - g_fd))
                    ok = err < 1e-4
                    failures += not ok
                    label = (f"{name}/{basis}/{xc}, "
                             f"D4={'on' if with_d4 else 'off'}")
                    print(f"[{'PASS' if ok else 'FAIL'}] {label}: "
                          f"max|analytic - finite-diff| = {err:.3e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
