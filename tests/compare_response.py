"""
Cross-check and benchmark the two response-term implementations:

  per-atom : solve nuclear CPHF once per atom (reference; pyscf's own
             solve_mo1, batched internally across perturbations)
  zvector  : Handy-Schaefer Z-vector -- one solve regardless of natom

They must agree. The interesting question this answers is *where the
Z-vector actually starts paying off*, and whether the derivative-Fock
build (make_h1), which both paths need and neither can avoid, keeps
dominating.

Usage:
    python compare_response.py [max_waters] [basis]
e.g.
    python compare_response.py 8 6-31g
"""
import sys
import os
import time
import numpy
from pyscf import gto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import nonscf_grad as ng  # noqa: E402

XC = "r2scan"


def water_chain(n, spacing=3.0):
    """n waters strung out along z -- crude but fine for scaling behaviour."""
    at = []
    for i in range(n):
        z = i * spacing
        at += [f"O 0.00 0.00 {z:.4f}",
               f"H 0.96 0.10 {z + 0.05:.4f}",
               f"H -0.30 0.87 {z - 0.20:.4f}"]
    return "; ".join(at)


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    basis = sys.argv[2] if len(sys.argv) > 2 else "6-31g"

    print(f"functional={XC}  basis={basis}  waters=1..{nmax}")
    print(f"pyscf threads: ", end="")
    from pyscf import lib
    print(lib.num_threads())
    print()
    print(f"{'natm':>5} {'nao':>5} {'perturb':>8} | {'per-atom':>10} {'zvector':>10} "
          f"{'speedup':>8} | {'make_h1':>9} {'solve':>8} | {'agree':>10}")
    print("-" * 96)

    failures = 0
    for n in range(1, nmax + 1):
        mol = gto.M(atom=water_chain(n), basis=basis, unit="Angstrom", verbose=0)
        mf = ng.run_hf(mol)
        dm = mf.make_rdm1()
        F = ng.ks_fock_on_density(mol, dm, XC)

        t = time.time(); g_ref = ng.response_grad(mf, F);         t_ref = time.time() - t
        t = time.time(); g_z = ng.zvector_response_grad(mf, F);   t_z = time.time() - t

        # how much of the z-vector path is the irreducible derivative-Fock build?
        t = time.time(); mf.Hessian().make_h1(mf.mo_coeff, mf.mo_occ)
        t_h1 = time.time() - t

        err = numpy.abs(g_ref - g_z).max()
        scale = max(numpy.abs(g_ref).max(), 1e-30)
        ok = err / scale < 1e-5
        failures += not ok

        print(f"{mol.natm:>5} {mol.nao:>5} {3*mol.natm:>8} | {t_ref:>9.2f}s {t_z:>9.2f}s "
              f"{t_ref/max(t_z,1e-9):>7.1f}x | {t_h1:>8.2f}s {max(t_z-t_h1,0):>7.2f}s | "
              f"{err:>10.2e} {'' if ok else '  <-- MISMATCH'}")

    print()
    if failures:
        print(f"*** {failures} size(s) DISAGREED between the two methods ***")
    else:
        print("all sizes agree: the Z-vector reproduces the per-atom reference")
    print()
    print("Reading this: 'make_h1' is the derivative-Fock build, needed by BOTH")
    print("methods and irreducible. 'solve' is what the Z-vector actually")
    print("replaces. The speedup is capped by how much of the time is make_h1,")
    print("so watch whether the solve column grows faster than make_h1 with size.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
