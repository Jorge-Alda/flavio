
from math import sqrt
import numpy as np
from flavio.config import config
from functools import lru_cache
from flavio import citations
from flavio.physics.common import lambda_K


def z(mB, mM, q2, par, t0=None):
    r"""Form factor expansion parameter $z$.
    Parameters
    ----------
    - `mB`:
        initial pseudoscalar meson mass
    - `mM`:
        final meson meson mass
    - `q2`:
        momentum transfer squared $q^2$
    - `par`: 
    """
    mB0 = par['m_B0']
    mDs = par['m_D*0']
    t0 = t0 or (mB-mM)**2  # t0 = tm
    tp = (mB0+mDs)**2
    sq2 = sqrt(tp-q2)
    st0 = sqrt(tp-t0)
    res = (sq2-st0)/(sq2+st0)
    return res


def pole(ff, mres, q2, mB, mM, par):
    mresdict = {'A0': 0, 'A1': 2, 'A2': 2, 'V': 1, 'T1': 1, 'T2': 2, 'T3': 2}
    m0 = mres[mresdict[ff]]
    p = 1
    for m in m0:
        p *= z(mB, mM, q2, par, m**2)
    return 1/p


process_dict = {}
process_dict['Bc->J/psi'] = {'B': 'Bc', 'V': 'J/psi',   'q': 'b->c'}


def ff(process, q2, par, n=(3, 4, 4)):
    pd = process_dict[process]
    mres = []
    mp = []
    for i in range(n[0]):
        mp.append(par[process + ' HDL m0 p' + str(i)])
    mres.append(mp)
    mp = []
    for i in range(n[1]):
        mp.append(par[process + ' HDL m1- p' + str(i)])
    mres.append(mp)
    mp = []
    for i in range(n[2]):
        mp.append(par[process + ' HDL m1+ p' + str(i)])
    mres.append(mp)
    mB = par['m_'+pd['B']]
    mV = par['m_'+pd['V']]
    ff = {}
    # setting a0_A0 and a0_T2 according to the exact kinematical relations,
    # cf. eq. (16) of arXiv:1503.05534
    citations.register("Straub:2015ica")
    par_prefix = process + ' HDL'
    for i in ["A0", "A1", "A2", "V", "T1", "T2", "T3"]:
        a = [par[par_prefix + ' a' + str(j) + '_' + i] for j in range(4)]
        zv = z(mB, mV, q2, par)
        zs = [zv**j for j in range(4)]
        ff[i] = pole(i, mres, q2, mB, mV, par)*np.dot(a, zs)
    ff['A12'] = ((mB+mV)**2*(mB**2-mV**2-q2)*ff['A1'] -
                 lambda_K(mB**2, mV**2, q2)*ff['A2'])/(16*mB*mV**2*(mB+mV))
    ff['T23'] = ((mB**2-mV**2)*(mB**2+3*mV**2-q2)*ff['T2'] -
                 lambda_K(mB**2, mV**2, q2)*ff['T3'])/(8*mB*mV**2*(mB-mV))
    return ff
