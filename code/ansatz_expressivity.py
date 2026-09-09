"""Can the finite TE-EX ansatz actually realize an arbitrary symmetry-allowed
transformation?

dla_dimension.py shows the exchange generators span a 129-dimensional algebra,
so 129 real parameters are needed to reach a generic element of
SU(5) x SU(9) x SU(5) x U(1).  The ansatz of L layers has only 5L parameters,
and in the trained model the pulse areas are additionally bounded by
|theta| <= gamma = 0.3 through the tanh parameterization.

We compile a target that acts as a random SU(5) on the S_tot = 0 block and as
the identity on the other blocks, and report the best achievable

    F = |Tr(V^dag U(theta))| / 20      (1 = exact, up to a global phase)

for several depths, both with unconstrained pulse areas and with the bound the
trained model actually operates under.

Expressivity is monotone in L -- a deeper ansatz contains a shallower one by
setting the extra angles to zero -- so the slightly lower fidelities returned at
L = 40 and L = 60 are an artefact of optimizing more parameters under the same
restart and step budget, not a loss of expressivity.

Appends L_expressivity.
"""
import json, os
import numpy as np
import torch
from eo_core import get_ops

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
torch.set_default_dtype(torch.float64)
N_SPINS, GAMMA, DEPTHS = 6, 0.3, [6, 13, 26, 40, 60]


def setup():
    h_list, sz_ops = get_ops(N_SPINS)
    sz_tot = sum(sz_ops)
    idx = (torch.diagonal(sz_tot).real.abs() < 1e-12).nonzero().squeeze()
    h_r = [h[idx][:, idx] for h in h_list]
    I2 = torch.eye(2, dtype=torch.complex128)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128) / 2
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128) / 2

    def kron_list(ms):
        r = ms[0]
        for m in ms[1:]:
            r = torch.kron(r, m)
        return r

    sx = sum(kron_list([X if k == i else I2 for k in range(N_SPINS)]) for i in range(N_SPINS))
    sy = sum(kron_list([Y if k == i else I2 for k in range(N_SPINS)]) for i in range(N_SPINS))
    s2 = (sx @ sx + sy @ sy + sz_tot @ sz_tot)[idx][:, idx]
    _, B = torch.linalg.eigh(s2)
    return [torch.linalg.eigh(h) for h in h_r], B, len(idx)


def random_su5_target(B, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.randn(5, 5, generator=g, dtype=torch.float64)
         + 1j * torch.randn(5, 5, generator=g, dtype=torch.float64))
    Q, R = torch.linalg.qr(A)
    Q = Q * (torch.diagonal(R) / torch.diagonal(R).abs())
    Q = Q / torch.det(Q) ** (1 / 5)
    M = torch.eye(dim, dtype=torch.complex128)
    M[:5, :5] = Q
    return B @ M @ B.conj().T


def unitary(theta, eig, dim):
    U = torch.eye(dim, dtype=torch.complex128)
    for l in range(theta.shape[0]):
        for i in range(5):
            w, V = eig[i]
            ph = torch.exp(-1j * w.to(torch.complex128) * theta[l, i].to(torch.complex128))
            U = (V * ph) @ V.conj().T @ U
    return U


def fit(L, bounded, eig, V_target, dim, restarts=4, steps=3000):
    best = 0.0
    for r in range(restarts):
        torch.manual_seed(100 * r + L)
        p = torch.nn.Parameter(torch.randn(L, 5) * (1.0 if bounded else 0.5))
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(steps):
            theta = GAMMA * torch.tanh(p) if bounded else p
            F = torch.trace(V_target.conj().T @ unitary(theta, eig, dim)).abs() / dim
            (1 - F).backward()
            opt.step()
            opt.zero_grad()
        best = max(best, float(F.detach()))
    return best


if __name__ == '__main__':
    eig, B, dim = setup()
    V_target = random_su5_target(B, dim)
    rows = []
    print(f'{"L":>4} {"params":>7} {"F free":>9} {"F bounded":>11}')
    for L in DEPTHS:
        ff = fit(L, False, eig, V_target, dim)
        fb = fit(L, True, eig, V_target, dim)
        rows.append({'n_layers': L, 'n_params': 5 * L, 'fidelity_free': ff,
                     'fidelity_bounded': fb})
        print(f'{L:>4} {5*L:>7} {ff:>9.4f} {fb:>11.4f}', flush=True)

    with open(os.path.join(OUT, 'results.json')) as f:
        results = json.load(f)
    results['L_expressivity'] = {
        'target': 'random SU(5) on the S_tot=0 block, identity elsewhere',
        'figure_of_merit': '|Tr(V^dag U)| / 20',
        'dla_dim_needed': 129, 'gamma_bound': GAMMA, 'rows': rows}
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
