"""What does the spin chain contribute beyond the classical tanh encoder?

The TE-EX encoder already applies a non-linearity BEFORE any quantum evolution:
every pulse area is theta_{l,i} = gamma * tanh(j_base_{l,i} + w_{l,i}.x), so the
6 x 5 ansatz contains 30 tanh units parameterized by in_w (360) + j_base (30)
= 390 numbers.  The spin chain then maps those 30 angles to 11 observables, and
a BatchNorm + Linear head reads them out (142 parameters, 532 in total).

Three families of controls, all trained with the identical optimizer and budget
(Adam, batch 128, 1000 steps, cosine schedule):

  (A) classical tanh networks of varying width, to locate the accuracy of a
      purely classical model at a matched *parameter count*;
  (B) the identical encoder (Linear(12,30) + tanh, 390 parameters) followed by a
      FIXED, untrained map 30 -> 11 and the identical BN+Linear readout, so that
      the only thing that differs from TE-EX is what performs the 30 -> 11 step;
  (C) a purely linear model, for reference.

Reference numbers from run_all.py: logistic regression on PCA-12 = 83.21%,
TE-EX trained pulses + linear readout = 88.08%, TE-EX + MLP head = 90.89%.
Appends J_classical_controls to results/results.json.
"""
import json, os
import numpy as np
import torch
from eo_core import load_mnist_pca, train, DTYPE_R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
GAMMA = 0.3


def evaluate(model, x_te, y_te):
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, len(x_te), 1000):
            correct += (model(x_te[s:s + 1000]).argmax(1) == y_te[s:s + 1000]).sum().item()
    return correct / len(x_te)


class Plain(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.classifier = torch.nn.Sequential(*layers).to(DTYPE_R)

    def forward(self, x):
        return self.classifier(x)


class FixedMapControl(torch.nn.Module):
    """TE-EX with the spin chain replaced by a fixed, untrained map 30 -> 11.

    kind='linear'  f = theta @ A            (A random, fixed)
    kind='quad'    f_k = theta^T M_k theta  (M_k random symmetric, fixed) --
                   the same *algebraic form* as a quantum expectation value
                   <psi|O_k|psi>, but with no dynamics behind it.
    """

    def __init__(self, kind, map_seed):
        super().__init__()
        self.kind = kind
        self.encoder = torch.nn.Linear(12, 30).to(DTYPE_R)      # == in_w + j_base
        g = torch.Generator().manual_seed(map_seed)
        self.register_buffer('A', torch.randn(30, 11, generator=g, dtype=DTYPE_R) / 30 ** 0.5)
        Ms = torch.randn(11, 30, 30, generator=g, dtype=DTYPE_R) / 30
        self.register_buffer('M', Ms + Ms.transpose(1, 2))
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(11), torch.nn.Linear(11, 10)).to(DTYPE_R)

    def forward(self, x):
        theta = torch.tanh(self.encoder(x)) * GAMMA
        if self.kind == 'linear':
            f = theta @ self.A
        else:
            f = torch.einsum('bi,kij,bj->bk', theta, self.M, theta)
        return self.classifier(f)


def sweep(build, seeds):
    accs = []
    for t in seeds:
        torch.manual_seed(t)
        m = build()
        train(m, x_tr, y_tr, steps=1000, batch=128, lr=0.003,
              sched='cos', sched_kw={'T_max': 1000}, log_every=0)
        accs.append(evaluate(m, x_te, y_te))
    n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return accs, n_par


if __name__ == '__main__':
    x_tr, y_tr, x_te, y_te, _, _ = load_mnist_pca(12)
    out = {'note': 'TE-EX linear readout = 88.08% with 390 encoder + 142 head = 532 params'}

    # (A) classical tanh networks, width sweep
    out['tanh_width_sweep'] = {}
    for h in [6, 8, 11, 15, 20, 30]:
        accs, n_par = sweep(lambda h=h: Plain([
            torch.nn.Linear(12, h), torch.nn.Tanh(),
            torch.nn.BatchNorm1d(h), torch.nn.Linear(h, 10)]), range(42, 47))
        out['tanh_width_sweep'][f'hidden{h}'] = {
            'params': n_par, 'test_accs': accs,
            'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
        print(f'tanh hidden {h:2d} ({n_par:3d} params): '
              f'{np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%', flush=True)

    # (C) purely linear
    accs, n_par = sweep(lambda: Plain([
        torch.nn.BatchNorm1d(12), torch.nn.Linear(12, 10)]), range(42, 47))
    out['linear_only'] = {'params': n_par, 'test_accs': accs,
                          'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
    print(f'linear only  ({n_par:3d} params): {np.mean(accs)*100:.2f}%', flush=True)

    # (B) identical encoder + fixed untrained 30 -> 11 map + identical readout
    for kind in ['linear', 'quad']:
        per_map, flat = {}, []
        for ms in range(5):
            accs, n_par = sweep(lambda k=kind, s=ms: FixedMapControl(k, s), range(42, 45))
            per_map[f'map{ms}'] = accs
            flat += accs
            print(f'fixed random {kind:6s} map#{ms}: {np.mean(accs)*100:.2f}%', flush=True)
        out[f'fixed_random_{kind}_map'] = {
            'params': n_par, 'per_map': per_map, 'mean': float(np.mean(flat)),
            'std': float(np.std(flat)), 'n_maps': 5, 'seeds_per_map': 3}
        print(f'fixed random {kind:6s} OVERALL {np.mean(flat)*100:.2f}% '
              f'+/- {np.std(flat)*100:.2f}%', flush=True)

    # (D) the chain itself with the linear readout, over the same five seeds, so that
    #     the reference number is a mean like the controls (seed 42 = 88.08% of run_all.py)
    from eo_core import FastEONet
    accs, n_par = sweep(lambda: FastEONet(n_layers=6, n_inputs=12, gamma=0.3, head='linear',
                                          init_scale=0.01, jbase_init='zeros'), range(42, 47))
    out['teex_linear_5seeds'] = {'params': n_par, 'test_accs': accs,
                                 'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
    print(f'TE-EX linear readout, 5 seeds ({n_par:3d} params): '
          f'{np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%', flush=True)

    with open(os.path.join(OUT, 'results.json')) as f:
        results = json.load(f)
    results['J_classical_controls'] = out
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
