"""Initial-state ablation: singlet-product |S>^{x3} instead of Neel |010101>.

The isotropic exchange interaction conserves total spin, so dynamics from the
total-singlet product state is confined to the 5-dimensional S_tot=0 block of
the M_z=0 sector (vs. the full 20-dimensional manifold from the Neel state).
Paper config, seed 42. Appends F_singlet_init to rerun_results/results.json.
"""
import json, os
import torch
from eo_core import FastEONet, load_mnist_pca, full_test_accuracy, train

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

# |S> = (|01> - |10>)/sqrt(2) on pairs (0,1),(2,3),(4,5)
s = torch.zeros(4, dtype=torch.complex128)
s[0b01] = 2 ** -0.5
s[0b10] = -2 ** -0.5
psi0 = torch.kron(torch.kron(s, s), s)

x12_tr, y_tr, x12_te, y_te, _, _ = load_mnist_pca(12)
torch.manual_seed(42)
m = FastEONet(n_layers=6, n_inputs=12, gamma=0.3, head='mlp',
              init_scale=0.01, jbase_init='zeros')
m.psi0.copy_(psi0)
train(m, x12_tr, y_tr, steps=1000, batch=128, lr=0.003,
      sched='cos', sched_kw={'T_max': 1000}, log_every=200, log_prefix='S ')
acc = full_test_accuracy(m, x12_te, y_te)
print(f'singlet-product init, paper config, seed 42: test acc = {acc*100:.2f}%')

with open(os.path.join(OUT, 'results.json')) as f:
    results = json.load(f)
results['F_singlet_init'] = {'test_acc': acc, 'seed': 42}
with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
