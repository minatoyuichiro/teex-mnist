"""Batch-Normalization ablation: paper config with the BN layer removed from
the classical head (Linear(11,64) -> ReLU -> Linear(64,10)).

Everything else identical to config B of run_all.py (PCA-12, 6 layers,
gamma 0.3, batch 128, 1000 steps, cosine schedule, seed 42).  Appends
H_no_bn to results/results.json.
"""
import json, os
import torch
from eo_core import FastEONet, load_mnist_pca, full_test_accuracy, train, DTYPE_R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

x12_tr, y_tr, x12_te, y_te, _, _ = load_mnist_pca(12)
torch.manual_seed(42)
m = FastEONet(n_layers=6, n_inputs=12, gamma=0.3, head='mlp',
              init_scale=0.01, jbase_init='zeros')
m.classifier = torch.nn.Sequential(
    torch.nn.Linear(11, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)).to(DTYPE_R)
train(m, x12_tr, y_tr, steps=1000, batch=128, lr=0.003,
      sched='cos', sched_kw={'T_max': 1000}, log_every=200, log_prefix='noBN ')
acc = full_test_accuracy(m, x12_te, y_te)
print(f'no-BN head, paper config, seed 42: test acc = {acc*100:.2f}%')

with open(os.path.join(OUT, 'results.json')) as f:
    results = json.load(f)
results['H_no_bn'] = {'test_acc': acc, 'seed': 42}
with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
