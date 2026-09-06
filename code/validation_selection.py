"""Model selection on a held-out validation split (no test-set peeking).

The 60,000 MNIST training images are split once (seed 0) into 50,000 for
training and 10,000 for validation.  PCA is fit on the 50,000 only.  A grid of
configurations (PCA dimension x number of layers x gamma) is trained on the
50,000 and scored on the validation split.  The configuration with the best
validation accuracy is then retrained on the full 60,000 (paper protocol,
seed 42) and evaluated ONCE on the 10,000-image test set.

Everything else (head, optimizer, batch 128, 1000 steps, cosine schedule)
follows the paper configuration.  Appends G_validation_selection to
results/results.json.
"""
import json, os, time, itertools
import numpy as np
import torch
from sklearn.decomposition import PCA
from eo_core import FastEONet, full_test_accuracy, train, DTYPE_R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
T0 = time.time()

def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist.npz'))
x_all = d['x_train'].reshape(-1, 784) / 255.0
y_all = torch.tensor(d['y_train'], dtype=torch.long)
x_test_raw = d['x_test'].reshape(-1, 784) / 255.0
y_test = torch.tensor(d['y_test'], dtype=torch.long)

rng = np.random.default_rng(0)
perm = rng.permutation(len(x_all))
tr_idx, va_idx = perm[:50000], perm[50000:]

def pca_split(n_pca):
    pca = PCA(n_components=n_pca).fit(x_all[tr_idx])
    xtr = torch.tensor(pca.transform(x_all[tr_idx]), dtype=DTYPE_R)
    xva = torch.tensor(pca.transform(x_all[va_idx]), dtype=DTYPE_R)
    return xtr, y_all[tr_idx], xva, y_all[va_idx]

GRID = dict(n_pca=[8, 12, 16], n_layers=[4, 6, 8], gamma=[0.1, 0.3, 0.6])
rows = []
pca_cache = {}
for n_pca, n_layers, gamma in itertools.product(*GRID.values()):
    if n_pca not in pca_cache:
        pca_cache[n_pca] = pca_split(n_pca)
    xtr, ytr, xva, yva = pca_cache[n_pca]
    torch.manual_seed(42)
    m = FastEONet(n_layers=n_layers, n_inputs=n_pca, gamma=gamma, head='mlp',
                  init_scale=0.01, jbase_init='zeros')
    train(m, xtr, ytr, steps=1000, batch=128, lr=0.003,
          sched='cos', sched_kw={'T_max': 1000}, log_every=0)
    acc = full_test_accuracy(m, xva, yva)
    rows.append({'n_pca': n_pca, 'n_layers': n_layers, 'gamma': gamma, 'val_acc': acc})
    log(f'PCA{n_pca:2d} L{n_layers} gamma{gamma:.1f}: val acc {acc*100:.2f}%')

best = max(rows, key=lambda r: r['val_acc'])
log(f'best on validation: {best}')

# retrain the validation-selected configuration on the full training set
# (paper protocol, seed 42) and evaluate once on the test set
pca = PCA(n_components=best['n_pca']).fit(x_all)
xtr_full = torch.tensor(pca.transform(x_all), dtype=DTYPE_R)
xte = torch.tensor(pca.transform(x_test_raw), dtype=DTYPE_R)
torch.manual_seed(42)
m = FastEONet(n_layers=best['n_layers'], n_inputs=best['n_pca'], gamma=best['gamma'],
              head='mlp', init_scale=0.01, jbase_init='zeros')
train(m, xtr_full, y_all, steps=1000, batch=128, lr=0.003,
      sched='cos', sched_kw={'T_max': 1000}, log_every=0)
test_acc = full_test_accuracy(m, xte, y_test)
log(f'validation-selected config on the test set: {test_acc*100:.2f}%')

with open(os.path.join(OUT, 'results.json')) as f:
    results = json.load(f)
results['G_validation_selection'] = {
    'split': 'train 50000 / val 10000 (seed 0), PCA fit on the 50000 only',
    'grid': GRID, 'rows': rows, 'best_on_val': best, 'test_acc_of_best': test_acc,
    'paper_config': {'n_pca': 12, 'n_layers': 6, 'gamma': 0.3}}
with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
log('done')
