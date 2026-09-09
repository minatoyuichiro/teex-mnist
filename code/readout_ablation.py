"""Which observables carry the classification?

The 11-dimensional feature vector splits into two groups that are read out by
two different primitives of the platform:

  * 5 link correlators  c_ij = <h_ij> = 4<S_i.S_j> = 1 - 4<P_S>, i.e. the
    singlet probability of the pair -- measured by Pauli spin blockade;
  * 6 local magnetizations <S_z,i> -- measured by energy-selective
    spin-to-charge conversion (Elzerman readout).

This script retrains the paper configuration with the head restricted to each
group, five seeds each.  Everything else (PCA-12, 6 layers, gamma 0.3, Adam,
batch 128, 1000 steps, cosine schedule) is unchanged.

NOTE: the head is rebuilt after the model here so that all three arms consume
the RNG in the same order.  This shifts the 11-feature control to 90.5% rather
than the 90.89% of run_all.py; the three arms are comparable with each other,
but the control is not the paper number.  Appends I_readout_ablation.
"""
import json, os
import numpy as np
import torch
from eo_core import FastEONet, load_mnist_pca, train, DTYPE_R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
SEEDS = 5


class SubsetEONet(FastEONet):
    """FastEONet exposing only a slice of the 11 observables to the head."""
    SL = slice(None)

    def features(self, x_batch):
        return super().features(x_batch)[:, self.SL]


def run(sl, n_features):
    accs = []
    for t in range(SEEDS):
        torch.manual_seed(42 + t)
        m = SubsetEONet(n_layers=6, n_inputs=12, gamma=0.3, head='mlp',
                        init_scale=0.01, jbase_init='zeros')
        m.SL = sl
        m.classifier = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64), torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(), torch.nn.Linear(64, 10)).to(DTYPE_R)
        train(m, x_tr, y_tr, steps=1000, batch=128, lr=0.003,
              sched='cos', sched_kw={'T_max': 1000}, log_every=0)
        m.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(x_te), 1000):
                correct += (m(x_te[s:s + 1000]).argmax(1) == y_te[s:s + 1000]).sum().item()
        accs.append(correct / len(x_te))
    return accs


if __name__ == '__main__':
    x_tr, y_tr, x_te, y_te, _, _ = load_mnist_pca(12)
    arms = {
        'psb_only_5_correlators': (slice(0, 5), 5),
        'sz_only_6_magnetizations': (slice(5, 11), 6),
        'all_11_control': (slice(None), 11),
    }
    results_new = {}
    for name, (sl, nf) in arms.items():
        accs = run(sl, nf)
        results_new[name] = {'test_accs': accs, 'mean': float(np.mean(accs)),
                             'std': float(np.std(accs)), 'n_features': nf}
        print(f'{name:26s} {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%', flush=True)

    with open(os.path.join(OUT, 'results.json')) as f:
        results = json.load(f)
    results['I_readout_ablation'] = {'seeds': list(range(42, 42 + SEEDS)), **results_new}
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
