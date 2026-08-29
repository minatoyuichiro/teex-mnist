"""Robustness evaluation of the trained paper-config model (model_B.pt).

Two noise channels, evaluated at test time on the full 10,000-image test set
(the model itself is trained noiselessly):

(a) Quasi-static exchange-pulse noise: every pulse area is multiplied by
    (1 + eps) with eps ~ N(0, sigma^2), drawn independently per link, per
    layer and per sample (shot-to-shot charge-noise model, multiplicative
    because J depends exponentially on the barrier gate voltage).

(b) Finite measurement shots: each of the 11 expectation values is replaced
    by a shot-noise-corrupted estimate  h_k + N(0, Var[O_k]/n_shots)  with
    the exact quantum variance Var[O_k] = <O_k^2> - <O_k>^2  (Gaussian /
    central-limit approximation of binomial sampling).

Results appended to rerun_results/results.json; figure to
rerun_results/figures/robustness.png.
"""
import json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eo_core import FastEONet, load_mnist_pca

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
FIG = os.path.join(OUT, 'figures')
T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


class NoisyEONet(FastEONet):
    """FastEONet with multiplicative quasi-static pulse-area noise and an
    optional finite-shot Gaussian model on the measured features."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.noise_sigma = 0.0
        self.n_shots = None          # None = infinite shots (exact expectations)
        self.noise_gen = None        # torch.Generator for reproducibility
        # <O^2> operators for the exact single-shot variance
        self.register_buffer('h2_stack', torch.stack(
            [h @ h for h in self.h_stack]))
        self.register_buffer('sz2_stack', torch.stack(
            [s @ s for s in self.sz_stack]))

    def propagate(self, x_batch):
        B = x_batch.shape[0]
        psi = self.psi0.unsqueeze(0).expand(B, -1).clone()
        for l in range(self.n_layers):
            for i in range(self.n_spins - 1):
                j = torch.tanh(self.j_base[l, i] + x_batch @ self.in_w[l, i]) * self.gamma
                if self.noise_sigma > 0:
                    eps = torch.randn(B, dtype=j.dtype, generator=self.noise_gen)
                    j = j * (1.0 + self.noise_sigma * eps)
                V = self.evecs[i]
                phase = torch.exp(-1j * self.evals[i].unsqueeze(0)
                                  * j.unsqueeze(1).to(torch.complex128))
                psi = (psi @ V.conj()) * phase @ V.T
        return psi

    def features(self, x_batch):
        psi = self.propagate(x_batch)
        e_h = torch.einsum('bi,kij,bj->bk', psi.conj(), self.h_stack, psi).real
        e_z = torch.einsum('bi,kij,bj->bk', psi.conj(), self.sz_stack, psi).real
        feats = torch.cat([e_h, e_z], dim=1)
        if self.n_shots is not None:
            e_h2 = torch.einsum('bi,kij,bj->bk', psi.conj(), self.h2_stack, psi).real
            e_z2 = torch.einsum('bi,kij,bj->bk', psi.conj(), self.sz2_stack, psi).real
            var = torch.clamp(torch.cat([e_h2, e_z2], dim=1) - feats ** 2, min=0.0)
            sem = torch.sqrt(var / self.n_shots)
            feats = feats + sem * torch.randn(feats.shape, dtype=feats.dtype,
                                              generator=self.noise_gen)
        return feats


@torch.no_grad()
def accuracy(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for s in range(0, len(x), bs):
        correct += (model(x[s:s + bs]).argmax(1) == y[s:s + bs]).sum().item()
    return correct / len(x)


def main():
    x12_tr, y_tr, x12_te, y_te, _, _ = load_mnist_pca(12)
    m = NoisyEONet(n_layers=6, n_inputs=12, gamma=0.3, head='mlp',
                   init_scale=0.01, jbase_init='zeros')
    m.load_state_dict(torch.load(os.path.join(OUT, 'model_B.pt'),
                                 weights_only=True), strict=False)

    m.noise_gen = torch.Generator()
    baseline = accuracy(m, x12_te, y_te)
    log(f'noiseless baseline: {baseline*100:.2f}%')
    assert abs(baseline - 0.9089) < 1e-6, 'baseline mismatch'

    R = 5  # noise realizations per setting

    # (a) pulse-area noise sweep
    sigmas = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    pulse = {}
    for s in sigmas:
        accs = []
        for r in range(R):
            m.noise_sigma, m.n_shots = s, None
            m.noise_gen.manual_seed(1000 + r)
            accs.append(accuracy(m, x12_te, y_te))
        pulse[s] = accs
        log(f'pulse noise sigma={s:.2f}: {np.mean(accs)*100:.2f}% '
            f'+/- {np.std(accs)*100:.2f}%')

    # (b) finite-shot sweep (no pulse noise)
    shot_list = [100, 300, 1000, 3000, 10000]
    shots = {}
    for n in shot_list:
        accs = []
        for r in range(R):
            m.noise_sigma, m.n_shots = 0.0, n
            m.noise_gen.manual_seed(2000 + r)
            accs.append(accuracy(m, x12_te, y_te))
        shots[n] = accs
        log(f'{n:6d} shots: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%')

    # (c) combined: sigma=0.05 with 1000 shots (a representative realistic point)
    accs = []
    for r in range(R):
        m.noise_sigma, m.n_shots = 0.05, 1000
        m.noise_gen.manual_seed(3000 + r)
        accs.append(accuracy(m, x12_te, y_te))
    combined = accs
    log(f'combined sigma=0.05 + 1000 shots: {np.mean(accs)*100:.2f}% '
        f'+/- {np.std(accs)*100:.2f}%')

    # save into results.json
    with open(os.path.join(OUT, 'results.json')) as f:
        results = json.load(f)
    results['E_robustness'] = {
        'baseline': baseline,
        'pulse_noise': {str(k): v for k, v in pulse.items()},
        'finite_shots': {str(k): v for k, v in shots.items()},
        'combined_sigma005_1000shots': combined,
        'realizations': R,
    }
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # figure: two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    xs = np.array(sigmas)
    mu = np.array([np.mean(pulse[s]) for s in sigmas]) * 100
    sd = np.array([np.std(pulse[s]) for s in sigmas]) * 100
    ax1.axhline(baseline * 100, color='grey', linestyle='--', linewidth=1,
                label=f'Noiseless ({baseline*100:.1f}%)')
    ax1.errorbar(xs * 100, mu, yerr=sd, marker='o', color='tab:blue', capsize=3)
    ax1.set_xlabel(r'Pulse-area noise $\sigma_J$ (%)')
    ax1.set_ylabel('Test accuracy (%)')
    ax1.set_title('(a) Quasi-static exchange noise')
    ax1.grid(alpha=0.3)
    ax1.legend()

    xn = np.array(shot_list)
    mu2 = np.array([np.mean(shots[n]) for n in shot_list]) * 100
    sd2 = np.array([np.std(shots[n]) for n in shot_list]) * 100
    ax2.axhline(baseline * 100, color='grey', linestyle='--', linewidth=1,
                label=f'Noiseless ({baseline*100:.1f}%)')
    ax2.errorbar(xn, mu2, yerr=sd2, marker='s', color='tab:red', capsize=3)
    ax2.set_xscale('log')
    ax2.set_xlabel('Measurement shots per observable')
    ax2.set_title('(b) Finite sampling')
    ax2.grid(alpha=0.3, which='both')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'robustness.png'), dpi=150)
    log('figure written: robustness.png')
    log('DONE')


if __name__ == '__main__':
    main()
