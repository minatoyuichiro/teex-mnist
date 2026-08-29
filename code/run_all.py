"""Full experiment suite for the TE-EX MNIST paper.

A: notebook cell-0 config, full-test-set accuracy
A2: notebook cell-3 (linear head) config, full-test-set accuracy
C: classical baselines + random-frozen-quantum ablation
B: paper config (PCA12, 6 layers, batch 128, 1000 steps) + all figures
D: stability, 5 seeds of config B with per-trial test accuracy
"""
import json, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eo_core import FastEONet, load_mnist_pca, full_test_accuracy, train, DTYPE_R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)
results = {}
T0 = time.time()

def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)

def save():
    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

# ---------------- data ----------------
x8_tr, y_tr, x8_te, y_te, pca8, raw = load_mnist_pca(8)
x12_tr, _, x12_te, _, pca12, _ = load_mnist_pca(12)
x_test_raw = raw['x_test']

# ---------------- A: notebook cell 0, exact config ----------------
log('=== A: notebook cell-0 config (PCA8, 4 layers, MLP head, batch32, 301 steps) ===')
torch.manual_seed(42)
mA = FastEONet(n_layers=4, n_inputs=8, gamma=0.3, head='mlp', init_scale=0.01, jbase_init='zeros')
hA = train(mA, x8_tr, y_tr, steps=301, batch=32, lr=0.002,
           sched='step', sched_kw={'step_size': 150, 'gamma': 0.5}, log_prefix='A ')
accA = full_test_accuracy(mA, x8_te, y_te)
results['A_cell0_config'] = {'test_acc': accA, 'final_batch_acc': hA['acc'][-1],
                             'history_loss': hA['loss'], 'history_acc': hA['acc']}
log(f'A test acc (10000 samples) = {accA*100:.2f}%'); save()

# ---------------- A2: notebook cell 3, linear head ----------------
log('=== A2: notebook cell-3 config (PCA8, 8 layers, gamma0.6, linear head) ===')
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)
mA2 = FastEONet(n_layers=8, n_inputs=8, gamma=0.6, head='linear', init_scale=0.1, jbase_init='randn')
hA2 = train(mA2, x8_tr, y_tr, steps=301, batch=32, lr=0.003,
            sched='cos', sched_kw={'T_max': 400}, log_prefix='A2 ')
accA2 = full_test_accuracy(mA2, x8_te, y_te)
results['A2_cell3_config'] = {'test_acc': accA2, 'final_batch_acc': hA2['acc'][-1]}
log(f'A2 test acc = {accA2*100:.2f}%'); save()

# ---------------- C: classical baselines ----------------
log('=== C: classical baselines ===')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
for d, xtr, xte in [(8, x8_tr, x8_te), (12, x12_tr, x12_te)]:
    sc = StandardScaler().fit(xtr.numpy())
    lr_ = LogisticRegression(max_iter=2000).fit(sc.transform(xtr.numpy()), y_tr.numpy())
    a = lr_.score(sc.transform(xte.numpy()), y_te.numpy())
    results[f'C_logreg_pca{d}'] = {'test_acc': a}
    log(f'logreg PCA{d}: {a*100:.2f}%')
save()

class MLPHead(torch.nn.Module):
    """Identical classical head fed directly with PCA features."""
    def __init__(self, d):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d, 64), torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(), torch.nn.Linear(64, 10)).to(DTYPE_R)
    def forward(self, x):
        return self.classifier(x)

for d, xtr, xte, steps, batch in [(8, x8_tr, x8_te, 301, 32), (12, x12_tr, x12_te, 1000, 128)]:
    torch.manual_seed(42)
    mm = MLPHead(d)
    train(mm, xtr, y_tr, steps=steps, batch=batch, lr=0.003,
          sched='cos', sched_kw={'T_max': steps}, log_every=0)
    a = full_test_accuracy(mm, xte, y_te)
    results[f'C_mlp_direct_pca{d}'] = {'test_acc': a, 'steps': steps, 'batch': batch}
    log(f'MLP-direct PCA{d} (steps={steps},batch={batch}): {a*100:.2f}%')
save()

# random frozen quantum weights (paper config), train head only
log('--- C: random frozen quantum features + trained head ---')
torch.manual_seed(42)
mF = FastEONet(n_layers=6, n_inputs=12, gamma=0.3, head='mlp', init_scale=0.1, jbase_init='randn')
mF.in_w.requires_grad_(False); mF.j_base.requires_grad_(False)
hF = train(mF, x12_tr, y_tr, steps=1000, batch=128, lr=0.003,
           sched='cos', sched_kw={'T_max': 1000}, log_every=200, log_prefix='F ')
accF = full_test_accuracy(mF, x12_te, y_te)
results['C_random_frozen_quantum'] = {'test_acc': accF}
log(f'random frozen quantum: {accF*100:.2f}%'); save()

# ---------------- B: paper config ----------------
log('=== B: paper config (PCA12, 6 layers, MLP head, batch128, 1000 steps) ===')
torch.manual_seed(42)
mB = FastEONet(n_layers=6, n_inputs=12, gamma=0.3, head='mlp', init_scale=0.01, jbase_init='zeros')
hB = train(mB, x12_tr, y_tr, steps=1000, batch=128, lr=0.003,
           sched='cos', sched_kw={'T_max': 1000}, log_every=100, log_prefix='B ')
accB = full_test_accuracy(mB, x12_te, y_te)
results['B_paper_config'] = {'test_acc': accB, 'history_loss': hB['loss'], 'history_acc': hB['acc']}
log(f'B test acc = {accB*100:.2f}%'); save()
torch.save(mB.state_dict(), os.path.join(OUT, 'model_B.pt'))

# quantum-features + linear head, paper config (honest "linear readout" number)
torch.manual_seed(42)
mBL = FastEONet(n_layers=6, n_inputs=12, gamma=0.3, head='linear', init_scale=0.01, jbase_init='zeros')
train(mBL, x12_tr, y_tr, steps=1000, batch=128, lr=0.003,
      sched='cos', sched_kw={'T_max': 1000}, log_every=0)
accBL = full_test_accuracy(mBL, x12_te, y_te)
results['B_linear_head'] = {'test_acc': accBL}
log(f'B linear-head test acc = {accBL*100:.2f}%'); save()

# ---------------- D: stability, 5 seeds ----------------
log('=== D: stability (5 seeds of config B) ===')
stab = []
stab_curves = []
for t in range(5):
    torch.manual_seed(42 + t); np.random.seed(42 + t)
    mt = FastEONet(n_layers=6, n_inputs=12, gamma=0.3, head='mlp', init_scale=0.01, jbase_init='zeros')
    ht = train(mt, x12_tr, y_tr, steps=1000, batch=128, lr=0.003,
               sched='cos', sched_kw={'T_max': 1000}, log_every=0)
    at = full_test_accuracy(mt, x12_te, y_te)
    stab.append(at); stab_curves.append(ht['acc'])
    log(f'trial {t+1}/5 seed {42+t}: test acc {at*100:.2f}%')
results['D_stability'] = {'test_accs': stab, 'mean': float(np.mean(stab)),
                          'std': float(np.std(stab))}
log(f'stability: {np.mean(stab)*100:.2f}% +/- {np.std(stab)*100:.2f}%'); save()

# ---------------- figures (from model B) ----------------
log('=== figures ===')

def smooth(v, k=25):
    v = np.asarray(v)
    return np.convolve(v, np.ones(k) / k, mode='valid')

# fig2: training curve
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(hB['loss'], color='grey', alpha=0.5, label='Loss (batch)')
ax1.set_xlabel('Training Steps'); ax1.set_ylabel('Cross Entropy Loss', color='grey')
ax2 = ax1.twinx()
ax2.plot(np.arange(len(smooth(hB['acc']))) + 12, smooth(hB['acc']), color='navy',
         linewidth=2, label='Accuracy (smoothed)')
ax2.set_ylabel('Mini-batch Accuracy', color='navy'); ax2.set_ylim(0, 1)
fig.tight_layout(); plt.savefig(os.path.join(FIG, 'training_curve.png'), dpi=150); plt.close()

# mnist.png: sample predictions
mB.eval()
rng = np.random.default_rng(0)
idx = rng.choice(len(x12_te), 8, replace=False)
with torch.no_grad():
    preds = mB(x12_te[idx]).argmax(1)
plt.figure(figsize=(15, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(x_test_raw[idx[i]], cmap='gray')
    ok = preds[i].item() == y_te[idx[i]].item()
    plt.title(f'P:{preds[i].item()} T:{y_te[idx[i]].item()}', color='green' if ok else 'red')
    plt.axis('off')
plt.savefig(os.path.join(FIG, 'mnist.png'), dpi=150, bbox_inches='tight'); plt.close()

# pca_components.png
plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(pca12.components_[i].reshape(28, 28), cmap='RdBu', vmin=-0.2, vmax=0.2)
    plt.title(f'PC {i+1}'); plt.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'pca_components.png'), dpi=150,
                                bbox_inches='tight'); plt.close()

# spin_dynamics.png: pulse-by-pulse <Sz> for test sample 0
with torch.no_grad():
    x0 = x12_te[0:1]
    psi = mB.psi0.unsqueeze(0).clone()
    sz_hist = [[float(torch.einsum('bi,ij,bj->b', psi.conj(), mB.sz_stack[s], psi).real)
                for s in range(6)]]
    labels = ['Initial']
    for l in range(mB.n_layers):
        for i in range(mB.n_spins - 1):
            j = torch.tanh(mB.j_base[l, i] + x0 @ mB.in_w[l, i]) * mB.gamma
            V = mB.evecs[i]
            phase = torch.exp(-1j * mB.evals[i].unsqueeze(0) * j.unsqueeze(1).to(torch.complex128))
            psi = (psi @ V.conj()) * phase @ V.T
            sz_hist.append([float(torch.einsum('bi,ij,bj->b', psi.conj(), mB.sz_stack[s], psi).real)
                            for s in range(6)])
            labels.append(f'L{l+1}-P{i+1}')
sz_hist = np.array(sz_hist)
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, 6))
for s in range(6):
    plt.plot(sz_hist[:, s], marker='o', markersize=4, label=f'Spin {s+1}', color=colors[s], alpha=0.8)
plt.xticks(range(len(labels)), labels, rotation=45, fontsize=7)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylim(-0.6, 0.6)
plt.ylabel(r'Expectation Value $\langle S_z \rangle$'); plt.xlabel('Pulse Sequence (Layer-Link)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.5); plt.tight_layout()
plt.savefig(os.path.join(FIG, 'spin_dynamics.png'), dpi=150); plt.close()

# weight_heatmap.png: layer 1
w = mB.in_w[0].detach().numpy()
plt.figure(figsize=(10, 6))
im = plt.imshow(w, cmap='RdBu_r', aspect='auto', vmin=-np.abs(w).max(), vmax=np.abs(w).max())
plt.xticks(range(12), [f'PC{i+1}' for i in range(12)])
plt.yticks(range(5), [f'Link {i+1}\n(S{i+1}-S{i+2})' for i in range(5)])
plt.colorbar(im, label='Weight Magnitude')
plt.xlabel('Input PCA Components'); plt.ylabel('Quantum Gate Links')
for (jj, ii), val in np.ndenumerate(w):
    plt.text(ii, jj, f'{val:.2f}', ha='center', va='center', fontsize=7,
             color='white' if abs(val) > 0.6 * np.abs(w).max() else 'black')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'weight_heatmap.png'), dpi=150); plt.close()

# confusion_matrix.png (full test set)
with torch.no_grad():
    all_preds = torch.cat([mB(x12_te[s:s+2000]).argmax(1) for s in range(0, 10000, 2000)])
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(y_te.numpy(), all_preds.numpy()):
    cm[t, p] += 1
plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap='Blues')
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xticks(range(10)); plt.yticks(range(10))
plt.xlabel('Predicted label'); plt.ylabel('True label')
plt.colorbar(); plt.tight_layout()
plt.savefig(os.path.join(FIG, 'confusion_matrix.png'), dpi=150); plt.close()

# tsne.png (2000 test samples, 11-dim quantum features)
from sklearn.manifold import TSNE
with torch.no_grad():
    feats = torch.cat([mB.features(x12_te[s:s+2000]) for s in range(0, 10000, 2000)]).numpy()
sub = rng.choice(10000, 2000, replace=False)
emb = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(feats[sub])
plt.figure(figsize=(9, 7))
scat = plt.scatter(emb[:, 0], emb[:, 1], c=y_te.numpy()[sub], cmap='tab10', s=8, alpha=0.8)
plt.colorbar(scat, ticks=range(10), label='Digit Class')
plt.xlabel('t-SNE component 1'); plt.ylabel('t-SNE component 2')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'tsne.png'), dpi=150); plt.close()

# signature.png: radar chart of class-averaged raw expectations
feat_names = [f'Ex {i+1}{i+2}' for i in range(5)] + [f'Sz {i+1}' for i in range(6)]
angles = np.linspace(0, 2 * np.pi, 11, endpoint=False).tolist() + [0]
fig, axes = plt.subplots(2, 5, figsize=(20, 8), subplot_kw={'projection': 'polar'})
for d in range(10):
    ax = axes[d // 5, d % 5]
    prof = feats[y_te.numpy() == d].mean(0)
    vals = prof.tolist() + [prof[0]]
    ax.plot(angles, vals, color='teal'); ax.fill(angles, vals, color='teal', alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(feat_names, fontsize=6)
    ax.set_title(f'Digit: {d}', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'signature.png'), dpi=150); plt.close()

# stability figure
plt.figure(figsize=(9, 5))
curves = np.array([smooth(c) for c in stab_curves])
xs = np.arange(curves.shape[1]) + 12
plt.plot(xs, curves.mean(0), color='tab:blue', label='Mean batch accuracy (5 seeds)')
plt.fill_between(xs, curves.mean(0) - curves.std(0), curves.mean(0) + curves.std(0),
                 color='tab:blue', alpha=0.2, label='±1 SD')
plt.xlabel('Training Steps'); plt.ylabel('Smoothed Mini-batch Accuracy'); plt.ylim(0, 1)
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, 'stability.png'), dpi=150); plt.close()

save()
log('ALL DONE')
print(json.dumps({k: (v['test_acc'] if isinstance(v, dict) and 'test_acc' in v else
                      v.get('mean') if isinstance(v, dict) else v)
                  for k, v in results.items()}, indent=2))
