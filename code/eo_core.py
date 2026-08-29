"""Vectorized, numerically-equivalent reimplementation of the notebook's
TE-EX (Time-Encoded Exchange) model.

Speed tricks (both exact, no approximation):
  - each link Hamiltonian h_i is constant, so expm(-i h_i j) = V_i diag(exp(-i w_i j)) V_i^dag
    with (w_i, V_i) precomputed by eigendecomposition;
  - the whole batch is propagated at once with two (B,64)x(64,64) matmuls per pulse.
"""
import numpy as np
import torch

DTYPE_R = torch.float64
DTYPE_C = torch.complex128


def get_ops(n_spins):
    I = torch.eye(2, dtype=DTYPE_C)
    X = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE_C)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE_C)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=DTYPE_C)

    def kron_list(ms):
        r = ms[0]
        for m in ms[1:]:
            r = torch.kron(r, m)
        return r

    h_list = []
    for i in range(n_spins - 1):
        ox = [I] * n_spins; ox[i] = X; ox[i + 1] = X
        oy = [I] * n_spins; oy[i] = Y; oy[i + 1] = Y
        oz = [I] * n_spins; oz[i] = Z; oz[i + 1] = Z
        h_list.append(kron_list(ox) + kron_list(oy) + kron_list(oz))

    sz_ops = []
    for i in range(n_spins):
        oz = [I] * n_spins; oz[i] = Z * 0.5
        sz_ops.append(kron_list(oz))
    return h_list, sz_ops


class FastEONet(torch.nn.Module):
    """Equivalent to the notebook models; head is configurable.

    head='mlp'    -> Linear(nf,64), BatchNorm1d(64), ReLU, Linear(64,10)   (Cell 0)
    head='linear' -> BatchNorm1d(nf), Linear(nf,10)                        (Cell 3)
    """

    def __init__(self, n_spins=6, n_layers=4, n_inputs=8, gamma=0.3,
                 head='mlp', init_scale=0.01, jbase_init='zeros'):
        super().__init__()
        self.n_spins, self.n_layers, self.gamma = n_spins, n_layers, gamma

        self.in_w = torch.nn.Parameter(
            torch.randn(n_layers, n_spins - 1, n_inputs, dtype=DTYPE_R) * init_scale)
        if jbase_init == 'zeros':
            self.j_base = torch.nn.Parameter(
                torch.zeros(n_layers, n_spins - 1, dtype=DTYPE_R))
        else:
            self.j_base = torch.nn.Parameter(
                torch.randn(n_layers, n_spins - 1, dtype=DTYPE_R) * init_scale)

        nf = (n_spins - 1) + n_spins
        if head == 'mlp':
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(nf, 64), torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(), torch.nn.Linear(64, 10)).to(DTYPE_R)
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.BatchNorm1d(nf), torch.nn.Linear(nf, 10)).to(DTYPE_R)

        h_list, sz_ops = get_ops(n_spins)
        # eigendecomposition of each (Hermitian) link Hamiltonian
        evals, evecs = [], []
        for h in h_list:
            w, V = torch.linalg.eigh(h)
            evals.append(w)          # real (64,)
            evecs.append(V)          # complex (64,64)
        self.register_buffer('evals', torch.stack(evals))            # (L-1? no: n_links,64)
        self.register_buffer('evecs', torch.stack(evecs))            # (n_links,64,64)
        self.register_buffer('h_stack', torch.stack(h_list))         # (n_links,64,64)
        self.register_buffer('sz_stack', torch.stack(sz_ops))        # (n_spins,64,64)

        psi0 = torch.zeros(2 ** n_spins, dtype=DTYPE_C)
        psi0[0b010101] = 1.0                                         # |010101> = index 21
        self.register_buffer('psi0', psi0)

    def propagate(self, x_batch):
        """x_batch (B, n_inputs) -> final states psi (B, 64)."""
        B = x_batch.shape[0]
        psi = self.psi0.unsqueeze(0).expand(B, -1).clone()
        for l in range(self.n_layers):
            for i in range(self.n_spins - 1):
                j = torch.tanh(self.j_base[l, i] + x_batch @ self.in_w[l, i]) * self.gamma
                V = self.evecs[i]
                phase = torch.exp(-1j * self.evals[i].unsqueeze(0)
                                  * j.unsqueeze(1).to(DTYPE_C))     # (B,64)
                psi = (psi @ V.conj()) * phase @ V.T
        return psi

    def features(self, x_batch):
        psi = self.propagate(x_batch)
        # <h_i> then <Sz_i>, matching the notebook's feature order
        e_h = torch.einsum('bi,kij,bj->bk', psi.conj(), self.h_stack, psi).real
        e_z = torch.einsum('bi,kij,bj->bk', psi.conj(), self.sz_stack, psi).real
        return torch.cat([e_h, e_z], dim=1)

    def forward(self, x_batch):
        return self.classifier(self.features(x_batch))


def load_mnist_pca(n_pca, mnist_path=None):
    """PCA identical to sklearn's (fit on train, centered, no whitening)."""
    if mnist_path is None:
        import os
        candidates = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist.npz'),
                      os.path.expanduser('~/.keras/datasets/mnist.npz')]
        mnist_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
    from sklearn.decomposition import PCA
    d = np.load(mnist_path)
    x_train = d['x_train'].reshape(-1, 784) / 255.0
    x_test = d['x_test'].reshape(-1, 784) / 255.0
    pca = PCA(n_components=n_pca)
    x_train_pca = torch.tensor(pca.fit_transform(x_train), dtype=DTYPE_R)
    x_test_pca = torch.tensor(pca.transform(x_test), dtype=DTYPE_R)
    y_train = torch.tensor(d['y_train'], dtype=torch.long)
    y_test = torch.tensor(d['y_test'], dtype=torch.long)
    return x_train_pca, y_train, x_test_pca, y_test, pca, d


@torch.no_grad()
def full_test_accuracy(model, x_test, y_test, bs=1000):
    model.eval()
    correct = 0
    for s in range(0, len(x_test), bs):
        logits = model(x_test[s:s + bs])
        correct += (logits.argmax(1) == y_test[s:s + bs]).sum().item()
    return correct / len(x_test)


def train(model, x_train, y_train, steps=301, batch=32, lr=0.002,
          sched='step', sched_kw=None, log_every=50, log_prefix=''):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if sched == 'step':
        sk = sched_kw or {'step_size': 150, 'gamma': 0.5}
        scheduler = torch.optim.lr_scheduler.StepLR(opt, **sk)
    else:
        sk = sched_kw or {'T_max': steps}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, **sk)
    crit = torch.nn.CrossEntropyLoss()
    history = {'loss': [], 'acc': []}
    for step in range(steps):
        model.train()
        idx = torch.randperm(len(x_train))[:batch]
        xb, yb = x_train[idx], y_train[idx]
        logits = model(xb)
        loss = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == yb).float().mean().item()
        history['loss'].append(loss.item()); history['acc'].append(acc)
        if log_every and step % log_every == 0:
            print(f'{log_prefix}step {step:5d}  loss {loss.item():.4f}  batch_acc {acc:.2f}',
                  flush=True)
    return history
