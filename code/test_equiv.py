"""Verify FastEONet matches the notebook's original loop implementation."""
import torch
from eo_core import FastEONet, get_ops, DTYPE_C


def forward_single_original(model, x_pca):
    """Verbatim port of the notebook's forward_single (cell 0)."""
    h_list = [model.h_stack[i] for i in range(model.n_spins - 1)]
    sz_ops = [model.sz_stack[i] for i in range(model.n_spins)]
    psi = torch.zeros(2 ** model.n_spins, 1, dtype=DTYPE_C)
    psi[21] = 1.0
    for l in range(model.n_layers):
        for i in range(model.n_spins - 1):
            j_val = torch.tanh(model.j_base[l, i] + torch.dot(model.in_w[l, i], x_pca)) * model.gamma
            u = torch.matrix_exp(-1j * h_list[i] * j_val.to(DTYPE_C))
            psi = u @ psi
    feats = [torch.real(psi.conj().T @ h @ psi).squeeze() for h in h_list] + \
            [torch.real(psi.conj().T @ sz @ psi).squeeze() for sz in sz_ops]
    return torch.stack(feats)


torch.manual_seed(0)
model = FastEONet(n_layers=4, n_inputs=8, gamma=0.3, init_scale=0.5, jbase_init='randn')
with torch.no_grad():
    model.j_base += torch.randn_like(model.j_base) * 0.5

x = torch.randn(7, 8, dtype=torch.float64) * 3
with torch.no_grad():
    fast = model.features(x)
    orig = torch.stack([forward_single_original(model, xi) for xi in x])
err = (fast - orig).abs().max().item()
print('max |fast - original| =', err)
assert err < 1e-10, 'MISMATCH'

# gradient check
x1 = x[:4]
y = torch.tensor([1, 2, 3, 4])
crit = torch.nn.CrossEntropyLoss()

model.zero_grad()
loss_f = crit(model.classifier(model.features(x1)), y)
loss_f.backward()
g_fast = model.in_w.grad.clone()

model.zero_grad()
feats_o = torch.stack([forward_single_original(model, xi) for xi in x1])
loss_o = crit(model.classifier(feats_o), y)
loss_o.backward()
g_orig = model.in_w.grad.clone()

gerr = (g_fast - g_orig).abs().max().item()
print('max grad diff =', gerr)
assert gerr < 1e-10, 'GRAD MISMATCH'
print('EQUIVALENCE OK')
