# TE-EX MNIST

Code and numerical results for the paper:

> **Hardware-Efficient Exchange-Only QML: Singlet-Triplet Spin Chains via
> Inter-pair Coupling without Magnetic Gradients**
> Yuichiro Minato (blueqat Inc.)
> [arXiv:2608.29017](https://arxiv.org/abs/2608.29017)

The Time-Encoded Exchange (TE-EX) protocol classifies MNIST digits with a
6-spin Heisenberg chain controlled solely by exchange pulses — no magnetic
field gradients. Data-dependent pulse areas of every exchange link are trained
end-to-end together with a small classical head.

## Results reproduced by this code

| Model (identical PCA-12 input, optimizer, budget) | Test acc. (%) |
|---|---|
| Logistic regression on PCA-12 (linear) | 83.21 |
| Classical MLP head on PCA-12 | 91.26 |
| TE-EX, frozen random pulses + MLP head | 52.97 |
| TE-EX, trained pulses + linear readout | 88.08 |
| **TE-EX, trained pulses + MLP head (full)** | **90.89** |

Stability over 5 seeds: 90.99% ± 0.24%. Robustness (trained model, test-time
noise): 89.9% under 10% quasi-static pulse-area noise; 89.8% with 10³
measurement shots per observable. Singlet-product initial-state ablation:
86.96% (dynamics confined to the 5-dimensional S_tot = 0 block).

All accuracies are on the full 10,000-image MNIST test set.

## Reproduce

```bash
pip install -r requirements.txt
cd code
python test_equiv.py        # exactness check vs. reference implementation (<1e-10)
python run_all.py           # full experiment suite + figures (~4 min on CPU)
python noise_eval.py        # robustness: pulse noise + finite shots (needs run_all first)
python singlet_ablation.py  # initial-state ablation
```

Outputs are written to `results/` (`results.json` and `figures/`). The
committed `results/` directory contains the exact run used in the paper.
`code/mnist.npz` is the standard MNIST dataset (LeCun et al.) in NumPy format,
included for one-command reproduction.

## Implementation note

`eo_core.py` propagates the whole batch at once and replaces `expm` per pulse
with a precomputed eigendecomposition of each link generator
(h = XX + YY + ZZ = 4 S·S) — exact, no approximation. Numerical equivalence
with a verbatim reference implementation is enforced by `test_equiv.py`
(state error < 1e-10, gradients match to machine precision).

## License

MIT — see [LICENSE](LICENSE).
