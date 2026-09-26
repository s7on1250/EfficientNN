# Homework 1: CNN

## Equations

In `equations.py` I add up the MACs from each Conv and Linear layer and count two FLOPs per MAC. The model has no biases. I leave out ReLU, pooling, and GAP arithmetic.

For memory, I compare the tensors that are alive at each step. The input is still held by the measurement script.

For bytes moved, I count one read and one write per layer activation, one read of each parameter, and a read/write for in-place ReLU. I assume reuse across overlapping pool windows.

For latency and energy, I used these models and fitted their parameters on the GPU:

```text
Latency = t0 + max(FLOPs/P, Bytes/W)
Energy  = E0 + α·FLOPs + β·Bytes
```

`P` and `W` are effective fitted rates, not the GPU's advertised peaks. Energy is for the whole GPU, including idle draw during measurement.

## How I ran it

I used a Tesla T4 in Kaggle, PyTorch, CUDA 12.8, cuDNN 9.10.2, and Python 3.12.13. The model ran in `eval()` and `torch.inference_mode()` with TF32 and cuDNN benchmarking disabled. Each configuration had 5 warmups and 15 timed passes, latency is their median. Energy came from NVML's whole-GPU counter.

The grid has 132 `(S,B)` pairs, including the extra values sampled with seed 2026. I fitted on 106 pairs and kept 26 for validation.

To rerun this in a Kaggle GPU notebook, copy the folder to `/kaggle/working/hw1`, install `nvidia-ml-py`, then run `python measure.py` and `python calibrate.py` from that folder.

## Results and discussion

The saved measurements and graphs come from the Kaggle run before I removed the Linear biases. To update them for this version, rerun `measure.py` and `calibrate.py` on Kaggle.

All 132 runs finished, with no OOM or other errors. Every row has latency, peak memory, and energy. `results/measurements.csv` has the raw measurements; `results/theta.json` has the fit, errors, and OOM summary; `results/figures/` has the plots.

| Validation metric | Mean absolute error | Mean relative error |
|---|---:|---:|
| Latency | 5.58 ms | 33.94% |
| Peak memory | 246 MB | 53.10% |
| Energy | 0.394 J | 32.06% |

The time and energy models fit the largest jobs fairly well. At `S=512, B=256`, time was 0.519 s measured versus 0.526 s predicted, and energy was 37.2 J versus 36.2 J. The fit puts `t0` and `E0` almost at zero, so it misses that floor. It minimizes ordinary squared errors, where large jobs carry much more weight. The fitted `β` is also almost zero; that does not mean moving data costs no energy.

Memory is even less accurate. Its formula is below the measured peak in all 132 cases. cuDNN workspaces and other temporary allocations may explain part of the gap, but I did not measure those parts separately. The largest measured peak is still below GPU RAM, so I cannot test an OOM boundary with this grid on the T4.

Batching does not lower energy per image indefinitely either. At `S=224`, it falls from 36.5 mJ at `B=1` to 20.7 mJ at `B=16`, then rises to 29.0 mJ at `B=256`. One fixed set of coefficients cannot capture every kernel choice and batch size. I treat the fitted memory/compute labels as model predictions, not direct proof of the GPU's bottleneck.
