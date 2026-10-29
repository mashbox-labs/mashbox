# TPU v6e and v5e

The two TPU generations these docs use, side by side, and which one to pick. Specs are Google's; prices are Google's on-demand list prices.

## At a glance

| | TPU v6e | TPU v5e |
|---|---|---|
| Memory per chip | 32 GB | 16 GB |
| Memory bandwidth per chip | 1,638 GBps | 800 GiBps |
| Peak bf16 per chip | 918 TFLOPs | 197 TFLOPs |
| Peak int8 per chip | 1,836 TOPs | 393 TOPs |
| Chip-to-chip bandwidth (ICI) | 800 GBps | 400 GBps |
| `--accelerator-type` (one chip) | `v6e-1` | `v5litepod-1` |
| `--version` | `v2-alpha-tpuv6e` | `v2-alpha-tpuv5-lite` |
| On-demand list price, one chip | $2.70 an hour | about $1.20 an hour |

## Details

- **v6e is the default** chip for every model.
- **v5e** has half the memory and about a fifth of the peak compute. A model runs there when its weights and working memory fit in 16 GB. Each model's row in the README says which chips it runs on.
- Each model's compiled kernel bundle is built per chip and per JAX version, and `models.json` lists which. On a chip or JAX version it was not built for, the model runs on plain JAX instead and prints one line saying so.

### Per model

<!-- chips:start -->
| Model | Runs on | TPU v5e | TPU v6e | TPU v5p slice (v5p-8) | TPU v6e slice (v6e-4) |
|---|---|---|---|---|---|
| Hy-Nomas | TPU v6e-4 | Not supported: runs only on a v6e-4 (a TPU v6e slice of 4 chips) | One chip: the full pipeline fits on a single TPU v6e chip (32 GB), 2.46 s of compute per second of video (measured 26 October 2026). One-chip mode in `mashbox run` ships in the next release; this version starts on a v6e-4 | Not supported: runs only on a v6e-4 (a TPU v6e slice of 4 chips) | Supported: 4 chips |
| Qwen3.8-27B | TPU v6e-1 | Supported: a v5e-4 (4 chips), split across them automatically; one v5e chip is too small: the weights alone are about 27 GB and a v5e chip has 16 GB | Supported: one chip (v6e-1) | Not supported | Supported: 4 chips, split across them automatically |
<!-- chips:end -->

## Examples

One TPU v6e chip, then one TPU v5e chip:

```bash
gcloud compute tpus tpu-vm create mashbox-1 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm create mashbox-2 --zone=us-central1-a --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite
```

Each one bills until you delete it ([Run on Google Cloud TPU](google-cloud.md), step 8).

## Sources

- Specs: Google's [v6e](https://cloud.google.com/tpu/docs/v6e) and [v5e](https://cloud.google.com/tpu/docs/v5e) pages.
- Prices: [cloud.google.com/tpu/pricing](https://cloud.google.com/tpu/pricing).
- Which model fits where: each model's row in `models.json`.
