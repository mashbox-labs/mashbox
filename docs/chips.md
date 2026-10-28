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
| On-demand list price, one chip | $2.70 an hour (read 19 October 2026) | about $1.20 an hour |

## Details

- **v6e is the default.** Every model is tested on it first.
- **v5e** has half the memory and about a fifth of the peak compute. A model runs there when its weights and working memory fit in 16 GB. Each model's row in the README says which chips it passed on, with the date.
- Each model's compiled kernel bundle is built per chip and per JAX version, and `models.json` lists which. On a chip or JAX version it was not built for, the model runs on plain JAX instead and prints one line saying so.

### Per model

| Model | TPU v6e | TPU v5e |
|---|---|---|
| [Qwen3.8-27B](models/qwen3-8-27b.md) | one chip (`v6e-1`) | does not fit: the weights alone are about 27 GB |
| Hy-Nomas (coming soon) | four chips (`v6e-4`) | not supported yet: needs more than 16 GB per chip as laid out |

## Examples

One TPU v6e chip, then one TPU v5e chip:

<!-- TODO source="release test run: copy-paste check of both create commands on a fresh machine" -->

```bash
gcloud compute tpus tpu-vm create mashbox-1 --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm create mashbox-2 --zone=us-central1-a --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite
```

Each one bills until you delete it ([Run on Google Cloud TPU](google-cloud.md), step 8).

## Sources

- Specs: Google's [v6e](https://cloud.google.com/tpu/docs/v6e) and [v5e](https://cloud.google.com/tpu/docs/v5e) pages, read 28 October 2026.
- Prices: [cloud.google.com/tpu/pricing](https://cloud.google.com/tpu/pricing), read 19 October 2026.
- Which model fits where: each model's row in `models.json`.

## Changelog

| Date | What changed |
|---|---|
| 28 October 2026 | Per-model table and examples added. |
