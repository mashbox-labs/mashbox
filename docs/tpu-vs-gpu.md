# TPU or GPU: price and bandwidth

Google Cloud rents TPUs and NVIDIA H100 GPUs by the hour. This page puts them side by side on Google's own list prices, with the memory bandwidth of each and what it means for Qwen3.8-27B. Every table is rendered from one file, [docs/data/cloud-prices.json](data/cloud-prices.json), and the measured speeds come from [models.json](../models.json).

## Price per hour

<!-- cloud:start prices -->
Google Cloud on-demand list prices in USD, from Google's [Cloud TPU pricing](https://cloud.google.com/tpu/pricing) and [GPU pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized) pages. Each price includes the host machine.

| Hardware | Memory | Per hour | Region | As Google lists it |
|---|---|---|---|---|
| 1 TPU v6e chip | 32 GB | $2.70 | us-east1, us-east5 | $2.70 per chip-hour |
| TPU v6e-4 (4 chips) | 128 GB | $10.80 | us-east1, us-east5 | 4 x $2.70 per chip-hour |
| 1 NVIDIA H100 80GB SXM | 80 GB | $11.06 | us-central1 | $88.49 per hour for a3-highgpu-8g (8 GPUs), divided by 8 |
<!-- cloud:end -->

<!-- cloud:start options -->
The same Google Cloud pages, per hour for one chip or one GPU:

| Per hour | On demand | DWS Flex-start | 1-year commitment | 3-year commitment |
|---|---|---|---|---|
| 1 TPU v6e chip | $2.70 | $1.35 | $1.89 | $1.22 |
| 1 NVIDIA H100 80GB SXM | $11.06 | $4.79 | $7.67 | $4.86 |
<!-- cloud:end -->

## Memory bandwidth

<!-- cloud:start bandwidth -->
| Hardware | Memory | Memory bandwidth | Bandwidth per $1 an hour, on demand | Spec |
|---|---|---|---|---|
| 1 TPU v6e chip | 32 GB | 1,638 GB/s | 607 GB/s | [Google](https://cloud.google.com/tpu/docs/v6e) |
| TPU v6e-4 (4 chips) | 128 GB | 6,552 GB/s | 607 GB/s | 4 x the chip |
| 1 NVIDIA H100 80GB SXM | 80 GB | 3,350 GB/s | 303 GB/s | [NVIDIA](https://www.nvidia.com/en-us/data-center/h100/) |
<!-- cloud:end -->

With one user and one token per step, a language model reads every weight from memory once for each token it writes. Memory bandwidth divided by the size of the weights is therefore the ceiling on tokens per second for that user without speculative decoding, and real runs land below it. [Speculative decoding](models/qwen3-8-27b.md#speculative-decoding) writes more than one token per pass, so it can go past this ceiling.

## Same price per hour: TPU v6e-4 and one H100

<!-- cloud:start pair -->
For about the same price per hour ($10.80 and $11.06), a TPU v6e-4 has 1.6x the memory (128 GB and 80 GB) and 1.96x the memory bandwidth (6,552 GB/s and 3,350 GB/s) of one H100.
<!-- cloud:end -->

## Qwen3.8-27B speed

<!-- cloud:start speed -->
Qwen3.8-27B (27B parameters, dense), our default weights file (26.94 GB, 8.01 bits per weight), output tokens per second:

| Hardware | Per hour | Users | Tokens per second | Kind |
|---|---|---|---|---|
| 1 TPU v6e chip | $2.70 | 1 | 38.6 | measured |
| 1 TPU v6e chip | $2.70 | 8 at once | 269 total | measured |
| 1 TPU v6e chip | $2.70 | 1 | ~61 | ceiling: 1,638 GB/s / 26.94 GB |
| TPU v6e-4 (4 chips) | $10.80 | 1 | ~243 | ceiling: 6,552 GB/s / 26.94 GB |
| 1 NVIDIA H100 80GB SXM | $11.06 | 1 | ~124 | ceiling: 3,350 GB/s / 26.94 GB |
<!-- cloud:end -->

Measured rows: the engine on one TPU v6e-1 in Google Cloud, 64-token prompts, 128 new tokens, greedy. Speeds through the OpenAI-compatible server are on the [Qwen3.8-27B page](models/qwen3-8-27b.md). The H100 row is a ceiling, not a measurement, for FP8 weights the same size as ours.

For one user, one TPU v6e chip at its measured speed costs less per output token than one NVIDIA H100 at its ceiling, as the next table shows.

## Cost per 1M output tokens

<!-- cloud:start cost -->
Qwen3.8-27B, estimated as price per hour / output tokens per hour, at the Google Cloud on-demand list prices above:

| Hardware | Users | Tokens per second | USD per 1M output tokens |
|---|---|---|---|
| 1 TPU v6e chip | 8 at once | 269 total, measured | $2.79 |
| 1 TPU v6e chip | 1 | 38.6, measured | $19.43 |
| 1 TPU v6e chip | 1 | ~61, ceiling | $12.34 |
| TPU v6e-4 (4 chips) | 1 | ~243, ceiling | $12.34 |
| 1 NVIDIA H100 80GB SXM | 1 | ~124, ceiling | $24.71 |
<!-- cloud:end -->

## Start

Cloud TPUs are rented by the hour through Google Cloud, with no hardware to buy, so the software is what decides how easy they are to use. Two lines on your own machine install `mashbox`, get a TPU v6e chip and start Qwen3.8-27B on it:

```bash
curl -fsSL https://mashbox.gg/install.sh | sh
mashbox tpu up qwen3-8-27b
```

Details: [Install](install.md). To run each step yourself with `gcloud`: [Quickstart](quickstart.md). Chip specs and zones: [TPU v6e and v5e](chips.md).
