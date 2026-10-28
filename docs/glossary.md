# Glossary

The words these docs use, each with one meaning. Google Cloud terms follow Google's own TPU docs.

## At a glance

| Term | Meaning |
|---|---|
| Accelerator type | The `--accelerator-type` value: which TPU and how many chips, for example `v6e-1` (one v6e chip) or `v6e-4` (four on one VM). |
| AGENTS.md | The file in the repo root that gives a coding agent the rules and the steps, from a Google account with no Cloud setup to a deleted chip. |
| bf16 | A 16-bit number format (bfloat16). Most open models publish their weights in it; we call it full precision. |
| Billing account | The Google Cloud account that pays. A project needs one linked before it can create a chip. |
| Bits per weight (bpw) | A weights file's size in bits divided by the number of weights in it. bf16 is 16. |
| BSHJ | Our weight file format (`.bshj`), built to load fast. One file per weights type. |
| Chip | One TPU chip. A chip bills from create to delete, running or idle. |
| Compile cache | Compiled programs saved on disk (`~/.cache/mashbox/xla` by default), so later starts skip compiling. |
| Culet | Our TPU kernels. Each release ships only the ones its model needs, compiled, as one kernel bundle. |
| Full precision | The weights exactly as the model's makers published them, usually bf16. |
| Greedy | Sampling at temperature 0: the model always takes its most likely next token, so the same prompt gives the same answer. |
| Hugging Face revision | A commit id in a Hugging Face repo. `models.json` pins one per download, so everyone gets the same files. |
| JAX | Google's Python library for numerical code on TPU. Every model port here is written in it. |
| Kernel bundle | One Python wheel that holds a release's compiled kernels. `mashbox run` downloads it from the model's Hugging Face repo and checks its sha256. |
| Lamproite | Our software platform for TPUs: Scaife, Culet, BSHJ, Recut and Masterstone together. |
| Masterstone | The report that grades a port's outputs against the original model's. |
| models.json | The registry in the repo root: one row per model with its chips, run command, files, kernels and licence. Model cards and parts of these docs are generated from it. |
| On-demand | A chip that stays yours until you delete it, billed by the hour at Google's list price. Every command in these docs makes on-demand chips. |
| OpenAI-compatible server | A server with the same routes as OpenAI's API (`/v1/chat/completions` and others), so OpenAI client libraries work against it. |
| Plain JAX | Running a model without our compiled kernels. Slower, with the same outputs within rounding. Used when the kernel bundle was not built for your chip or JAX version. |
| Port | A model's code rewritten in JAX to run on TPU. |
| Project ID | The fixed, unique id of a Google Cloud project, for example `my-tpu-project-513001`. Commands use the ID, never the project name. |
| Quota | How many TPU cores Google lets one project use in one zone. The on-demand row for v6e is "TPU v6e cores per project per zone". |
| Real time | For video models: seconds of compute per second of video. 1.00 is real time; lower is faster. |
| Recut | Porting an open model to TPU. |
| Scaife | Our serving engine for TPU. |
| sha256 | A checksum. `mashbox` checks every file it downloads against the sha256 in `models.json` and stops on a mismatch. |
| Software version | The `--version` value: the TPU VM image, for example `v2-alpha-tpuv6e`. |
| Spot | Cheaper chips Google can take back at any time. The quota page calls them "Preemptible". These docs do not use them. |
| SSH tunnel | `-- -L <port>:127.0.0.1:<port>` on `gcloud compute tpus tpu-vm ssh`. It brings a server on the chip to the same port on your machine, so nothing is open to the internet. |
| Thinking mode | A mode of some language models that writes its reasoning before the answer. Off by default. |
| TPU | Tensor Processing Unit, Google's chip for machine learning, rented through Google Cloud. |
| TPU v5e | A TPU generation with 16 GB of memory per chip. A model runs there when it fits. |
| TPU v6e | The default TPU generation for every model here, with 32 GB of memory per chip. |
| TPU VM | The machine a chip is attached to. You connect to it with SSH and run the model there. |
| Weights type | How a weights file stores its numbers: full precision, or a reduced-size type with fewer bits per weight. Each file name says its type. |
| World model | A model that generates video of a place you steer while it plays. |
| Zone | A Google Cloud location, for example `us-east5-b`. Quota and capacity are counted per zone. |

## Details

### Chip counts

`v6e-1` is one chip on one TPU VM. `v6e-4` is four chips on one TPU VM, billed as four chips. A model's page says which one it needs, and `models.json` lists it in the model's row.

### What bills

Only chips bill in these docs. A project, a billing account, an enabled API and quota cost nothing on their own. See [Run on Google Cloud TPU](google-cloud.md), "What it costs".

## Sources

- TPU terms, accelerator types and software versions: Google's [Cloud TPU docs](https://cloud.google.com/tpu/docs).
- Quota row names: the Google Cloud console quotas page.
