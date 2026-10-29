# <Model name> on TPU

<One paragraph: who made the model, what goes in, what comes out, which chip it runs on, and the one thing a reader
can do with it. No adjectives that a number could replace.>

## What it is

![<what the reader sees>](../media/<model id>-demo-1.<ext>)

*<Caption: model, weights type, chip, settings, seed.>*

<What the model is, in two to four sentences.>

How it runs:

- **Port:** <where the code is, plain JAX, how it is split over chips>
- **Kernels:** <the compiled kernel bundle, where it downloads from, how it is checked>
- **Weights:** <whose weights, which repo, which revision>

## What you need

| Chip | Status |
|---|---|
| TPU v6e-<N> | <Supported | Not supported: why> |
| TPU v5e-<N> | <...> |

<Cost per hour at Google's on-demand list price, and the chip-hours that adds up to.
Anything else the reader needs: Google account, quota, disk.>

## Install

<Create the chip and connect, then install on it. Link docs/google-cloud.md for the Google Cloud setup.>

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
```

## Run

<One command that runs the model, with what it prints or opens. Then the commands that delete the chip.>

```bash
mashbox run <model id>
```

## Options

<The options of `mashbox run <model id>`, each with its default and what it does.>

<!-- cli:start run -->
<!-- cli:end -->

## Variants

| File | Weights type | GB | Bits per weight | Quality vs full precision | Speed | Chips |
|---|---|---|---|---|---|---|
| `<file>` | <type> | <GB> | <bpw> | <measured, with the metric> | <measured, with the unit> | <chips> |

<How the files were made, in one generic line. The original weights stay in the maker's repo.>

## Performance

| Chip | What | Result | Compared with |
|---|---|---|---|
| TPU v6e-<N> | <what was measured, with the settings> | <number with its unit> | <baseline, with the hardware on its side> |

## Use it with your coding agent

<The one prompt to give an agent, and a link to AGENTS.md. Say which steps the agent stops and asks for.>

## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| `<the exact error text>` | <cause> | <fix> |

## FAQ

- **<Short question a reader would ask.>** <Answer in one to three sentences, with a link.>

## Changelog

| Version | What changed |
|---|---|
| <x.y.z> | <what changed for the user> |

## Credits and licences

<Who made the model, with links. Their licence at the pinned revision, and where the file is kept unchanged
(licenses/<model id>/). The mashbox code: see LICENSE. Compiled kernels: see LICENSE-KERNELS.>
