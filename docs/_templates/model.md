<!--
Model page template. Copy to docs/models/<model id>.md. Keep every "## " heading below, in this order, with these exact
words; the site and coding agents find sections by them. A section with nothing tested yet keeps its heading, one
"Pending" line that says which run fills it, and a TODO comment naming the source:
  <!- - TODO source="<test record or release run that fills this>" - ->   (no spaces in the real marker)
Values that are also in models.json (chips, run command, licence, files) must match it. Every command is copy-paste
tested on a fresh machine before the page ships. Every number names the chip, the date and the record it comes from.
Words follow docs/glossary.md.
-->
# <Model name> on TPU

<One paragraph: who made the model, what goes in, what comes out, which chip it runs on, and the one thing a reader
can do with it. No adjectives that a number could replace.>

> Status: <staging | released DATE>. <One sentence on what is and is not tested yet.>

## What it is

<!-- Media slot (media standard): a clip, terminal recording or screenshot of the released build running the documented
command on the named chip. Caption: model and weights type, chip, settings, seed, UTC date. -->
![<what the reader sees>](../media/<model id>-demo-1.<ext>)

*<Caption: model, weights type, chip, settings, seed, date.>*

<What the model is, in two to four sentences.>

How it runs:

- **Port:** <where the code is, plain JAX, how it is split over chips>
- **Kernels:** <the compiled kernel bundle, where it downloads from, how it is checked>
- **Weights:** <whose weights, which repo, which revision>

## What you need

| Chip | Status |
|---|---|
| TPU v6e-<N> | <passed DATE | pending | not supported: why> |
| TPU v5e-<N> | <...> |

<Cost per hour at Google's on-demand list price, with the date it was read, and the chip-hours that adds up to.
Anything else the reader needs: Google account, quota, disk.>

## Install

<Create the chip and connect, then install on it. Link docs/google-cloud.md for the Google Cloud setup.>

```bash
python3 -m pip install uv && python3 -m uv tool install --python 3.12 "mashbox[tpu]"
```

## Run

<One command that runs the model, with what it prints or opens. Then the commands that delete the chip.>

```bash
python3 -m mashbox run <model id>
```

## Options

<Every option and flag, one row each. Copy the defaults from `python3 -m mashbox run <model id> --help`.>

| Option | Default | What it does |
|---|---|---|
| `--<option>` | <default> | <what it does, and when to change it> |

## Variants

| File | Weights type | GB | Bits per weight | Quality vs full precision | Speed | Chips |
|---|---|---|---|---|---|---|
| `<file>` | <type> | <GB> | <bpw> | <measured, with the metric> | <measured, with the unit> | <chips> |

<How the files were made, in one generic line. The original weights stay in the maker's repo.>

## Performance

| Chip | What we measured | Date | Record |
|---|---|---|---|
| TPU v6e-<N> | <number with its unit, against what baseline; name the hardware on both sides> | <date> | <link> |

## Use it with your coding agent

<The one prompt to give an agent, and a link to AGENTS.md. Say which steps the agent stops and asks for.>

## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| `<the exact error text from our own test runs>` | <cause> | <fix> |

## FAQ

- **<Short question a first-time user asked or would ask.>** <Answer in one to three sentences, with a link.>

## Changelog

| Version | Date | What changed |
|---|---|---|
| <x.y.z> | <date> | <what changed for the user> |

## Credits and licences

<Who made the model, with links. Their licence as recorded at our pin, and where the file is kept unchanged
(licenses/<model id>/). Our code: see LICENSE. Compiled kernels: see LICENSE-KERNELS.>
