---
${front_matter}
---

# ${name} on Google Cloud TPU

${summary}

| | |
|---|---|
| In | ${in} |
| Out | ${out} |
| Original model | [${upstream_hf}](${upstream_url}) by ${maker} |
| Code | [mashbox v${version}](${code_url}) |

## Run

On a Google Cloud TPU VM. The [docs](${docs_url}) cover setup from a new Google account, and coding agents can follow [AGENTS.md](${agents_url}).

```bash
python3 -m pip install "mashbox[tpu]"
${run}
```

${notes}

## Chips

${chips_table}

## Results

${results_table}

${results_charts}

## Files

${files_table}

## Kernels

${kernels_section}

## Licence

The weights keep the original licence: ${license_text}. MASHBOX code: ${our_license}. ${kernels_license}
