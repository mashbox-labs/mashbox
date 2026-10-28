---
${front_matter}
---

# ${name} GGUF

${summary}

| | |
|---|---|
| Original model | [${upstream_hf}](${upstream_url}) by ${maker} |
| Runs on | llama.cpp, stock build |
| Code and notes | [mashbox v${version}](${code_url}) |

## Run

```bash
llama-server -hf ${hf_repo}:${default_tag}
```

${notes}

## Files

${files_table}

## Results

${results_table}

${results_charts}

## Licence

The weights keep the original licence: ${license_text}.
