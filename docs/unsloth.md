# Unsloth fine-tunes

Fine-tune with [Unsloth](https://github.com/unslothai/unsloth), save a GGUF, and run it with mashbox. `mashbox inspect` on the
file or repo says whether it runs here, and why not, before anything loads. See [GGUF files](gguf.md).

```python
# after training with Unsloth
model.save_pretrained_gguf("my-model", tokenizer, quantization_method="q4_k_m")
# or straight to the Hub
model.push_to_hub_gguf("you/my-model-GGUF", tokenizer, quantization_method="q4_k_m")
```

Then run the `.gguf` file Unsloth wrote, or the Hub repo:

```bash
mashbox run you/my-model-GGUF:Q4_K_M
```

The chat template and stop tokens come from the file, so the template you trained with is the one that runs. Unsloth's
published GGUF repos are read the same way:

```bash
mashbox run unsloth/Qwen3.8-27B-GGUF
mashbox run unsloth/Qwen3-4B-GGUF
mashbox run unsloth/Llama-3.2-3B-Instruct-GGUF
```

Which quant: Unsloth's `UD-` files mix block types per layer, and mashbox reads every type in them. Every type runs on
the exact reference path, which is slow: see [What runs](gguf.md#what-runs). Run
`mashbox inspect <file or repo>` to see the mix and the chip memory it needs.
