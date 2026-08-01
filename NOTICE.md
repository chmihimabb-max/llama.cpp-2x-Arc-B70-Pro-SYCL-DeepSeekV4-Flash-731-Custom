# Notices and attribution

## This repository

The scripts and documentation in this repository (`scripts/`, `README.md`)
are MIT licensed — see [LICENSE](LICENSE).

## Vendored components

### llama.cpp

`vendor/llama.cpp` is a **git submodule** referencing upstream
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), pinned to commit
`876a4321163249c43ca4e986818fab5ab081f282` (master, includes `deepseek4`
support from PR [#24162](https://github.com/ggml-org/llama.cpp/pull/24162)).

- **Unmodified** — zero local changes; the pin exists only for reproducibility.
- License: **MIT**, Copyright (c) 2023-2024 The ggml authors.
  The full license text ships with the submodule at
  `vendor/llama.cpp/LICENSE` once initialized
  (`git submodule update --init --recursive`).

Redistribution/reference under these terms is permitted by the MIT license,
which requires only that the copyright and license notice be retained — they
are, inside the submodule checkout.

### Model weights

The DeepSeek-V4-Flash-0731 GGUF weights are **not** distributed by this repo.
Download them from
[unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
and use them under the model's own license terms.
