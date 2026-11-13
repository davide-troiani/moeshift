# Supported models and tools

MoEShift runs on Macs with Apple Silicon. Its local HTTP server provides chat
and text completions. Choose a model below and add its GGUF file to the
[JSON configuration](../configs/README.md#ordinary-gguf-models).

The tables use the architecture identifiers stored in GGUF files. Models also
need a supported quantization and enough memory. A successful synthetic test
checks execution with small generated weights; it does not validate the quality
or performance of a trained model.

## Models optimized for SSD streaming

These two architectures can read prepared experts from SSD to reduce memory
use. Follow the [setup instructions](../configs/README.md#manual-setup).
They can also load directly from GGUF, which may require much more memory.

| Model / architecture | Tested quantization | Verification |
|---|---|---|
| Qwen 3.8 Flash Next — `qwen4exp` | GSQ-RCO IQ3_XXS | Short and long prompts tested; regression comparison produced identical replies |
| Qwen3.6-35B-A3B — `qwen35moe` | UD Q6_K XL | Q6_K XL inference tested; cache and expert-operation tests pass; latest model regression comparison pending |

See the [model benchmarks](../README.md#tested-models-and-benchmarks) for measured
speed and memory use. Other models sharing these identifiers are not covered
by those measurements.

## Models loaded directly from GGUF

Use `moeshift` for chat and completions, or `llama-cli` / `llama-completion`
from a source build. These models do not need extracted experts.

| Architecture / family | Verification |
|---|---|
| Llama and Mistral — `llama`, `llama4`, `deci`, `mistral3`, `mistral4` | Synthetic inference checked; trained models not tested |
| Qwen — `qwen`, `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `qwen3next`, `qwen35` | Synthetic inference checked; trained models not tested |
| Mamba and other recurrent / hybrid families — `mamba`, `mamba2`, `jamba`, `falcon-h1`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7`, `granitehybrid`, `nemotron_h`, `nemotron_h_moe`, `plamo2`, `kimi-linear` | Synthetic inference checked; trained models not tested |
| Gemma — `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4` | Not tested with trained models |
| Visual model families — `qwen2vl`, `qwen3vl`, `qwen3vlmoe`, `deepseek2-ocr`, `hunyuan_vl`, `minicpm`, `minicpm3`, `minicpm4`, `cogvlm`, `chameleon`, `paddleocr` | Not tested with trained models |
| T5 — `t5` | Not tested with trained models |
| `falcon`, `grok`, `gpt2`, `gptj`, `gptneox`, `mpt`, `baichuan`, `starcoder` | Not tested with trained models |
| `refact`, `bloom`, `stablelm`, `phi2`, `phi3`, `phimoe`, `plamo`, `plamo3` | Not tested with trained models |
| `codeshell`, `orion`, `internlm2`, `starcoder2`, `xverse`, `command-r`, `cohere2`, `dbrx` | Not tested with trained models |
| `olmo`, `olmo2`, `olmoe`, `openelm`, `arctic`, `deepseek`, `deepseek2`, `deepseek32` | Not tested with trained models |
| `chatglm`, `glm4`, `glm4moe`, `glm-dsa`, `bitnet`, `jais`, `jais2`, `nemotron` | Not tested with trained models |
| `exaone`, `exaone4`, `exaone-moe`, `granite`, `granitemoe`, `plm`, `bailingmoe`, `bailingmoe2` | Not tested with trained models |
| `dots1`, `arcee`, `afmoe`, `ernie4_5`, `ernie4_5-moe`, `hunyuan-moe`, `hunyuan-dense`, `smollm3` | Not tested with trained models |
| `gpt-oss`, `lfm2`, `lfm2moe`, `smallthinker`, `seed_oss`, `grovemoe`, `apertus`, `minimax-m2` | Not tested with trained models |
| `rnd1`, `mimo2`, `step35`, `maincoder`, `talkie` | Not tested with trained models |

For image or audio input, use the tools below with a matching model and
companion files. The MoEShift HTTP server accepts text only.

## Tools for other tasks

The tools below are available when [building from source](build.md#additional-tools).
Their build and help commands have been checked. Inference with trained
media, speech and embedding models has not been tested.

| Task | Tool | Models / requirements |
|---|---|---|
| Image and supported audio input | `llama-mtmd-cli` | A supported model and matching projector; includes supported Qwen VL, Gemma and MiniCPM variants |
| Speech synthesis | `llama-tts-server`, `llama-tts` | VoxCPM2 for the TTS server; matching speech models for each tool |
| Embeddings / text encoding | `llama-embedding` | Encoder architectures listed below; also usable through the library API |
| Diffusion text generation | `llama-diffusion` | `dream`, `llada`, `llada-moe` |
| Draft-model execution | Library API and speculative examples | `dflash`; not available through the MoEShift server |
| Audio decoding component | Library API / audio tools | `wavtokenizer-dec`; not a standalone chat model |

Encoder architectures: `bert`, `modern-bert`, `nomic-bert`, `nomic-bert-moe`,
`neo-bert`, `jina-bert-v2`, `jina-bert-v3`, `eurobert`, `gemma-embedding`,
`llama-embed`, `pangu-embedded`, `t5encoder`.

`clip` identifies a companion model, not a standalone chat model. Unknown
architecture identifiers are rejected during loading.

[Build and test instructions](build.md) explain how to repeat the checks.

Saved conversation caches from intermediate MoEShift revisions may need to be
rebuilt for architectures other than Qwen4Exp. Keep the model files and original
prompts; current Qwen4Exp caches remain compatible.
