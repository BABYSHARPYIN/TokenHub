# 共享分层推理计算池

v1 layered inference pool. Python package name: `pool`.

The only model is quantized **Qwen2.5-7B-Instruct**, 28 layers. Mac always runs layers `[0, 14)`. Windows always runs layers `[14, 28)`. Tokenizer, plaintext, embeddings, `lm_head`, sampling, and answer/thinking assembly stay on Mac. Windows never tokenizes, detokenizes, or emits plaintext.

Hidden states and KV travel over cleartext HTTP. v1 does not encrypt them, bill for them, or authenticate tenants.

The browser is a console. A local agent on each machine does the work. The browser does not run layer compute. vllm-metal (Mac) and CUDA vLLM (Windows / WSL2) are engines only. `HiddenStateRunner` hooks decoder internals for a layer range. It does not call `vllm` chat, `LLM.generate`, or OpenAI chat completions.

Default `pytest` uses the mock engine and needs no GPU.

## 1. Mac single-machine 7B smoke

This path is the one to run first. It does not need the Windows machine.

Mock engine (no weights, proves the console and the front-half fallback):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pool --role mac --engine mock --host 127.0.0.1 --port 7410 --mode auto
```

Open http://127.0.0.1:7410 . Leave mode on `auto` or choose `single-mac`, then send a prompt.

Windows is not configured, so the session is front-half smoke:

- `degraded` is `mac_front_half`
- the console says this is front-half smoke and that output quality does not count
- prefill and decode run only Mac layers `[0, 14)`
- KV stays in memory on the Mac
- sampling applies `lm_head` to those front-half hidden states
- layers `[14, 28)` do not run, and the agent does not call a full-model chat API

The mock engine's tokens are a UTF-8 stand-in, not the Qwen vocabulary. The banner still applies.

Real 7B smoke (Apple Silicon, macOS 15 or newer, vllm-metal installed):

```bash
pip install -e ".[metal]"
python -m pool --role mac --engine vllm-metal --model Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 7410 --mode auto --max-model-len 512
```

Same console contract: with Windows unreachable, `auto` and `single-mac` stay on layers `[0, 14)`. Do not expect coherent 7B answers from this path. A 16 GB Mac is tight if vLLM also allocates its own cache; `--max-model-len 512` keeps that cache small. The pool still only *executes* the front half.

`mode=dual` with Windows down fails the session. It does not silently switch to front-half.

## 2. Mac + Windows/WSL2 dual-machine (half and half)

Windows node and KV run inside WSL2 so CUDA vLLM can see the NVIDIA GPU. Publish two cleartext ports from WSL2 to the Windows host's public address (example: `7421` for KV, `7422` for the layer node). There is no tunnel encryption in v1.

On the Windows / WSL2 side:

```bash
python -m pool --role windows --engine vllm-cuda \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --kv-port 7421 --node-port 7422 \
  --max-model-len 512
```

Mock the Windows half first if the GPU stack is not ready yet: replace `--engine vllm-cuda` with `--engine mock`.

On the Mac:

```bash
python -m pool --role mac --engine vllm-metal \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 7410 --mode auto \
  --windows-kv http://WINDOWS_PUBLIC_IP:7421 \
  --windows-node http://WINDOWS_PUBLIC_IP:7422 \
  --health-timeout 3 \
  --max-model-len 512
```

`configs/mac.example.json` and `configs/windows.example.json` freeze the same ports, model, and layer ranges. Replace `WINDOWS_PUBLIC_IP` before using the Mac file. A leftover placeholder is treated as "Windows not configured", so `auto` stays on front-half smoke.

What `auto` does before a session:

1. Health-check Windows KV and the Windows node together, budget about 3 seconds.
2. If both are up and the node reports layers `[14, 28)`, run the dual path. `degraded` is empty.
3. Mac embeds the prompt, runs prefill `[0, 14)`, then Windows runs prefill `[14, 28)`. KV for every layer is stored on Windows.
4. Mac applies the final norm and `lm_head` to the layer-27 hidden state and samples.
5. Decode repeats that half-and-half step. Windows sees float32 activations and KV, never the prompt string or token ids.

If a dual session has already started and Windows drops, the session fails and its KV is deleted. The dispatcher does not switch to front-half in the middle of that session.

## 3. Risks

- **No encryption.** Layer activations and KV cross the public internet in the clear. Anyone who can read the packets can read the hidden states. v1 does not add TLS, authentication, or a crypto envelope.
- **Front-half quality does not count.** `mac_front_half` runs only layers `[0, 14)` and projects those hidden states with `lm_head`. It is a pipeline smoke test. It is not a 28-layer answer, and the Mac agent will not quietly run all 28 layers to make the text look better.
- **macOS 15+.** The real Mac engine is vllm-metal (vLLM's API/scheduler plus MLX/Metal). Plan on macOS 15 or newer. The mock engine runs anywhere.
- **WSL2.** The real Windows engine is CUDA vLLM. Run it in WSL2 with the NVIDIA GPU visible, and publish ports `7421` and `7422` yourself. The pool does not ship a Windows-native app.
- **No chat-API fakery.** Neither engine has a stock call that executes a layer range on external hidden states and writes KV to this store. The adapters locate decoder modules and run the range. If the engine cannot be hooked, startup fails. It does not fall back to `LLM.generate`, `LLM.chat`, or `/v1/chat/completions`.
- **Not zero-knowledge.** Windows does not see plaintext, but hidden states are a function of the prompt. Possession of the states plus the same weights can still leak information.
- **Memory.** vLLM may allocate its own paged cache for the full model at load even though the pool executes only one half and stores KV outside the engine. Drop unused depth only after load; a 16 GB Mac can still run out of memory.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Required coverage without a GPU:

- `tests/test_dispatcher_mock_loop.py` — full dual mock, no plaintext on the Windows payload, `degraded == ""`
- `tests/test_dispatcher_single_mac_fallback.py` — Windows down or `single-mac` uses only Mac `[0, 14)`, local KV, `degraded == "mac_front_half"`; `mode=dual` with Windows down fails; a mid-session drop fails and cleans KV

Real vllm-metal / CUDA tests are skipped or fail closed when those stacks are absent. No model weights are downloaded by the default suite.
