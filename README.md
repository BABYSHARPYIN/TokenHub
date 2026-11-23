# TokenHub

TokenHub 是共享分层推理计算池，仓库名是 BABYSHARPYIN/TokenHub。安装用的发行名是 `tokenhub`。Python 导入包是 `pool`，启动命令是 `python -m pool`。

默认且唯一的演示模型是 Hugging Face safetensors 检查点 **Qwen/Qwen3-VL-8B-Instruct**（`Qwen3VLForConditionalGeneration`）。这不是 GGUF 仓库 `Qwen/Qwen3-VL-8B-Instruct-GGUF`，也没有 llama.cpp 路径。

语言塔（text tower）共 36 层。每一跳只跑调度器刚分配的一段连续层，跑完把隐状态交回调度器，再要下一跳。下一跳可以是另一台机器，也可以还是同一进程的下一段。两台能力相近的节点常常得到 `[0, 18)` 和 `[18, 36)`，这是调度结果，不是配置文件里的固定切分。语言塔宽度按官方 `text_config` 锁死：`hidden_size` 4096，`num_attention_heads` 32，`num_key_value_heads` 8，`head_dim` 128，`intermediate_size` 12288，`vocab_size` 151936，`eos_token_id` 151645。

视觉塔留在 Mac，和分词器、明文、词嵌入、`lm_head` 在一起。它是深度 27 的 ViT，DeepStack 索引为 8、16、24。DeepStack 索引属于视觉塔，不是语言塔的切分点。Windows 不跑视觉编码，也不保留视觉模块。任何模式（`auto`、`single-mac`、`dual`）都如此。

分词、明文、词嵌入、`lm_head`、采样，以及答案和思维链组装，都只留在 Mac。Windows 不得分词、解码，也不得输出明文。

隐状态和 KV 通过明文 HTTP 传输。第一版不做加密、不计费，也不做多租户鉴权。

浏览器只是控制台。各端本机的轻量代理负责真正的计算。浏览器不承担分层解码。Mac 引擎是 vllm-metal，Windows / WSL2 引擎是 CUDA vLLM，它们只是引擎，不是整套平台。`HiddenStateRunner` 钩住解码器内部，按语言塔层区间做前向，不会调用 `vllm` 的 chat、`LLM.generate`，也不会走 OpenAI 风格的 chat completions。

默认 `pytest` 使用 mock 引擎，不下载多 GB 权重，也不需要 GPU。

## 1. Mac 单机前半层冒烟

请先跑这条路径，不需要 Windows。

mock 引擎（不加载权重，用来验证控制台和前半层降级）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pool --role mac --engine mock --host 127.0.0.1 --port 7410 --mode auto
```

打开 http://127.0.0.1:7410 。模式保持 `auto`，或改成 `single-mac`，然后提交一个问题。

未配置 Windows 时，会话会走前半层冒烟：

- `degraded` 为 `mac_front_half`
- 控制台会标明这是前半层冒烟，输出质量不作数
- Prefill 和 decode 都只跑 Mac 的语言塔 `[0, 18)`
- KV 写在 Mac 本机内存里
- 采样是把 `lm_head` 用在这些前半层隐状态上
- 不会跑 `[18, 36)`，也不会为了把句子写顺而在 Mac 上跑满 36 层语言塔
- 不会调用完整模型的 chat API
- 若请求里带图像，视觉编码仍只在 Mac 上做

mock 引擎产出的 token 只是 UTF-8 占位，不是 Qwen 词表。上面的提示同样适用。

真实权重冒烟（Apple Silicon、macOS 15 或更新，并已安装 vllm-metal）：

```bash
pip install -e ".[metal]"
python -m pool --role mac --engine vllm-metal --model Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 7410 --mode auto --max-model-len 512
```

控制台约定相同：Windows 不可达时，`auto` 和 `single-mac` 都只跑 `[0, 18)`。不要指望这条路径给出完整 36 层语言塔的通顺回答。如果 vLLM 还要给自己分配缓存，16 GB 的 Mac 会比较紧；`--max-model-len 512` 可以把那份缓存压小。计算池实际执行的仍然只是语言塔前半层，视觉塔留在本机。

`mode=dual` 且 Windows 不通时，会话直接失败，不会悄悄改成前半层。

## 2. Mac + Windows/WSL2 双机对半

Windows 的 Node 和 KV 跑在 WSL2 里，以便 CUDA vLLM 能用到 NVIDIA 显卡。请自行把两个明文端口从 WSL2 映射到 Windows 宿主机的公网地址（示例：KV 用 `7421`，层节点用 `7422`）。第一版没有隧道加密。

在 Windows / WSL2 上：

```bash
python -m pool --role windows --engine vllm-cuda \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 --kv-port 7421 --node-port 7422 \
  --max-model-len 512
```

如果 GPU 栈还没就绪，可以先把 `--engine vllm-cuda` 换成 `--engine mock`，用 mock 跑 Windows 这一半。

在 Mac 上：

```bash
python -m pool --role mac --engine vllm-metal \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 7410 --mode auto \
  --windows-kv http://WINDOWS_PUBLIC_IP:7421 \
  --windows-node http://WINDOWS_PUBLIC_IP:7422 \
  --health-timeout 3 \
  --max-model-len 512
```

`configs/mac.example.json` 和 `configs/windows.example.json` 写了端口、模型、语言塔宽度、视觉塔归属，以及 CUDA 的 `offload`（默认 `auto`）。配置里没有层区间。使用 Mac 配置前，请先把 `WINDOWS_PUBLIC_IP` 换成真实地址。如果还留着占位符，会被当成「未配置 Windows」，`auto` 模式会继续走前半层冒烟。

`auto` 在每次会话开始前会做这些事：

1. 探活 KV 和其他节点，预算大约 3 秒。节点上报的是能力（引擎、显存、是否允许 offload），不是一份写死的层表。
2. 如果在线节点的容量够覆盖全部 36 层，就按显存排出若干连续、互不重叠的跳。能力相近的两台机器通常仍是 Mac `[0, 18)`、另一节点 `[18, 36)`，`degraded` 为空。显存更小的机器会分到更短的一段。
3. 每一跳只把当前这段层交给对应节点。节点把隐状态交回调度器，调度器再给出下一跳。Prefill 和 decode 都是这个状态机，不是写死的「先 Mac 再 Windows」两跳。
4. 若请求带图像，视觉编码留在 Mac。词嵌入也在 Mac。语言塔最后一层的隐状态回到 Mac，做最终 norm 和 `lm_head`，然后采样。
5. 其他节点只看到 float32 激活和 KV，看不到问题原文、图像或 token id。

同一台 Mac 可以分两跳核对整座语言塔：先 `[0, 18)`，下一跳再把 `[18, 36)` 分回这台 Mac。这仍然是两次按段前向，不是一次把 36 层塞进同一次引擎调用。这种只在 Mac 上跑完的核对仍然标成输出质量不计。

`mode=single-mac`，或 `auto` 但没有足够节点覆盖 36 层时，Mac 只跑它能托管的前缀（现在是前半 `[0, 18)`），`degraded` 为 `mac_front_half`，控制台写明输出质量不计。`mode=dual` 而节点不够覆盖 36 层时，会话直接失败。

已经按全语言塔开始的会话，若当前这一跳的节点中途掉线，该请求失败并删除 KV。调度器不会在一跳执行到一半时改派。节点加入或离开之后，下一跳边界或下一次会话才会重新分配。

CUDA 节点如果装不下分配到的语言层权重和必要的 KV，会自动设置 vLLM 的 `cpu_offload_gb` 和 `swap_space`，把放不下的部分放进 CPU 内存，并在日志里写明进入 memory-offload、速度会变慢。模型仍是同一份 safetensors，不会改成量化或更少的层。连一层计算加最小工作集都放不进 GPU 时，启动失败。`offload` 可设为 `auto`（默认）、`force` 或 `disable`，`offload_cap_gb` 可限制卸出的 GiB。vllm-metal 使用统一内存，不做这种显存到内存的降级，只打一行说明。

## 3. 风险

- **不做加密。** 层激活和 KV 在公网上明文传输。能读到这些数据包的人就能读到隐状态。第一版不加 TLS、不加鉴权，也不加加密信封。
- **前半层输出质量不作数。** `mac_front_half` 只跑语言塔前缀 `[0, 18)`，再用 `lm_head` 投影这些隐状态。只在 Mac 上分两跳核对整塔时，输出质量同样不计。两种路径都不会把 36 层合成一次引擎调用。
- **视觉塔不跨机。** 27 层 ViT 和 DeepStack 8/16/24 只留在 Mac。Windows 导出语言层之后会卸掉视觉模块。
- **需要 macOS 15 或更新。** Mac 的真实引擎是 vllm-metal（vLLM 的 API/调度 + MLX/Metal）。请使用 macOS 15 或更新系统。mock 引擎在哪里都能跑。
- **需要 WSL2。** Windows 的真实引擎是 CUDA vLLM。请在能看到 NVIDIA 显卡的 WSL2 里运行，并自行放行 `7421` 和 `7422`。本项目不提供 Windows 原生应用。
- **禁止用 chat API 假装分层。** 两个引擎都没有现成接口，能按层区间对外部隐状态做前向，并把 KV 写到本项目的存储里。适配器会定位语言塔解码模块并只跑指定区间。如果钩不住引擎，启动就失败，不会退回到 `LLM.generate`、`LLM.chat` 或 `/v1/chat/completions`。
- **不是零知识。** Windows 看不到明文，但隐状态是由问题计算出来的。同时掌握隐状态和同一套权重时，仍可能泄露信息。
- **显存和内存。** 即便计算池只执行一半语言层、并把 KV 存在引擎之外，vLLM 加载时仍可能按整模分配自己的 paged cache。未使用的层深只能在加载后再裁。16 GB 的 Mac 仍可能内存不够。视觉塔留在 Mac，会再占一份内存。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

无 GPU 时必须覆盖：

- `tests/test_dispatcher_mock_loop.py`：完整双机 mock 路径，发给 Windows 的内容不含明文，`degraded == ""`，Mac 只跑 `[0, 18)`，Windows 只跑 `[18, 36)`
- `tests/test_dispatcher_single_mac_fallback.py`：Windows 不通或 `single-mac` 时，只跑 Mac `[0, 18)`，使用本机 KV，`degraded == "mac_front_half"`；`mode=dual` 且 Windows 不通时会话失败；中途掉线会失败并清理 KV

真实的 vllm-metal / CUDA 测试在缺少对应环境时会跳过，或在加载失败时直接报错，而不会假装成功。默认测试套件不会下载模型权重。
