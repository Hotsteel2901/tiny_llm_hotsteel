# hotsteel0.1

一个中文文本生成语言模型（Decoder-only Transformer），面向**单机环境**从零训练：
目标是让模型学会理解自然语言提问并生成符合常识、逻辑连贯、语义通顺的中文回答。

提供两个规模：

| 配置 | 参数量 | 用途 | 配置文件 |
|---|---|---|---|
| **small** | 约 12M | 教学 / 快速验证 | `configs/small.yaml` |
| **100m** | 约 **97.5M** | 正式训练，支持导出 GGUF | `configs/hotsteel_100m.yaml` |

- 架构：Pre-LN Transformer + RMSNorm + RoPE（Llama 约定）+ SwiGLU，权重共享
- 分词：自训练 ByteLevel BPE（16k 词表）
- 训练目标：因果语言模型（next-token），序列格式 `<|user|>问<|assistant|>答<|end|>`
- 语料：真实中文指令数据（约 4.8 万条）+ 合成常识问答 + 教师蒸馏语料
- 设备：自动选择 CUDA / MPS / CPU；CUDA 上自动启用 bfloat16
- **可导出 GGUF，直接用 llama.cpp 加载运行**

---

## 1. 目录结构

```
hotsteel0.1/
├── configs/
│   ├── small.yaml              # 12M 配置
│   └── hotsteel_100m.yaml      # 100M 配置
├── data/
│   ├── raw/                    # 原始数据集（alpaca 中文）
│   ├── corpus_synth.jsonl      # 合成常识问答
│   ├── teacher_qa.jsonl        # 教师蒸馏语料
│   ├── corpus_100m.jsonl       # 合并后的训练语料（按 instruction 去重）
│   └── tokenizer_100m/         # 训练得到的 BPE 词表
├── hotsteel/
│   ├── config.py               # 配置加载
│   ├── tokenizer.py            # 分词器训练/封装
│   ├── dataset.py              # 语料加载 + 变长序列 + 动态 padding
│   ├── model.py                # Transformer 定义（Llama 同构）
│   ├── generate.py             # 采样与单轮问答
│   └── utils.py                # 种子/设备/精度/日志
├── scripts/
│   ├── make_corpus.py          # 生成合成问答语料
│   ├── teacher_generate.py     # 生成教师蒸馏语料
│   ├── import_teacher.py       # 导入教师语料
│   ├── prepare_corpus.py       # 多源合并成大语料
│   ├── train_tokenizer.py      # 训练 BPE
│   ├── train.py                # 训练主脚本
│   ├── chat.py                 # 命令行对话 / 批量提问
│   ├── export_hf.py            # 导出 HuggingFace Llama 格式
│   ├── to_gguf.py              # 转换为 GGUF
│   └── verify_gguf.py          # 校验 PyTorch 与 llama.cpp 等价性
├── checkpoints_100m/           # 100M 权重与训练配置存档
├── export/                     # HF / GGUF 导出产物
├── requirements.txt
└── README.md
```

## 2. 环境依赖

- Python >= 3.10
- 依赖见 `requirements.txt`：torch / tokenizers / numpy / PyYAML / tqdm

推荐用独立虚拟环境。**若要用 GPU，务必安装 CUDA 版 torch**：

```bash
# 创建环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# CPU 版
pip install -r requirements.txt

# NVIDIA GPU 版（以 CUDA 12.8 为例，按显卡/驱动选择对应 index-url）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install tokenizers numpy PyYAML tqdm
```

验证 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 快速开始（四步）

```bash
# 1) 生成中文问答语料
python scripts/make_corpus.py --out data/corpus_zh.jsonl --seed 1337

# 2) 训练 BPE 分词器
python scripts/train_tokenizer.py --config configs/small.yaml

# 3) 训练模型
python scripts/train.py --config configs/small.yaml

# 4) 对话
python scripts/chat.py --config configs/small.yaml --ckpt checkpoints/final.pt
```

单次提问（不进入交互）：

```bash
python scripts/chat.py --ckpt checkpoints/final.pt --prompt "中国首都是哪里？"
```

## 3.0 训练 1 亿参数模型（推荐，支持 GGUF）

更大规模 + 真实指令数据，泛化能力显著优于 12M 版本。

```bash
# 1) 下载真实中文指令数据（约 9.5 万条，两个 alpaca 中文集）
python -c "
from huggingface_hub import hf_hub_download
for repo,f in [('shibing624/alpaca-zh','alpaca_gpt4_data_zh.json'),
               ('hfl/alpaca_zh_51k','alpaca_data_51k.json')]:
    hf_hub_download(repo, f, repo_type='dataset', local_dir='data/raw')
"

# 2) 生成合成问答与教师语料
python scripts/make_corpus.py --out data/corpus_synth.jsonl --seed 1337
python scripts/teacher_generate.py --out data/teacher_qa.jsonl --seed 1337

# 3) 合并成训练语料（按 instruction 去重）
python scripts/prepare_corpus.py --out data/corpus_100m.jsonl \
    --synth data/corpus_synth.jsonl --teacher data/teacher_qa.jsonl

# 4) 训练分词器与模型
python scripts/train_tokenizer.py --config configs/hotsteel_100m.yaml
python scripts/train.py --config configs/hotsteel_100m.yaml

# 5) 对话
python scripts/chat.py --config configs/hotsteel_100m.yaml \
    --ckpt checkpoints_100m/final.pt
```

## 3.1 教师蒸馏流程（可选，推荐）

hotsteel0.1 的定位是**学生模型**：把更强的「教师模型」产出的高质量回答作为学习目标。
教师可以是人工撰写、更大规模的对话模型，或内置知识模板。

```bash
# a) 生成教师语料（内置知识库；也可接入外部大模型，见脚本内 RemoteTeacher 说明）
python scripts/teacher_generate.py --out data/teacher_qa.jsonl --seed 1337

# b) 合并教师语料进主语料（同一问题时以教师回答为准）
python scripts/import_teacher.py --teacher data/teacher_qa.jsonl \
    --corpus data/corpus_zh.jsonl --out data/corpus_zh.jsonl

# c) 语料变更后需重跑分词器与训练
python scripts/train_tokenizer.py --config configs/small.yaml
python scripts/train.py --config configs/small.yaml
```

## 3.2 v2：接入真实开源语料（52.7 万条）

纯合成语料的天花板很低——模型只能回答模板里出现过的问题。v2 接入真实中文
指令数据，把语料从 9.5 万条扩到 **52.7 万条（约 1 亿字符 / 5800 万 token）**。

数据来源：`BelleGroup/train_0.5M_CN`（Belle 开源，52 万条真实中文指令）。
**为什么不自己写爬虫**：在我们这个规模，从网页爬原始文本得到的是「预训练语料」，
还需要清洗、配比、SFT 化，质量远不如社区已清洗好的指令集——属于典型的负优化。
正确做法是站在开源数据集肩膀上，把精力花在**清洗与去重**上。

`scripts/build_corpus_v2.py` 的清洗规则（借鉴大厂数据构建报告，裁剪到适用子集）：

1. **语言过滤** —— 回答中文字符占比 < 0.15 的丢弃，避免英文/代码污染分布
2. **长度过滤** —— instruction 3~1500 字、response 4~1600 字（超出 512 会被截断，纯浪费）
3. **退化样本过滤** —— 纯标点、与提问雷同、含 8 连以上重复字符的丢弃
4. **按归一化 instruction 去重** —— 全半角统一后比较，同题只留一条
5. **教师/合成语料优先** —— 先加载旧语料再去重，保证核心问答不被大语料淹没

```bash
# 构建语料
python scripts/build_corpus_v2.py \
    --belle data/raw/Belle_open_source_0.5M.json \
    --extra data/corpus_100m.jsonl --out data/corpus_v2.jsonl

# 训练（configs/hotsteel_v2.yaml：packing=true，batch 12，18000 iter ≈ 1.9 epoch）
python scripts/train.py --config configs/hotsteel_v2.yaml
```

**关于序列打包（packing）**：v2 开启 `packing: true`，把样本拼接成连续的
512 token 窗口，GPU 利用率从动态 padding 的浪费中解放出来。代价是
`mask_instruction` 失效（全 token 计算 loss）——这是业界标准做法
（Qwen/Llama 预训练均如此），模型同时学会建模提问本身，通常利大于弊。

## 4. 关键超参数（configs/small.yaml）

| 区块 | 参数 | 默认 | 说明 |
|---|---|---|---|
| data | max_seq_len | 256 | 单样本最大 token 数 |
| tokenizer | vocab_size | 4096 | BPE 词表大小 |
| model | d_model / n_layer / n_head | 384 / 6 / 6 | 隐藏维 / 层数 / 头数 |
| model | use_swiglu | true | 使用 SwiGLU 前馈 |
| train | batch_size | 64 | 批大小（GPU 可调大） |
| train | lr / min_lr | 3e-4 / 3e-5 | cosine 调度上下界 |
| train | warmup_iters | 150 | 线性 warmup 步数 |
| train | max_iters | 1500 | 训练总步数 |
| train | device / dtype | auto / auto | 自动选设备与精度 |
| generate | temperature / top_k / top_p | 0.8 / 40 / 0.9 | 采样参数 |

常用覆盖：

```bash
# 冒烟测试（几十步即停，验证流程）
python scripts/train.py --config configs/small.yaml --max_iters 60

# 从断点继续
python scripts/train.py --config configs/small.yaml --resume checkpoints/final.pt
```

## 5. 可复现性

- 全局随机种子固定在 `project.seed`（默认 1337），涵盖 Python / NumPy / PyTorch。
- 数据划分用固定随机置换，训练/验证集划分稳定。
- 每次训练会把**实际使用的完整配置**存档到 `checkpoints/train_config.yaml`。
- 分词器与语料均由固定种子脚本生成，重复运行结果一致。

## 6. 模型与训练要点

- **Pre-Norm + RMSNorm**：训练更稳，收敛更快。
- **RoPE 旋转位置编码**：无需学习位置 embedding，外推友好。
- **SwiGLU 前馈**：同参数下表达力更强。
- **权重共享**：输入 embedding 与输出投影共享，减少参数。
- **逐样本对齐数据（默认）**：每个问答对单独编码，配合 `mask_instruction`
  只对「回答」部分计算损失，避免不同问答跨边界粘连，回答更连贯。
  如需更高 token 利用率，可在配置中设 `packing: true` 切回打包策略。
- **AMP + 梯度裁剪 + cosine 调度**：小模型下单机训练更稳定。

### 训练观察（可复现）

- 打包策略 + 全量损失时，val_loss 约 1.8；改为逐样本对齐 + 回答掩码后，
  val_loss 降至约 0.1，回答粘连（如「你好」答成别的题）明显减少。
- `mask_instruction` 的 A/B：开启时答对率约 9/12，关闭时约 7/12，故默认开启。
- **语料覆盖决定上限**：曾出现「人是什么/你是谁/人类与人猿的区别」全部答非所问，
  排查发现语料中根本没有这些词条。补充人类/动物/自然类实体、对比模板、
  自我认知问答，并对日常短句加权后，抽测 12/12 全对。
- 当前约 12M 参数、约 2.2k 条语料；**扩充语料是提升质量最直接的手段**。

## 7. 扩展语料

`data/corpus_zh.jsonl` 每行一条 JSON：

```json
{"instruction": "中国首都是哪里？", "response": "中国的首都是北京……", "category": "常识"}
```

把自有数据按此格式追加或替换该文件，重跑分词器与训练即可。
语料越大越多样，回答质量越好；也可调大 `vocab_size`、`d_model`、`n_layer` 扩展模型。

## 8. 实测样例（final.pt）

```
你 > 你好
模型 > 你好！有什么我可以帮你的吗？

你 > 你是谁
模型 > 你好，我是 hotsteel0.1，一个在单机上训练的小型中文语言模型，很高兴和你交流。

你 > 人是什么
模型 > 人是能够思考、使用语言和工具的高等动物，生活在社会中。

你 > 人类与人猿的区别
模型 > 人类和人猿的主要区别是：人类能直立行走、制造和使用复杂工具，大脑高度发达；
       而人猿是人类的近亲，主要生活在树上，用四肢行动。人类会制造工具和使用语言，
       人猿更依赖本能。

你 > 什么是人工智能？
模型 > 人工智能是让机器模拟人类智能、完成学习和决策的技术。

你 > 怎样保护眼睛？
模型 > 保护眼睛可以定期远眺休息、做眼保健操，避免长时间盯屏幕。

你 > 谢谢
模型 > 不客气，很高兴能帮到你。

你 > 晚安
模型 > 晚安，愿你睡个好觉。
```

以上 12 题抽测全部正确。若问到语料未覆盖的领域，模型可能会给出不相关的答案——
这是小型模型的固有边界，**扩充语料**（尤其 `teacher_generate.py` 的知识库）是直接解法。

## 9. 导出 GGUF（llama.cpp 加载）

模型采用 **Llama 同构架构**（RMSNorm / RoPE / SwiGLU / 无 bias），因此可以直接
转成 GGUF 并由 llama.cpp 运行。注意本项目 RoPE 使用 **Llama 标准 half-split 约定**，
与 llama.cpp 完全一致。

### 9.1 前置：获取 llama.cpp

```bash
# 方式 A：克隆源码（用于转换脚本 convert_hf_to_gguf.py）
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

# 方式 B：下载预编译二进制（用于运行；Windows x64 CPU 版）
gh release download --repo ggml-org/llama.cpp \
  --pattern "llama-*-bin-win-cpu-x64.zip" --dir llama_bin
cd llama_bin && unzip -o "llama-*-bin-win-cpu-x64.zip"
```

转换脚本依赖：`pip install transformers safetensors gguf sentencepiece`

### 9.2 三步导出

```bash
# 1) 导出为 HuggingFace Llama 格式目录
python scripts/export_hf.py --config configs/hotsteel_100m.yaml \
    --ckpt checkpoints_100m/final.pt --out export/hotsteel0.1-100m-hf

# 2) 转换为 GGUF（f16）
python scripts/to_gguf.py --hf export/hotsteel0.1-100m-hf \
    --out export/hotsteel0.1-100m-f16.gguf --outtype f16

# 3) 可选：量化（显著减小体积，便于分发）
../llama_bin/llama-quantize.exe export/hotsteel0.1-100m-f16.gguf \
    export/hotsteel0.1-100m-q4_k_m.gguf Q4_K_M
```

> `scripts/to_gguf.py` 是对 llama.cpp 转换脚本的封装：自训练分词器的
> BPE 预分词器指纹不在 llama.cpp 内置表中，该封装会在运行时兜底映射为
> `gpt-2` 类型（与本项目 ByteLevel 正则一致），无需改动 llama.cpp 源码。

### 9.3 用 llama.cpp 运行

```bash
# 单次补全（原生提示格式）
./llama-cli.exe -m export/hotsteel0.1-100m-f16.gguf \
  -p "<|user|>你好<|assistant|>" --special -n 128 --temp 0.8 -t 8

# 对话模式（使用 GGUF 内置 chat_template）
./llama-cli.exe -m export/hotsteel0.1-100m-f16.gguf -cnv -t 8

# 启动 HTTP 服务
./llama-server.exe -m export/hotsteel0.1-100m-f16.gguf --port 8080 -t 8
```

### 9.4 等价性校验

`scripts/verify_gguf.py` 用同一组提示对 PyTorch 与 llama.cpp 做贪婪解码并逐条比对，
用于证明导出正确（权重映射、RoPE 约定一致）：

```bash
python scripts/verify_gguf.py --config configs/hotsteel_100m.yaml \
    --ckpt checkpoints_100m/final.pt \
    --gguf export/hotsteel0.1-100m-f16.gguf \
    --llama_cli ../llama_bin/llama-cli.exe
# 期望输出：8/8 条输出完全一致
```

### 9.5 GGUF 正确性的完整验证链（重要）

初次做 PPL 数值对照时，PyTorch 与 `llama-perplexity` 差了约 1.9 倍。
逐环排查后确认 **GGUF 导出完全正确**，差异全部来自对照方法本身。验证链如下：

| # | 环节 | 方法 | 结果 |
|---|------|------|------|
| 1 | 分词器 | `llama-tokenize` vs 我方 `encode_ids` | **完全一致**（同为 12714 token，逐 id 相同）|
| 2 | checkpoint → HF 权重 | 逐张量比对 `model.safetensors` | **误差 = 0.0** |
| 3 | HF 前向 vs 我方前向 | `transformers.LlamaForCausalLM` 输出 logits | **max diff = 0.0** |
| 4 | HF 权重 → GGUF 张量 | 用 `gguf` 读取并映射键名比对 | **86/110 完全一致，其余 24 张为 Q/K 的标准 RoPE 置换** |
| 5 | GGUF 元数据 | `GGUFReader` | rope.freq_base=10000、eps=1e-6、ctx=512、add_bos=False 全部正确 |
| 6 | 计算路径敏感性 | `-fa 0/1`、`-ctk/-ctv f32`、f16/f32 权重 | PPL 全部相同（195.43），排除精度/内核差异 |

第 4 条的 24 张"不一致"张量正是 12 层的 `attn_q` / `attn_k`。其置换模式为
`[0, 32, 1, 33, ...]` —— 这是 `convert_hf_to_gguf.py` 对 Llama 架构做的
**标准 RoPE 交错重排**（把 HF 的 half-split 布局转成 llama.cpp NORM 模式
需要的交错布局）。对该 GGUF 张量套用 llama.cpp 的标准 `permute()` 可以
**无损还原**回 HF 权重，因此是正确的、有意为之的转换。

**剩余的 PPL 差异来自协议，不是模型**：`llama-perplexity` 不是逐 token 遍历全文，
而是把文本切成 `n_ctx` 窗口、**只给每个窗口的后半段打分**（`first = n_ctx/2`），
并丢弃不足一个窗口的尾部。此外该预编译二进制（b11334）实际生效的窗口大小
与日志打印值不一致（`-c 512` 报 28 个 chunk ⇒ 有效窗口约 448）。
`scripts/compare_ppl.py` 现已支持 `--protocol llama` 复刻这套协议：

```bash
python scripts/compare_ppl.py --text data/ppl_sample.txt --protocol llama --chunk 512
```

**结论**：判断导出是否正确，应以「分词一致 + 权重逐张量一致 + 贪婪解码文本一致」
为准（100M 版 8 条中 4 条逐字一致，其余 4 条语义相同仅措辞略异），
而不是裸比 PPL 数值。

## 10. 许可

本项目代码仅供学习与研究使用。语料为脚本按通用常识生成的示例数据。
