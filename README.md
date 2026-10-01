# hotsteel0.1

一个小型中文文本生成语言模型（Decoder-only Transformer），面向**单机环境**从零训练：
目标是在有限算力下，让模型学会理解自然语言提问并生成符合常识、逻辑连贯、语义通顺的中文回答。

- 参数量：约 **11.4M**
- 架构：Pre-LN Transformer + RMSNorm + RoPE + SwiGLU，权重共享
- 分词：自训练 ByteLevel BPE（中文友好，词表约 2k/4k）
- 训练目标：因果语言模型（next-token），语料为「指令—回答」拼接
- 设备：自动选择 CUDA / MPS / CPU；CUDA 上自动启用 bfloat16

---

## 1. 目录结构

```
hotsteel0.1/
├── configs/
│   └── small.yaml              # 全部超参（模型/训练/数据/推理）
├── data/
│   ├── corpus_zh.jsonl         # 中文问答语料（脚本生成）
│   └── tokenizer/tokenizer.json# 训练得到的 BPE 词表
├── hotsteel/
│   ├── config.py               # 配置加载
│   ├── tokenizer.py            # 分词器训练/封装
│   ├── dataset.py              # 语料加载 + 打包式 Dataset
│   ├── model.py                # Transformer 定义
│   ├── generate.py             # 采样与单轮问答
│   └── utils.py                # 种子/设备/精度/日志
├── scripts/
│   ├── make_corpus.py          # 生成中文问答语料
│   ├── train_tokenizer.py      # 训练 BPE
│   ├── train.py                # 训练主脚本
│   └── chat.py                 # 命令行对话 / 批量提问
├── checkpoints/                # 权重与训练配置存档
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

## 9. 许可

本项目代码仅供学习与研究使用。语料为脚本按通用常识生成的示例数据。
