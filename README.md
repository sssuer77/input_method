# 智能输入法 · 下一词预测

个人学习项目：基于 RNN 的中文输入法词语联想模型。根据用户已输入的文本，预测下一个最可能出现的词语，并返回概率最高的 **Top-5** 候选词。

示例：输入「自然语言」，模型输出类似 `['处理', '理解', '的', '描述', '生成']`。

## 功能概览

- **数据预处理**：对话语料分词、词表构建、滑动窗口构造「上下文 → 下一词」样本
- **模型训练**：Embedding + RNN + Linear，CrossEntropyLoss + Adam，支持 TensorBoard
- **效果评估**：Top-1 / Top-5 准确率
- **交互预测**：命令行累加输入历史并实时联想

## 技术栈

| 组件 | 说明 |
|------|------|
| PyTorch | 模型定义与训练 |
| jieba | 中文分词 |
| pandas / scikit-learn | 数据读写与划分 |
| TensorBoard | 训练损失可视化 |
| torchinfo | 模型结构摘要 |

## 项目结构

```
input_method/
├── data/
│   ├── raw/                  # 原始语料（如 synthesized_.jsonl）
│   └── processed/            # 预处理结果：词表、训练/测试集
├── logs/                     # TensorBoard 日志
├── models/                   # 保存的最优模型 model.pt
├── src/
│   ├── config.py             # 路径与超参数
│   ├── process.py            # 数据预处理
│   ├── tokenizer.py          # jieba 分词与词表编解码
│   ├── dataset.py            # Dataset / DataLoader
│   ├── model.py              # InputMethodModel（Embedding + RNN + Linear）
│   ├── train.py              # 训练
│   ├── evaluate.py           # 评估
│   └── predict.py            # 交互式预测
└── README.md
```

## 数据集

任务数据来自开放对话语料：[Jax-dan/HundredCV-Chat](https://huggingface.co/datasets/Jax-dan/HundredCV-Chat)。

预处理流程（`process.py`）：

1. 从对话中抽取句子
2. 按 8:2 划分训练集 / 测试集
3. 用训练集构建词表（含 `<unk>`）
4. 滑动窗口（长度 `SEQ_LEN=5`）生成 `{input, target}` 样本，写出 JSONL

默认会从原始 jsonl 中采样 `frac=0.1`，以加快实验；可按需修改。

## 模型结构

```
输入词索引 → Embedding(64) → RNN(hidden=128) → 取最后时间步 → Linear(vocab_size) → Softmax / Top-K
```

- **损失**：`CrossEntropyLoss`（下一词多分类）
- **优化器**：Adam，学习率 `1e-3`
- **默认轮数**：30 epoch，按训练 loss 保存最优权重到 `models/model.pt`

## 环境准备

建议 Python 3.10+，在项目根目录安装依赖：

```bash
pip install torch pandas scikit-learn jieba tqdm tensorboard torchinfo
```

将原始语料放到：

```
data/raw/synthesized_.jsonl
```

## 使用方法

所有脚本在 `src/` 目录下运行：

```bash
cd src
```

### 1. 数据预处理

```bash
python process.py
```

生成：

- `data/processed/vocab.txt`
- `data/processed/indexed_train.jsonl`
- `data/processed/indexed_test.jsonl`

### 2. 训练

```bash
python train.py
```

训练过程写入 `logs/`，可用 TensorBoard 查看：

```bash
tensorboard --logdir ../logs
```

### 3. 评估

```bash
python evaluate.py
```

输出 Top-1 / Top-5 准确率。

### 4. 交互预测

```bash
python predict.py
```

按提示输入词语，输入 `q` 或 `quit` 退出。

## 超参数（`src/config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SEQ_LEN` | 5 | 输入上下文长度 |
| `BATCH_SIZE` | 64 | 批大小 |
| `EMBEDDING_DIM` | 64 | 词向量维度 |
| `HIDDEN_SIZE` | 128 | RNN 隐层维度 |
| `LEARNING_RATE` | 1e-3 | 学习率 |
| `EPOCHS` | 30 | 训练轮数 |

## 说明

本仓库为个人学习与实践项目，用于理解「下一词预测」、RNN 语言模型、分词词表与训练评测流水线；非生产级输入法产品。
