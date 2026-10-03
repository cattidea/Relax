<div align="center">

# Relax

**面向文本、视觉、音频与智能体的强化学习框架。**

<img src="./assets/Relax.jpg" width="800" alt="Relax">

<p>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  </a>
  <a href="https://arxiv.org/abs/2604.11554">
    <img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red" alt="arXiv">
  </a>
  <a href="https://redai-studio.github.io/Relax">
    <img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation">
  </a>
  <a href="https://github.com/redai-studio/Relax/discussions/48" target="_blank">
    <img src="https://img.shields.io/badge/WeChat-green?logo=wechat" alt="WeChat QR">
  </a>
  <a href="https://github.com/redai-studio/Relax/discussions/30" target="_blank">
    <img src="https://img.shields.io/badge/Docker-Image-blue?logo=docker" alt="Docker Image">
  </a>
</p>

<p>
  <a href="./README.md">📖 English</a> | <a href="./README_zh.md">📖 中文</a>
</p>
</div>

**Relax** 是小红书 AI 平台开源的强化学习后训练框架。它使用 Megatron-LM 训练模型、SGLang 生成样本，并通过 Ray Serve 管理服务。

## ✨ 核心能力

- **多模态训练。** 使用文本、图像、视频和音频训练模型，支持 Qwen3-Omni 等多模态模型。
- **异步训练。** 生成样本与模型训练并行进行，通过 [TransferQueue](https://github.com/redai-studio/TransferQueue) 传递数据。
- **Agentic RL。** 支持工具调用、环境反馈、多轮交互和多智能体训练。参阅 [Agentic 指南](docs/zh/guide/agentic-rollout.md)。
- **算法与奖励。** 使用内置 RL 算法和 On-Policy Distillation，添加[自定义奖励](docs/zh/guide/customize-training.md)，或通过 [GenRM](docs/zh/examples/generative-reward-model.md) 让模型为回答打分。
- **Rollout 弹性扩缩容。** 在训练过程中增减推理引擎。参阅[扩缩容指南](docs/zh/guide/elastic-rollout.md)。

## 🚀 快速开始

下面的示例在**单节点 8 卡上使用 GRPO 训练 Qwen3-4B**。GPU 与驱动要求见[安装指南](docs/zh/guide/installation.md)。

### 1. 启动训练容器

Docker 镜像已包含训练与推理依赖。将 `/path/to/workspace` 替换为宿主机目录，用于保存代码、模型和数据。

```bash
docker pull ghcr.io/redai-studio/relaxrl:latest
docker run -it --gpus all --ipc=host --network=host \
  -v /path/to/workspace:/workspace \
  ghcr.io/redai-studio/relaxrl:latest bash
```

### 2. 在容器内安装 Relax

```bash
git clone https://github.com/redai-studio/Relax.git /workspace/Relax
cd /workspace/Relax
pip install -e .
```

### 3. 下载模型与数据，启动训练

脚本使用 DAPO Math 训练集和 AIME 2024 评估集。本例将 ClearML 设为离线模式。

```bash
hf download Qwen/Qwen3-4B --local-dir /workspace/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /workspace/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /workspace/aime-2024
python scripts/tools/process_aime.py --input /workspace/aime-2024/aime-2024.jsonl

export EXP_DIR=/workspace
export CLEARML_OFFLINE_MODE=1
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

图像、视频和音频的数据准备步骤见[完整教程](docs/zh/guide/quick-start.md)。更换模型、奖励函数或训练参数，请参阅[自定义训练](docs/zh/guide/customize-training.md)。

## 🏗️ 系统架构

<div align="center">
  <img src="./assets/arch.png" width="80%" alt="Relax 架构图">
</div>

三种执行模式决定训练和样本生成（Rollout）如何分配 GPU：

| 模式                      | GPU 使用方式                                               |
| :------------------------ | :--------------------------------------------------------- |
| **Colocate（同步）**      | 训练与 Rollout 轮流使用同一组 GPU。                        |
| **Fully Async（全异步）** | 训练与 Rollout 使用不同的 GPU，辅助服务独立运行。          |
| **Hybrid（混合）**        | 训练与 Rollout 使用不同的 GPU；辅助计算在训练 GPU 上完成。 |

各模式的配置与取舍见[架构指南](docs/zh/guide/architecture.md)、[全异步训练](docs/zh/guide/fully-async-training.md)和 [Hybrid 训练](docs/zh/guide/hybrid-training.md)。

## 🤖 支持的模型与算法

| 模型系列                                                        | 示例规模                                   | 模态               |
| :-------------------------------------------------------------- | :----------------------------------------- | :----------------- |
| **Qwen3**                                                       | 4B, 30B-A3B (MoE)                          | 文本               |
| **Qwen3-VL**                                                    | 4B, 30B-A3B                                | 视觉 + 语言        |
| **Qwen3.5**                                                     | 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B | 文本 + 视觉        |
| **Qwen3-Omni**                                                  | 30B-A3B                                    | 文本 + 视觉 + 音频 |
| **Qwen3.6**                                                     | 27B, 35B-A3B                               | 文本 + 视觉        |
| **Qwen3.8**                                                     | 27B                                        | 文本 + 视觉        |
| **GLM5**                                                        | 744B-A40B (MoE)                            | 文本               |
| **Kimi K2.6**                                                   | ~1T-A32B (MoE)                             | 视觉 + 语言        |
| **[dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)** | 3B                                         | 视觉 + 语言        |

可用配置见[模型配置](scripts/models/)与[训练脚本](scripts/training/)。新增模型架构请参阅[模型接入指南](docs/zh/guide/external-model-integration.md)。

**算法：** PPO、GRPO、M2PO、RLOO、REINFORCE++、REINFORCE++-baseline、GSPO、SAPO、CISPO 和 On-Policy Distillation。目标函数、训练配方与执行模式限制见[算法参考](docs/zh/examples/algorithms.md)。

## 📚 文档与示例

访问[完整文档](https://redai-studio.github.io/Relax/zh/)，或直接查看：

- **配置训练：** [训练参数](docs/zh/guide/configuration.md) · [LoRA](docs/zh/guide/low-rank-adaptation-training.md)
- **监控训练：** [指标](docs/zh/guide/metrics-service-detailed.md) · [故障恢复](docs/zh/guide/health-check-manager.md) · [通知](docs/zh/guide/notification-system.md)
- **导出模型：** [Checkpoint 转换](docs/zh/guide/model-conversion.md)

| 示例                                                       | 任务                               |
| :--------------------------------------------------------- | :--------------------------------- |
| [算法配方](examples/algorithms/)                           | RLOO、REINFORCE++、CISPO 等算法    |
| [DeepEyes](examples/deepeyes/)                             | 使用 Qwen3-VL 进行视觉语言 RL 训练 |
| [Search-R1](examples/search_r1/)                           | 单智能体与多智能体搜索             |
| [Mini-SWE-Agent](examples/mini_swe_agent/)                 | 软件工程任务                       |
| [NeMo Gym](examples/nemo_gym_agentic/)                     | 在 Gym 环境中训练智能体            |
| [On-Policy Distillation](examples/on_policy_distillation/) | 通过教师模型训练学生模型           |

## 🧩 使用 Relax 的项目

| 项目                                                     | 描述                                                                |
| :------------------------------------------------------- | :------------------------------------------------------------------ |
| [HyperEyes](https://github.com/DeepExperience/HyperEyes) | 并行搜索多个实体的多模态搜索智能体。                                |
| [Iris](https://github.com/AllSpark-Research/Iris)        | 基于 Qwen3.5/3.6 训练的开放权重搜索智能体，用于需要多步搜索的任务。 |

## 🤝 参与贡献

从[贡献指南](docs/zh/guide/how-to-contribute.md)开始。[Skills 目录](skills/)提供开发、调试、审查与文档编写流程。

## 📢 项目动态

<details>
<summary>展开更新记录</summary>

- **2026-08-20：** 新增 M2PO、RLOO 与两种 REINFORCE++ 变体，参阅[算法参考](docs/zh/examples/algorithms.md)。
- **2026-08-19：** 新增多智能体训练与 [Search-R1 配方](examples/search_r1/)。
- **2026-08-18：** [LoRA 训练](docs/zh/guide/low-rank-adaptation-training.md)新增 MoE 模型支持。
- **2026-05-26：** 新增 [Hybrid 训练](docs/zh/guide/hybrid-training.md)。
- **2026-05-11：** 新增 Qwen3.6 文本与视觉语言模型支持。
- **2026-04-15：** Relax 正式开源。

</details>

## 📝 引用

如果你在研究中使用 Relax，请引用：

```bibtex
@software{relax2026,
  title  = {Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale},
  author = {Relax Contributors},
  url    = {https://arxiv.org/abs/2604.11554},
  year   = {2026}
}
```

## 📜 许可证

Relax 使用 [Apache 2.0](LICENSE) 许可证。

## 🙏 致谢

感谢所有贡献者，以及 Relax 所依赖的开源项目：

- [Slime](https://github.com/THUDM/slime)
- [SGLang](https://github.com/sgl-project/sglang)
- NVIDIA 的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 与 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [TransferQueue](https://github.com/Ascend/TransferQueue)
- [Ray](https://github.com/ray-project/ray)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
