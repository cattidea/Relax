<div align="center">

# Relax

**Reinforcement learning for text, vision, audio, and agents.**

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

**Relax** is an RL post-training framework from the Xiaohongshu AI Infra Team. It uses Megatron-LM for training, SGLang for inference, and Ray Serve to manage services.

## ✨ Highlights

- **Multimodal training.** Train models such as Qwen3-Omni on text, images, video, and audio.
- **Async training.** Generate samples while training runs. [TransferQueue](https://github.com/redai-studio/TransferQueue) moves data between the two.
- **Agentic RL.** Train agents with tools, environment feedback, and multi-turn or multi-agent interactions. See the [Agentic guide](docs/en/guide/agentic-rollout.md).
- **Algorithms and rewards.** Use built-in RL algorithms and on-policy distillation. Add [custom rewards](docs/en/guide/customize-training.md) or use a model to score responses with [GenRM](docs/en/examples/generative-reward-model.md).
- **Elastic rollout.** Add or remove inference engines during training. See the [scaling guide](docs/en/guide/elastic-rollout.md).

## 🚀 Quick Start

This example trains **Qwen3-4B with GRPO on one node with 8 GPUs**. Check the [installation guide](docs/en/guide/installation.md) for GPU and driver requirements.

### 1. Start a training container

The Docker image includes the training and inference dependencies. Replace `/path/to/workspace` with a directory on the host to store code, models, and data.

```bash
docker pull ghcr.io/redai-studio/relaxrl:latest
docker run -it --gpus all --ipc=host --network=host \
  -v /path/to/workspace:/workspace \
  ghcr.io/redai-studio/relaxrl:latest bash
```

### 2. Install Relax inside the container

```bash
git clone https://github.com/redai-studio/Relax.git /workspace/Relax
cd /workspace/Relax
pip install -e .
```

### 3. Download the model and data, then start training

The script uses DAPO Math for training and AIME 2024 for evaluation. ClearML runs in offline mode in this example.

```bash
hf download Qwen/Qwen3-4B --local-dir /workspace/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /workspace/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /workspace/aime-2024
python scripts/tools/process_aime.py --input /workspace/aime-2024/aime-2024.jsonl

export EXP_DIR=/workspace
export CLEARML_OFFLINE_MODE=1
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

For image, video, and audio data preparation, see the [full tutorial](docs/en/guide/quick-start.md). To change the model, reward, or training settings, see [Customize Training](docs/en/guide/customize-training.md).

## 🏗️ Architecture

<div align="center">
  <img src="./assets/arch.png" width="80%" alt="Relax Architecture">
</div>

Choose how training and sample generation (rollout) share GPUs:

| Mode            | GPU use                                                                            |
| :-------------- | :--------------------------------------------------------------------------------- |
| **Colocate**    | Training and rollout take turns on the same GPUs.                                  |
| **Fully Async** | Training and rollout use separate GPUs. Auxiliary services run separately.         |
| **Hybrid**      | Training and rollout use separate GPUs. Auxiliary work stays on the training GPUs. |

See [Architecture](docs/en/guide/architecture.md), [Fully Async Training](docs/en/guide/fully-async-training.md), and [Hybrid Training](docs/en/guide/hybrid-training.md) for setup and tradeoffs.

## 🤖 Supported Models and Algorithms

| Model family                                                    | Example sizes                              | Modality              |
| :-------------------------------------------------------------- | :----------------------------------------- | :-------------------- |
| **Qwen3**                                                       | 4B, 30B-A3B (MoE)                          | Text                  |
| **Qwen3-VL**                                                    | 4B, 30B-A3B                                | Vision + Language     |
| **Qwen3.5**                                                     | 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B | Text + Vision         |
| **Qwen3-Omni**                                                  | 30B-A3B                                    | Text + Vision + Audio |
| **Qwen3.6**                                                     | 27B, 35B-A3B                               | Text + Vision         |
| **Qwen3.8**                                                     | 27B                                        | Text + Vision         |
| **GLM5**                                                        | 744B-A40B (MoE)                            | Text                  |
| **Kimi K2.6**                                                   | ~1T-A32B (MoE)                             | Vision + Language     |
| **[dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)** | 3B                                         | Vision + Language     |

Browse [model configurations](scripts/models/) and [training scripts](scripts/training/) for available setups. To add an architecture, see [Model Integration](docs/en/guide/external-model-integration.md).

**Algorithms:** PPO, GRPO, M2PO, RLOO, REINFORCE++, REINFORCE++-baseline, GSPO, SAPO, CISPO, and On-Policy Distillation. The [algorithm reference](docs/en/examples/algorithms.md) covers objectives, recipes, and execution-mode limits.

## 📚 Documentation and Examples

Read the [full documentation](https://redai-studio.github.io/Relax/en/) or go directly to:

- **Configure a run:** [Training settings](docs/en/guide/configuration.md) · [LoRA](docs/en/guide/low-rank-adaptation-training.md)
- **Monitor a run:** [Metrics](docs/en/guide/metrics-service-detailed.md) · [Failure recovery](docs/en/guide/health-check-manager.md) · [Notifications](docs/en/guide/notification-system.md)
- **Export a model:** [Checkpoint conversion](docs/en/guide/model-conversion.md)

| Example                                                    | Task                                             |
| :--------------------------------------------------------- | :----------------------------------------------- |
| [Algorithm recipes](examples/algorithms/)                  | RLOO, REINFORCE++, CISPO, and related algorithms |
| [DeepEyes](examples/deepeyes/)                             | Vision-language RL with Qwen3-VL                 |
| [Search-R1](examples/search_r1/)                           | Search with one or more agents                   |
| [Mini-SWE-Agent](examples/mini_swe_agent/)                 | Software engineering tasks                       |
| [NeMo Gym](examples/nemo_gym_agentic/)                     | Agent training with Gym environments             |
| [On-Policy Distillation](examples/on_policy_distillation/) | Train a student model from a teacher             |

## 🧩 Projects Using Relax

| Project                                                  | Description                                                                                  |
| :------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| [HyperEyes](https://github.com/DeepExperience/HyperEyes) | A multimodal search agent that searches for multiple entities in parallel.                   |
| [Iris](https://github.com/AllSpark-Research/Iris)        | Open-weight search agents trained from Qwen3.5/3.6 for tasks that require many search steps. |

## 🤝 Contributing

Start with the [contributing guide](docs/en/guide/how-to-contribute.md). The [skills directory](skills/) has workflows for development, debugging, review, and documentation.

## 📢 News

<details>
<summary>Project updates</summary>

- **2026-08-20:** Added M2PO, RLOO, and two REINFORCE++ variants. See the [algorithm reference](docs/en/examples/algorithms.md).
- **2026-08-19:** Added multi-agent training and the [Search-R1 recipe](examples/search_r1/).
- **2026-08-18:** Added MoE support for [LoRA training](docs/en/guide/low-rank-adaptation-training.md).
- **2026-05-26:** Added [Hybrid training](docs/en/guide/hybrid-training.md).
- **2026-05-11:** Added Qwen3.6 text and vision-language models.
- **2026-04-15:** Released Relax as open source.

</details>

## 📝 Citation

If you use Relax in your research, please cite:

```bibtex
@software{relax2026,
  title  = {Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale},
  author = {Relax Contributors},
  url    = {https://arxiv.org/abs/2604.11554},
  year   = {2026}
}
```

## 📜 License

Relax is licensed under [Apache 2.0](LICENSE).

## 🙏 Acknowledgements

Thanks to all contributors and to the projects that Relax builds on:

- [Slime](https://github.com/THUDM/slime)
- [SGLang](https://github.com/sgl-project/sglang)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), from NVIDIA
- [TransferQueue](https://github.com/Ascend/TransferQueue)
- [Ray](https://github.com/ray-project/ray)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
