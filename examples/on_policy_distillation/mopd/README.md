# Multi-Teacher On-Policy Distillation (MOPD)

## 概述

MOPD (Multi-Teacher On-Policy Distillation) 实现多教师在线蒸馏，允许一个学生模型同时从多个异构教师模型中学习。每个教师专注于不同的数据领域（如文本数学、视觉几何），框架按样本的 `data_source` 把它路由到对应教师，请求 token 级 log-probability，实现跨模态、跨领域的知识传递。

教师由 **Relax 托管**：只需通过 `--opd-teacher-routes` 提供 `{data_source: HF 权重路径}`，Relax 会为每个教师自动拉起一个专用 SGLang 服务，运行时注入真实 URL。**无需手动 `sglang.launch_server`、无需管理端口。**

本示例场景：

- **文本教师** (Qwen3-4B-Instruct-2507 / Qwen3.6-27B)：处理 GSM8K / dapo-math-17k 文本数学题
- **视觉教师** (Qwen3-VL-4B-Instruct / Qwen3.5-27B)：处理 Geometry3K / multimodal-open-r1 视觉几何题
- **学生** (Qwen3-VL-2B-Instruct / Qwen3.5-35B-A3B)：在两个领域上同时训练

## MOPD 支持 colocate 模式

MOPD teacher 与 student 的 actor/rollout 共享同一个 GPU 卡池（placement group），二者按 bundle 分区、随训练 lock-step offload/onload：

```
actor 卡池（一整块 placement group，大小 = ACTOR_GPUS）
├── rollout 区   bundle [0, ROLLOUT_GPUS)                     ← student rollout
└── teacher 区   bundle [ROLLOUT_GPUS, ACTOR_GPUS)             ← 各 teacher 依次错位排布
```

- **训练阶段**：actor 用满整块卡池训练 student。
- **rollout 阶段**：student rollout 占前段 bundle，teacher 占后段 bundle，两边都从 offload 状态 onload 回显存。
- **硬约束（框架强制校验）**：`ROLLOUT_GPUS + TEACHER_GPUS == ACTOR_GPUS`，不满足会在启动时直接抛 `ValueError`。
- `--colocate --offload` 必须同时开启；`--rollout-num-gpus` 必须等于 `ROLLOUT_GPUS`。

## 提供的脚本

| 脚本                                        | 规模                           | GPU 分配（actor / rollout / teacher）       |
| ------------------------------------------- | ------------------------------ | ------------------------------------------- |
| `run-mopd-qwen3-vl-2b-8xgpu-colocate.sh`    | 学生 2B，单机 8 卡，教师多副本 | 8 / 4 / 4（2 教师 × 2 副本，各 TP=1）       |
| `run-mopd-qwen35-9b-8xgpu-colocate.sh`      | 学生 9B，单机 8 卡             | 8 / 4 / 4（2 教师各 TP=2，1 副本）          |
| `run-mopd-qwen35-35ba3b-16xgpu-colocate.sh` | 学生 35B-A3B，2 节点 16 卡     | 16 / 8 / 8（TP=4 PP=2 EP=8；2 教师各 TP=4） |

### 8 卡布局（`run-mopd-qwen3-vl-2b-8xgpu-colocate.sh`，教师多副本）

```
┌───────────────────────────────────────────────────────────────┐
│                Single Node, shared 8-GPU pool                 │
├───────────────┬───────────────┬───────────────┬───────────────┤
│  bundle 0-3   │  bundle 4-5   │  bundle 6-7                   │
│ student rollout│ GSM8K teacher │ Geo3K teacher                │
│   (4 GPU)      │ 2 replicas    │ 2 replicas                   │
│                │ (TP=1 each)   │ (TP=1 each)                  │
├────────────────┴───────────────┴──────────────────────────────┤
│        training: actor uses ALL 8 bundles                     │
└───────────────────────────────────────────────────────────────┘
```

每个 teacher 内部按 `--teacher-num-gpus-per-engine` 拆成多副本，round-robin 分担同一 data_source 的请求量。

## 数据准备

用 `prepare_data.py`（本目录下）把原始数据转成统一 schema：`prompt`（str）、`label`（str）、`data_source`（str，路由键）、`images`（VL 内嵌 base64，可选）、`extra_info`（固定 `{"rm_type": "mopd", "source": ...}`）。

### Step 1: 下载数据

```bash
pip install datasets pandas hf

# GSM8K + Geometry3K
python -c "
from datasets import load_dataset
load_dataset('openai/gsm8k', 'main', split='train').to_parquet('/root/gsm8k/train.parquet')
load_dataset('hiyouga/geometry3k', split='train').to_parquet('/root/geo3k/train.parquet')
"

# dapo-math-17k + multimodal-open-r1-8k-verified
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified --local-dir /root/multimodal-open-r1-8k-verified
```

### Step 2: 合并数据

```bash
python examples/on_policy_distillation/mopd/prepare_data.py \
    --gsm8k-dir /root/gsm8k \
    --geo3k-dir /root/geo3k \
    --dapo-math-path /root/dapo-math-17k/dapo-math-17k.jsonl \
    --openr1mm-path /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
    --output-dir /root/data/MOPD \
    --test-ratio 0.05
```

输出 `train.parquet` / `test.parquet` / `test_small.parquet`（按 data_source 均衡采样，训练中快速 eval 用）。`--output-dir` 需要和训练脚本的 `PROMPT_SET`/`DATA_DIR` 对上，对不上直接用 `PROMPT_SET`/`EVAL_SET` 环境变量覆盖即可。

### Step 3: 下载模型

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /root/Qwen3-VL-2B-Instruct
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

## 启动训练

```bash
MODEL_DIR=/root/models DATA_DIR=/root/data bash run-mopd-qwen3-vl-2b-8xgpu-colocate.sh
MODEL_DIR=/root/models DATA_DIR=/root/data bash run-mopd-qwen35-35ba3b-16xgpu-colocate.sh
```

框架会自动为每个教师拉起托管 SGLang 服务，健康检查通过后开始训练。

## 路由机制

1. 每条样本的 metadata 中带 `data_source` 字段（由数据管线自动注入）。
2. `--opd-teacher-key data_source` 指定路由键。
3. `--opd-teacher-routes` 定义 `data_source → HF 权重路径` 映射：

```json
{
  "openai/gsm8k": "/root/Qwen3-4B-Instruct-2507",
  "hiyouga/geometry3k": "/root/Qwen3-VL-4B-Instruct"
}
```

rollout 阶段按每条样本的 `data_source` 找到对应教师，请求 token 级 log-probability；某个教师请求失败或超时会逐样本回退到 rollout log-probs，仅当一整步内所有非空样本都失败才报错中断。

> `--resource` 里 `teacher` 的 GPU 数是教师总预算，先在教师间平均分配（要求整除），每个教师再按 `--teacher-num-gpus-per-engine` 拆成若干副本（副本 TP=该值），rollout 时先路由到教师再在其副本间 round-robin。

## 奖励路由：`rm_type=mopd`

所有数据源的 `extra_info.rm_type` 都固定为 `"mopd"`，统一分发到 `get_mopd_reward`，它再按 `data_source` 内部路由到具体打分函数，并**始终返回 float**。不要把某个数据源改成底层的 `"dapo"`/`"openr1mm"` 直接路由——它们返回值类型不一致（`dapo` 是 dict，其余是 float），混在同一 batch 会在 eval 聚合时报 `TypeError`。

## MOPD 专属参数

| 参数                            | 说明                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------- |
| `--opd-teacher-routes`          | `{data_source: HF 权重路径}` JSON，托管多教师                                |
| `--opd-teacher-key`             | 路由键 metadata 字段（默认 `data_source`）                                   |
| `--teacher-num-gpus-per-engine` | 每个教师 SGLang 引擎的 GPU 数（=教师 TP）。教师副本数 = 该教师 GPU 数 / 此值 |

### 稳定性注意

- `--colocate --offload` 必须同时开启，否则教师/rollout 训练时不换出显存，和 actor 抢卡导致 OOM。
- `ROLLOUT_GPUS + TEACHER_GPUS` 必须精确等于 `ACTOR_GPUS`，启动时报错而非运行时才 OOM。
- 35B 级别 student 建议把 `ACTOR_GPUS` 开够（本例 16 卡）：显存不够时 `optimizer.step()` 的 grad-norm all-reduce 会报 `ncclUnhandledCudaError` 而不是常见的 `torch.OutOfMemoryError`，容易被误判成别的问题。

## 扩展到 N 个教师

1. 用 `prepare_data.py` 为新领域样本生成带对应 `data_source` 的统一 schema 数据（或用 `--manifest` 接入已有 verl 格式数据）。
2. 在 `--opd-teacher-routes` 里加一条 `data_source → 权重路径`。
3. 增大 `--resource` 里 `teacher` 的 GPU 总预算，并**同步缩小 `--rollout-num-gpus`**（或整体扩大 `actor_gpus`），保证 `rollout + teacher == actor` 始终成立——新增教师不能凭空多占卡。

## 参考

- [On-Policy Distillation - Relax Docs](../README.md)
- [OPD 参数配置示例](../README.md#common-combinations)

## Qwen3.5-9B：五 Teacher、单引擎权重切换

只保留两个训练入口，共用同一份训练 recipe：

| 入口（本目录 `scripts/`）                 | Teacher 切换方式                                                      |
| ----------------------------------------- | --------------------------------------------------------------------- |
| `run-mopd-qwen35-9b-5teacher-baseline.sh` | 完整权重加载；默认 disk，`WEIGHT_SOURCE=memory` 使用五份完整 CPU 缓存 |
| `run-mopd-qwen35-9b-5teacher-adp.sh`      | 从 pinned CPU 内存传输差分包，在 GPU 原地无损恢复                     |

两者均为一个 Qwen3.5-9B Student、T1～T5 五个逻辑 Teacher、一个物理 TP1 Teacher
引擎。默认单机两卡，也可设置 `NUM_GPUS=8` 使用原八卡配置：

| `NUM_GPUS`  | rollout / Teacher 评分阶段                | Student 训练阶段    |
| ----------- | ----------------------------------------- | ------------------- |
| `2`（默认） | 1卡 Student rollout TP1 + 1卡 Teacher TP1 | 同两卡，TP2/DP1/PP1 |
| `8`         | 7卡 Student rollout TP1 + 1卡 Teacher TP1 | 同八卡，TP4/DP2/PP1 |

进入训练阶段时卸载 rollout/Teacher；训练结束后恢复推理权重，继续下一轮。
Teacher 始终只有一个实例，五份权重按请求切换。每条样本按 `extra_info.data_source` 选择一个 Teacher，
同 Teacher 请求成组处理。两种入口的模型、数据、训练参数和编排相同。

算法代码在 `relax/backends/sglang/adp/`，执行与生命周期接入点为
`relax/backends/sglang/adp_teacher_loader.py`、`relax/utils/opd/` 和
`relax/distributed/ray/shared_teacher_manager.py`。无需额外算法代码仓库或算法 PYTHONPATH。
ADP 入口复用 baseline 文件中的公共函数；直接运行 baseline 时禁止选择 ADP，避免误测。

### 本机已准备的输入

权重原文件仍在模型目录；工作目录中的输入统一位于 `local/mopd/qwen35-9b/`：

```text
models/Qwen3.5-9B/        Student 的共享文本输入视图
models/teachers/T1..T5/   五份 Teacher 的共享文本输入视图
train.jsonl              五路均衡的真实 DAPO 数据
teacher_manifest.*.json  disk / memory / adp 的运行输入
adp/                     五份运行时 capture、十个物理差分包和完整性 manifest
nvcomp/                  本机已有 nvCOMP 的独立 namespace 依赖
```

`local/` 和训练输出 `outputs/` 均不进入 Git。模型视图引用原模型文件，差分文件直接
引用本目录内的独立文件项，归档旧代码目录不会影响它们。没有任何自动下载步骤。
其他机器需自行准备模型、数据、兼容的 Torch/Triton/SGLang/Megatron 和 nvCOMP。
输入目录可由 `MOPD_INPUT_DIR` 覆盖；模型、数据、manifest、checkpoint 路径也可分别
通过 `MODEL_DIR`、`STUDENT_MODEL`、`PROMPT_SET`、`SHARED_TEACHER_MANIFEST`、`SAVE_DIR` 覆盖。

### 从 Relax 根目录验证

先确认所选两卡（或八卡）可供本次任务使用；脚本不会停止其他作业或全局清理 Ray。
当前服务器使用既有 Ray 集群，设置其实际地址后执行：

```bash
export RAY_ADDRESS="<ray-head-host>:6379"
export RAY_DASHBOARD_ADDRESS="http://<ray-head-host>:8265"
export MEGATRON="/path/to/Megatron-LM"
SCRIPTS=examples/on_policy_distillation/mopd/scripts

# 只检查配置：不访问 Ray、不创建输出、不加载模型
bash "$SCRIPTS/run-mopd-qwen35-9b-5teacher-baseline.sh" --dry-run
bash "$SCRIPTS/run-mopd-qwen35-9b-5teacher-adp.sh" --dry-run

# 默认两卡；两个独立实验按顺序执行，每个完成两次真实优化器更新
NUM_ROLLOUT=2 EXP_NAME=mopd-disk-check \
  bash "$SCRIPTS/run-mopd-qwen35-9b-5teacher-baseline.sh"
NUM_ROLLOUT=2 EXP_NAME=mopd-adp-check \
  bash "$SCRIPTS/run-mopd-qwen35-9b-5teacher-adp.sh"
```

默认前台等待 Ray job 完成。`RAY_NO_WAIT=1` 可改为提交后返回；这只代表提交成功。
新实验换 `EXP_NAME`；保持 `EXP_NAME` 和 `SAVE_DIR` 会恢复已有原生 checkpoint。
继续训练时 `NUM_ROLLOUT` 是目标总轮数，例如从第1步 checkpoint 继续到第3步设置4，
不会重新从 HF 权重开始。不要同时运行两个入口，它们共用所选卡池。
`NUM_GPUS=8` 切回八卡；卡数只支持2或8，TP和 rollout 数量随之调整。
该变量指定卡数，不绑定物理卡号；使用既有 Ray 时，以 placement 日志中的 GPU ID 为准。
默认 checkpoint 名含卡数，避免两卡训练误读八卡状态；手动设置 `EXP_NAME` 时也应区分。

默认 GBS40（20 prompts × 2 responses）、context4096、response256、保存间隔5，
最后一步仍保存原生模型和优化器。输出在 `outputs/mopd/$EXP_NAME/`；每次启动的
`runs/<timestamp>/` 保存独立 manifest、`submission.log`、控制器切换事件和 worker 事件。
`prepare_colocated` 是首次准备，`switch` 是 Teacher 间切换，`restore` 是训练后显存恢复；
三者分开计时。确认两步 `Actor training completed`、五个 Teacher 的成功评分和 Ray
最终状态，再检查切换事件，不能只看提交命令返回。

### 正确性与边界

共享文本视图显式统一 Student/Teacher 输入配置，保留各社区权重原文件；Teacher
使用 Student 已生成的 token IDs，不复现各社区聊天模板的原生行为。新模型必须先用
`prepare_shared_runtime_views.py` / `prepare_qwen35_9b_baseline.py` 验证配置、tokenizer
和张量布局。已有 capture 可用 `build_adp_teacher_packages.py` 编码，完成时发布包和
本仓库算法源码的 SHA；启动先校验，在线切换不重新读包或压缩。

当前支持配置是单机两卡/八卡、Qwen3.5-9B、Teacher TP1、文本 `student_sampled`，固定
Teacher 并发1、禁用 overlap/CUDA graph。切换或恢复完成状态不确定时关闭评分准入，
训练报错退出，不原地重建 Teacher 后继续。显存卸载后需要先恢复一份完整 CPU 基座，
再按需应用差分，不能算纯差分切换。disk 读取可能命中 OS 缓存，不能称冷盘基线。
首次准备、检查点保存和切换耗时分别统计；不承诺每次切换小于1秒。

### 八卡迁移验证（2026-10-19）

已同步上游 `903fa2f`，并通过八卡配置的两个入口分别完成两次优化器更新、最终原生
模型/优化器检查点保存和正常退出。每次实验中，T1～T5 各完成16次成功评分，
无评分失败。已验证训练后 Teacher 恢复，以及 Serve、嵌套 worker、SGLang spawn
使用同一份当前 checkout，避免机器上旧 Relax 副本遮蔽源码。

| 本轮样本      | 成功切换次数 | 切换中位数 | 最慢一次 |
| ------------- | -----------: | ---------: | -------: |
| 完整权重 disk |           12 |    4.201 s | 92.675 s |
| 内存差分 ADP  |           20 |    0.267 s |  3.954 s |

这里统计控制器 `teacher_switch` 的墙钟时间。队列按到达顺序组批，因此两次实验的
切换次数不同；disk 未清除系统页缓存，首次读取与后续缓存命中的耗时均保留。
ADP 首次准备为807.5秒，显式 CPU 缓冲为76.15 GB；首次准备和训练后的完整基座
恢复均不计入上表，也不把这20次样本作为长期稳定性或严格1秒上限的保证。

结果与完整日志在 `outputs/mopd/native_migration/attempt2/`，其中 `results.json`
为汇总，`summary.json` 记录作业、评分和清理结果。两次 Ray job 均为 `SUCCEEDED`，
原 GPU2～7 占卡已恢复。运行前需先停止本机用户的占卡进程，使八卡可用。

相关 CPU 回归、真实 CPU Serve→spawn 路径检查和 GPU 逐字节恢复检查均通过。
当时本机未缓存 pre-commit hooks，离线执行无法初始化；该轮只执行了相关 Ruff
检查和格式检查。代码备份前已补跑完整 pre-commit，结果见下文。

### 两卡验证（2026-10-19）

实际使用本机 GPU0/1（两张 H20G，每卡275040 MiB），Student 训练为 TP2/DP1/PP1，
rollout 和共享 Teacher 各为 TP1。baseline 与 ADP 均完成 step0、step1 两次真实
优化器更新，loss/grad norm 有限，五个 Teacher 各成功评分16次，无失败；均保存完整
原生模型/优化器 checkpoint（每份约131.8 GB）。

| Teacher 切换  | 成功次数 |  中位数 | 最慢一次 |
| ------------- | -------: | ------: | -------: |
| 完整权重 disk |       16 | 4.901 s |  7.240 s |
| 内存差分 ADP  |       41 | 0.266 s |  1.546 s |

这里仍是控制器 `teacher_switch` 的墙钟时间；请求到达与组批不同，切换次数也不同，
未清理 OS 页缓存。中位数比值约18.5，不代表整轮训练提速，也不满足每次小于1秒。
ADP 首次准备388.9秒，显式 CPU 包/恢复缓冲76.15 GB（不含其他运行时和临时内存）；
3次训练前后完整基座恢复另计，中位0.599秒、最大0.625秒。

baseline Ray 终态为 `SUCCEEDED`。首次 ADP 的训练、保存和清理已完成，但临时验收
监控因一次 `nvidia-smi` 查询超时而误停正在退出的 job，Ray 终态为 `STOPPED`。
原始失败报告保留；`adp_training_audit.json` 独立核对已完成的训练、评分和保存。
修正监控后，从该 checkpoint 成功恢复到 step2，完成一次真实更新、保存 iteration 2
检查点并正常退出（Ray `SUCCEEDED`）；五个 Teacher 各新增8次成功评分，无失败。
续训 loss/grad norm 有限、学习率保持1e-6，GPU0/1 已释放，Serve 应用已清理。

原始日志、运行配置、源码 SHA、CPU 回归记录和统计位于
`outputs/mopd/two_gpu_validation/attempt1/`，续训记录位于同级 `resume/`。
受影响的 CPU 回归为52项测试和23项子测试通过；两个入口的 shell 语法与相关 Ruff
检查通过。GPU2～7 的原占卡进程保持不变。此次只验证本机两卡，未执行多节点训练。

### 代码备份与换机恢复

备份前已完成 pre-commit 自动修复及全仓检查，所有 hooks 通过。修复后回归为466项
测试和29项子测试通过；GPU 差分编码、保存加载和批量恢复均通过逐字节一致性检查。
自动修复仅调整格式与 docstring，Python 执行逻辑经 AST 对比保持一致。

代码备份仓库为 `https://github.com/cattidea/Relax.git`，分支为
`yiyou/opt/mopd-teacher-weight-switch`。分支包含框架接入、算法源码、两个训练入口、
输入准备工具、文档和测试；模型、差分包、训练数据、checkpoint 和完整日志不进入 Git。

当前可复用的资产保存在共享盘。以下路径相对原 Relax 工作目录，实际绝对路径记录在
额外备份的 `migration-environment.json` 中：

- 输入与差分包：`local/mopd/qwen35-9b/`。
- 原始模型：`../models/mopd_qwen35_9b/`。
- 实验输出：`outputs/mopd/`。

新机器挂载同一共享盘时，可以继续使用原目录；若独立 clone 代码，将 `MOPD_INPUT_DIR`
指向原输入目录，并设置新机器实际的 `RAY_ADDRESS`、`RAY_DASHBOARD_ADDRESS` 和
`MEGATRON`。模型视图包含指向原始模型目录的链接，两处路径需同时可访问。
若共享盘路径改变，需重建模型视图并调整输入 manifest 的路径。

本次额外保存了源文件归档、补丁、Git bundle 和环境清单，位置为原工作目录旁的
`../relax-backups/mopd-teacher-weight-switch-20261019/`。
运行环境版本及外部源码状态见其中的 `migration-environment.json`；Git clone
不会安装 Torch/SGLang/Megatron/nvCOMP，换机时还需准备兼容的运行环境。
