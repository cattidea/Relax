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
| `run-mopd-qwen35-397ba17b-160xgpu-colocate.sh` | 学生397B，20节点160卡 | 160 / 80 / 80（5 Teacher，各16卡） |

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

## 397B：10/17 基线与逐项优化

本分支从 `3805d95` 重建，只迁入 1017 实验验证过的实现。初始入口复现003：GBS40、
Student 2P3D、TP4/PP5/CP8/EP32/ETP1、层数6/17/17/17/3、full/uniform/1重算。
每个后续提交对应下表的一项功能；收益是历史实验观测，不是本次机器上的重新测量。

当前入口默认：GBS160、mixed5（Decode Graph开、HiCache关）、6/14/14/14/12、PP3 selective其余full、256MiB同步buffer、Router阈值0。

### 使用与数据

在已准备好的160卡Ray集群上，由使用者执行：

```bash
MODEL_DIR=/shared/models DATA_DIR=/shared/data MEGATRON=/path/to/tested/Megatron-LM \
  bash examples/on_policy_distillation/mopd/run-mopd-qwen35-397ba17b-160xgpu-colocate.sh
```

`SOURCE_1_DATA` 至 `SOURCE_5_DATA`、`TEACHER_1_MODEL` 至 `TEACHER_5_MODEL` 可分别覆盖；
默认Teacher权重路径与Student相同，但启动五个独立服务。数据沿用已有prompt/label/images/
extra_info schema；行内data_source可省略，存在时必须匹配teacherN。源配比由YAML的
weight决定，与文件条数无关；每来源独立shuffle/循环，超长样本同源补齐。
1:2:3:4:5在GBS40/160/320下的配额分别为3/5/8/11/13、11/21/32/43/53、21/43/64/85/107。

`BATCH_SIZE`、`NUM_ROLLOUT`、`LOG_DIR`、`SGLANG_CONFIG`可覆盖。默认不保存checkpoint；
显式设置`SAVE_INTERVAL>0`时还需`SAVE_DIR`与稳定`EXP_NAME`。完整续训仍需独立验收。
脚本连接现有Ray，不安装依赖、不清理集群、不启用数值dump或实验监控。

### 运行环境边界

003运行快照：Python3.12.3、torch2.11.0+cu129、Ray2.56.1.2、SGLang0.5.17、
sglang-router0.3.2+slime.dpaware、mooncake-transfer-engine0.3.12.post1、ClearML2.1.3。
必须使用1017实验的实际Megatron/Bridge运行环境；当时没有Megatron Git元数据，
不能由Relax提交号或仓库Dockerfile推定其内容。不要为了迁移自动升级依赖或重建最新镜像。
Mooncake的metacache关闭和20000–22000端口范围已由003 wrapper并入入口。

历史日志与分析保留在`1017/experiments/`各编号目录，不复制进业务源码。迁移只复用原有
回归测试；按用户资源约束，本机不运行训练、Ray、测试及全仓hooks，需在目标环境复验。
已知Student非法概率、Teacher非有限logprob、Router偶发失败和退出Gloo噪音尚未根治。

### GDN旧运行时补丁

`docker/patch/megatron/gdn-thd-route-reuse.patch`对应015/016实测实现，必须配套部署到
每个训练节点的旧Megatron；提交此文件不等于运行环境已经安装。新式native route API不在
本分支的适配范围，不覆盖新版本。旧环境版本标签0.19.0，精确Git提交未知，以文件SHA核对：

| Megatron相对路径 | 修改前SHA256 | 修改后SHA256 |
| --- | --- | --- |
| `megatron/core/context_parallel_layout.py` | `75e645235aeaa2909895ee43169eeedfeffada95f57987e93d6205bc672a9ec5` | `0450ef5a7bb6143bd9f4d92fa7474fc24dea63cdd3b7e9b04ebf47fa14c37d38` |
| `megatron/core/ssm/gated_delta_net.py` | `137d3b2fa1f3a34a0174a1bfef5f9186dfb93994f05f2a1b08f17bf751fe7671` | `165ef2c961a7858605865b05774aad97bf85b6a3724dbc07c80fac4afb37d882` |

确认修改前SHA一致后，可在对应Megatron目录使用`patch --dry-run --fuzz=0 -p1`检查该补丁，
再正式应用并核对修改后SHA；已匹配修改后SHA的节点无需重复应用。两方向路由只在同一微批
跨层/重计算复用，不跨微批复用，也未验收训练CUDA Graph或真实GPU逐张量等价A/B。

### 提交与证据

| 步骤 | 来源 | 改动与已知收益 |
| --- | --- | --- |
| 01 | 003；ac26675、087421e | 完整baseline：按来源配额采样、Teacher双节点、397B入口；历史GBS40完成10轮/400条，无提速归因。 |
| 02 | 2c449ce；007→008 | pixels仅发物理PP0，grid仍发全员；消除其他PP的80%像素副本，不是整机内存下降80%。 |
| 03 | e3931d9；008 | PP0同机8 rank共享一份像素；GBS320像素容量模型977→122GiB/台（非整机峰值）。008完成10轮GBS160，入口默认升至160。 |
| 04 | 010配置 | 五个mixed Student、Decode Graph开、HiCache关、buffer256MiB；008→010观测rollout2387.84→1002.10秒，不能分别归因于上述组合变量。旧PD配置保留于baseline提交。 |
| 05 | 011→012 | PP从6/17/17/17/3调整为6/14/14/14/12；正常轮训练638.56→526.43秒（−17.56%），各一轮测速，不保证固定收益。 |
| 06 | 7e8a5ae；013→014 | Router绝对阈值4→0，沿用相对阈值；R1/R2窗口rollout2040.41→1808.43秒（−11.37%），不是严格均分保证。 |
| 07 | d0295fe；014→016 | 同微批GDN路由构造40→2次，跨层及重算复用；训练547.37→479.19秒（−12.46%）。保留旧Megatron补丁，需所有训练节点配套。 |
| 08 | 6bff36f；016→017 | 保持来源配额/Teacher路由，提交前跨来源混排；rollout936.88→725.62秒（−22.55%），端到端48.70→56.12 tokens/s/GPU（+15.24%）。 |
| 09 | 018；1f592b9拆分 | PP3 selective保留GDN/Attention等重算，去掉整块MoE重放；训练472.73→459.60秒（−2.78%），PP3显存采样43.98→61.37GiB。 |
