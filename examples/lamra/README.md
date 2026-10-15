# LamRA M-BEIR fine-tuning

Development integration of Qwen2-VL retrieval fine-tuning into Relax's existing
SFT producer, TransferQueue, Actor group, FSDP2, PEFT and distributed checkpoint
components. There is no rollout engine or vocabulary loss in this recipe.

The CPU contracts and adapter checkpoint/export/reload are tested. The two
Actor-resolution/validation dispatch changes are still pending repository-owner
approval; this branch is not yet an end-to-end validated GPU training recipe.

## Training contract

- One queue row carries an atomic query/positive pair. Positives and instructions
  are selected lazily; missing media or positives fail instead of being skipped.
- Eight ranks consume 64 pairs each. Each rank encodes its 64 queries followed by
  64 positives in 32-sequence forward chunks, then computes one mean, one-way
  InfoNCE over all 512 pairs at temperature 0.05.
- Features precede the first `<emb>` token; positions are legacy sequential
  positions. Image preprocessing uses the slow original Qwen2-VL processor,
  4–300 image tokens and the native 480-token text truncation.
- Remote features are detached, local features retain gradients, and FSDP
  averages parameter gradients. The native `1/world_size` gradient scale is
  intentional. An autograd gather or world-size loss multiplier changes it.
- Only language projection LoRA is trained: r128/alpha256/dropout0.05. Vision,
  embeddings and the unused vocabulary head remain frozen/excluded. Forward
  parameters are BF16; optimizer masters and gradient reduction are FP32.
- One complete dataset has 1,332,790 queries and 1,932,885 candidates. The last
  global batch contains 54 remaining pairs plus 458 repeated pairs from the
  current epoch's head, for 2,604 optimizer steps. No candidates or queries are
  filtered to make the batch divisible.
- Relax's index manager determines order. Position-seeded lazy choices support
  resuming this implementation; they do not reproduce HF's shuffle/RNG trajectory.
  This implementation uses FSDP2 and SDPA, so bitwise native-trainer equivalence
  is not claimed.

## Launch and resume

Provide these roots outside the repository and Ray working directory:

```text
MODEL_DIR/
  pt/                         existing merged LamRA PT model and tokenizer
  processor/                  original Qwen2-VL processor metadata
  initial_adapter.safetensors optional complete initial LoRA, checked exactly
DATA_DIR/
  metadata/query/union_train/mbeir_union_up_train.jsonl
  metadata/cand_pool/global/mbeir_union_train_cand_pool.jsonl
  metadata/instructions/query_instructions.tsv
  media/recovered/             preferred exact recovered image bytes
  media/original/              remaining images
SAVE_DIR/                     persistent run/checkpoint root
```

The script audits all query/positive media references before submitting. Set
`RAY_ADDRESS` to the verified Ray Jobs HTTP address and export the three roots,
then run on an available cluster:

```bash
bash scripts/entrypoint/ray-job.sh \
  examples/lamra/scripts/run-qwen2-vl-2B-mbeir-ft-8xgpu.sh
```

The default submission blocks. `RAY_NO_WAIT=1` explicitly requests nonblocking
submission. The existing entrypoint performs cluster cleanup: coordinate any
other job before invoking it. No global dependencies or drivers are changed by
this recipe.

`EXP_NAME` is stable across restarts; ClearML run names additionally include a
timestamp. Every 200 steps and at completion the existing DCP component saves
model adapter, optimizer and rank RNG with a `COMMITTED` marker. Two committed
checkpoints are retained. Restart with the same roots and `EXP_NAME` to load the
latest committed state. The final portable PEFT adapter is written under
`SAVE_DIR/EXP_NAME/adapter`; tokenizer and original processor remain in MODEL_DIR.

## 中文说明

此实现复用现有 SFT、TransferQueue、Actor、FSDP2、PEFT 与 DCP 组件。每行保持 query/positive
原子性，完整训练为 2,604 步；缺图直接失败，不做过滤。末批使用本 epoch 开头的真实 pair
补齐，lazy 正例仍重新采样。Relax 的 shuffle 与位置种子不同于原 HF 轨迹，不宣称逐位等价。

目前已验证 CPU 数值契约、DCP 保存恢复以及 adapter 导出重载；Actor 分派与配置校验接入仍等待
仓库所有者确认，GPU 端到端训练尚未验证。运行前提供上述三个外部目录及已核验的 RAY_ADDRESS，
通过 ray-job.sh 启动，并先协调现有作业。每 200 步和结束时保存可恢复 checkpoint，保留两份；
同目录、同 EXP_NAME 重启可从最近 COMMITTED 状态恢复。数据、权重、机器路径和运行凭据不进入仓库。
