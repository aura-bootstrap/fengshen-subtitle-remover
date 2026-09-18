# Wan-VACE 修复层联调手册（bring-up）

目标：在有 NVIDIA GPU 的机器上把 `scripts/wanvace_infer.py` 从"代码就绪"推到"实测通过"。sidecar 与 ProPainter 共用 PNG 契约，Go 侧零改动。

## 硬件门槛

| 档位 | 显存 | 说明 |
|---|---|---|
| vace-1.3B（推荐起步） | ≥12GB（配合 WANVACE_OFFLOAD=1 可更低） | 720p 字幕带裁剪后推理 |
| vace-14B | ≥24GB | 质量更高，速度慢一个量级 |

纯 CPU 不可行（扩散模型，参考 DiffuEraser：12G 卡 640×360 也要 92s/10s 片段）。

## 准备

```bash
# 1. 代码与权重（全程走代理，宿主 Clash 127.0.0.1:7897）
git clone https://github.com/Wan-Video/Wan2.1   # 放 vendor/Wan2.1
cd Wan2.1 && pip install -r requirements.txt     # 建议独立镜像 tag
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir Wan2.1-VACE-1.3B
```

## 自检

```bash
scripts/wanvace_check.sh /path/to/Wan2.1   # 全绿才可进下一步
```

## 首跑核对清单（脚本头同款，踩坑点）

1. **参数名**：当前 generate.py 的 VACE 参数为 `--task vace-1.3B --size W*H --frame_num N --ckpt_dir --src_video --src_mask --prompt --save_file`；若上游改名，改 `wanvace_infer.py` 里 cmd 构造段。
2. **mask 极性**：sidecar 按"白=重生成"编码 mask 视频；若首跑结果把字幕区当保留区，反转 `wanvace_infer.py` 里 `np.where(mk > 127, 255, 0)`。
3. **帧数约束**：Wan 因果 VAE 只接受 4k+1 帧，sidecar 已自动补齐（末帧重复+黑 mask），无需手工处理。
4. **显存溢出**：先 `WANVACE_OFFLOAD=1`；再不行把 `WANVACE_SIZE` 降到 `640*360`。

## 联调步骤

```bash
# 容器/环境里
export WANVACE_HOME=/work/vendor/Wan2.1
export WANVACE_CKPT=$WANVACE_HOME/Wan2.1-VACE-1.3B
export WANVACE_OFFLOAD=1

# 用 rand_ep1（4s/103 帧）做首个联调片段
bash scripts/run_container.sh remove data/raw/rand_ep1.mp4 \
  -o data/out/rand_ep1_vace.mp4 \
  --propainter --force-engine propainter \
  --propainter-script /src/scripts/wanvace_infer.py --grain --crf 15
```

- sidecar 任何失败 exit 3 → 自动降级运动层，日志见 `warn: painter failed`
- 通过标准：`data/out/rand_ep1_vace.mp4` 产出、EXIT=0、verify 残留统计不劣于 ProPainter 版（rand_ep1_fixed.mp4）

## 与 ProPainter 对比验收

同一 rand 片段抽同帧对比：VACE 版 vs `rand_epN_fixed.mp4`（ProPainter pp_max），看字幕区纹理连续性与帧间闪烁。
