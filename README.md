# RunPod Serverless · MiniMax H3 SelfLift + 锐化放大

[![Runpod](https://api.runpod.io/badge/chsengni/runpod-minimax-h3-selflift)](https://console.runpod.io/hub)

基于 [huchukato/runpod-minimax-h3](https://github.com/huchukato/runpod-minimax-h3) 精简，只保留两套工作流：

| 工作流 | 解码 |
|---|---|
| `MiniMaxH3-SelfLift-Sharpen-R2VA.json` | 普通 VAE（`minimax_h3_video_vae_fp16`） |
| `MiniMaxH3-SelfLift-2xVAE-Sharpen-R2VA.json` | 2X VAE（hyperVAE Krea2）+ `MiniMaxH3VAEDecodeFast` 分块快速解码 |

流程：最多 9 张参考图 + 3 段参考视频（可带原声）+ 3 段参考音频 → `MiniMaxH3ReferenceToVideo` → `SelfLiftAvatarH3Sampler`（低分辨率采样 + H3 锐化潜空间放大 `h3_upscaler_sharpness_2000steps_v0.1_fp32`）→ 音频 + 视频 → `VHS_VideoCombine` 输出 mp4。

提示词不经过 LLM 改写，直接送进模型——请在 prompt 里自己写 `<Picture 1>` `<Video 1>` `<Audio 1>` 标签引用参考素材。

## 自定义节点（已固定版本）
- `Kosinkadink/ComfyUI-VideoHelperSuite`
- `slmonker/selflift-Avatar@v0.1.4-experimental`（main 分支把 `model` 改名为 `low_res_model`，所以固定旧版）
- `yichengup/ComfyUI-YCNodes-MiniMax-H3`（`H3SigmaRefiner`）
- `TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler`（`MiniMaxH3VAEDecodeFast`）

## 模型
`models-manifest.txt` 里的文件会在 worker 首次启动时自动下载到容器磁盘 `/models`（约 93GB，可用 `MODEL_ROOT` 环境变量改位置）：

- DiT：`minimax_h3_fl2va_pruned_int8_convrot`
- 文本编码器：`qwen3vl_32b_heretic_minimax_h3_nvfp4`（NVFP4）
- VAE：视频 fp16 + 音频 fp32
- LoRA：Kijai lightx2v turbo 4 步（强度 0.65）
- 锐化放大：`h3_upscaler_sharpness_2000steps_v0.1_fp32`
- 2X VAE：Civitai 下载，需要设置环境变量 `CIVITAI_TOKEN`，不设置则跳过，只能用普通 VAE 工作流

注意：容器磁盘是临时的——worker 缩容到 0 后磁盘清空，下次冷启动会重新下载全部模型（约 93GB）。如需模型持久化，改用网络卷方案（把 `MODEL_ROOT` 指向 `/runpod-volume/models` 并挂载网络卷）。

原工作流用 bf16 模型；如需切换，在请求里用 `params` 覆盖，例如 `{"127:unet_name": "...", "128:clip_name": "..."}`。

## 部署
1. 推送本仓库（RunPod 从分支构建镜像）。
2. RunPod → Serverless → New Endpoint → GitHub Repo，选择本仓库。
3. GPU 选 RTX 5090（32GB），容器磁盘 150GB；模型下载到容器磁盘，无需挂载网络卷；需要 2X VAE 时设置 `CIVITAI_TOKEN`。

## 请求示例

图片、视频、音频至少要传一种。请在 prompt 里自己写 `<Picture 1>` `<Video 1>` `<Audio 1>` 标签引用参考素材。

```json
{
  "input": {
    "workflow": "2xvae",
    "prompt": "第一人称一镜到底的镜头，<Picture 1> 的角色拿着 <Picture 2> 的显卡从 <Picture 3> 的店铺跑向主视角……",
    "images": ["https://.../ref1.png", "https://.../ref2.png", "https://.../ref3.png"],
    "videos": ["https://.../dance.mp4", {"url": "https://.../walk.mp4", "audio": false}],
    "audios": ["https://.../voice.wav"],
    "seconds": 10
  }
}
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `workflow` | `normal`（普通 VAE）或 `2xvae`（2X VAE + 分块快速解码），也可写 json 文件名 | `normal` |
| `prompt` | 创意/指令文本（必填），用 `<Picture i>` / `<Video i>` / `<Audio i>` 引用参考素材 | |
| `images` | 0–9 张参考图，URL / base64 / data URI，依次对应 `<Picture 1..9>` | |
| `videos` | 0–3 段参考视频（2–15 秒，按 24fps 读取，最多 360 帧），对应 `<Video 1..3>`；可写成对象 `{"url": ..., "audio": false, "skip_frames": 0, "max_frames": 360}`，`audio` 默认 `true`，即同时送入视频原声 | |
| `audios` | 0–3 段独立参考音频，对应 `<Audio 1..3>`；对象写法可加 `start` / `duration`（秒） | |
| `seconds` | 视频时长（秒） | 10 |
| `aspect_ratio` / `megapixels` | 分辨率选择 | `16:9 (Widescreen)` / 1.5 |
| `seed` | 固定采样器种子 | 随机 |
| `steps` | 采样步数 | 8 |
| `lora_strength` | lightx2v turbo LoRA 强度 | 0.65 |
| `transition_step` / `lowres_scale` | SelfLift 低分→高分切换步 / 低分比例 | 6 / 0.5 |
| `params` | 原始覆盖 `{"节点id:输入名": 值}`，最后应用 | |

返回：`{"outputs": [{"filename": "selflift_00001.mp4", "b64": "..."}], "final_prompt": "输入的提示词（原样返回）", "seed": 123}`

健康检查：`{"input": {"action": "health"}}`
