#!/bin/bash
# Serverless worker entrypoint: materialize baked ComfyUI, point it at the
# network volume for models, then hand off to the runpod handler.
set -e

COMFYUI_DIR="/workspace/runpod-slim/ComfyUI"
BAKED="/opt/comfyui-baked"
VENV_DIR="$COMFYUI_DIR/.venv-${VENV_SUFFIX:-cu130}"

if [ ! -f "$COMFYUI_DIR/main.py" ]; then
    echo "Materializing baked ComfyUI into workspace..."
    mkdir -p "$COMFYUI_DIR"
    cp -a "$BAKED/." "$COMFYUI_DIR/"
fi

# Models live on the RunPod network volume, shared across every worker.
cat > "$COMFYUI_DIR/extra_model_paths.yaml" << 'YAML'
runpod_volume:
  base_path: /runpod-volume/models
  diffusion_models: diffusion_models
  loras: loras
  vae: vae
  text_encoders: text_encoders
  vae_approx: vae_approx
  latent_upscale_models: latent_upscale_models
YAML

# Auto-populate the network volume on first boot (models-manifest.txt).
# Concurrent cold workers serialize on a flock inside populate-volume.sh;
# disable with VOLUME_AUTOPOPULATE=false.
if [ "${VOLUME_AUTOPOPULATE:-true}" = "true" ]; then
    /opt/serverless/populate-volume.sh || echo "entrypoint: volume populate failed — jobs may error on missing models" >&2
fi

# SelfLift workflow: QwenH3PromptLocal hardcodes models/LLM/Qwen3.8 and the
# H3 latent upscaler scans models/latent_upscale_models — link both to the volume.
if [ -d /runpod-volume/models ]; then
    mkdir -p /runpod-volume/models/LLM/Qwen3.8 /runpod-volume/models/latent_upscale_models "$COMFYUI_DIR/models/LLM"
    for pair in "LLM/Qwen3.8" "latent_upscale_models"; do
        dest="$COMFYUI_DIR/models/$pair"
        [ -d "$dest" ] && [ ! -L "$dest" ] && rm -rf "$dest"
        ln -sfn "/runpod-volume/models/$pair" "$dest"
    done
fi

export PATH="$VENV_DIR/bin:$PATH"

# llama.cpp runtime fallback for QwenH3PromptLocal (cached on the volume)
QN="$COMFYUI_DIR/custom_nodes/ComfyUI_Qwen_H3_Prompt"
if [ -d "$QN" ] && [ ! -f "$QN/runtime_config.json" ]; then
    (cd "$QN" && python3 install_runtime.py --backend cuda13) || echo "entrypoint: Qwen llama runtime install failed" >&2
fi
exec python3 /opt/serverless/handler.py
