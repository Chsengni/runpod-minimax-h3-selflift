#!/bin/bash
# Serverless worker entrypoint: materialize baked ComfyUI, download models to
# the container disk, point ComfyUI at them, then hand off to the runpod handler.
#
# Models live on the container disk (MODEL_ROOT, default /models) — no network
# volume required. The disk is ephemeral: every cold start re-downloads the
# full model set (~110GB, see models-manifest.txt), so expect a long first
# boot after scale-to-zero. Override the location with MODEL_ROOT.
set -e

COMFYUI_DIR="/workspace/runpod-slim/ComfyUI"
BAKED="/opt/comfyui-baked"
VENV_DIR="$COMFYUI_DIR/.venv-${VENV_SUFFIX:-cu130}"
MODEL_ROOT="${MODEL_ROOT:-/models}"

if [ ! -f "$COMFYUI_DIR/main.py" ]; then
    echo "Materializing baked ComfyUI into workspace..."
    mkdir -p "$COMFYUI_DIR"
    cp -a "$BAKED/." "$COMFYUI_DIR/"
fi

mkdir -p "$MODEL_ROOT"

# Sanity check: ~110GB of models need room plus download temp space.
free_gb=$(df --output=avail -BG "$MODEL_ROOT" | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -lt 120 ]; then
    echo "entrypoint: WARNING: only ${free_gb}GB free on $MODEL_ROOT — ~110GB of models may not fit" >&2
fi

# Point ComfyUI at the container-disk model dir.
cat > "$COMFYUI_DIR/extra_model_paths.yaml" << YAML
container_disk:
  base_path: $MODEL_ROOT
  diffusion_models: diffusion_models
  loras: loras
  vae: vae
  text_encoders: text_encoders
  vae_approx: vae_approx
  latent_upscale_models: latent_upscale_models
YAML

# Auto-populate models on boot (models-manifest.txt). Each worker has a private
# disk, so every cold worker downloads the full set; disable with
# VOLUME_AUTOPOPULATE=false.
# Note: populate-volume.sh downloads into $VOL_ROOT/models, so strip a trailing
# /models from MODEL_ROOT to land files directly in $MODEL_ROOT.
if [ "${VOLUME_AUTOPOPULATE:-true}" = "true" ]; then
    VOL_ROOT="${MODEL_ROOT%/models}" /opt/serverless/populate-volume.sh || echo "entrypoint: model populate failed — jobs may error on missing models" >&2
fi

# SelfLift workflow: QwenH3PromptLocal hardcodes models/LLM/Qwen3.8 and the
# H3 latent upscaler scans models/latent_upscale_models — link both.
mkdir -p "$MODEL_ROOT/LLM/Qwen3.8" "$MODEL_ROOT/latent_upscale_models" "$COMFYUI_DIR/models/LLM"
for pair in "LLM/Qwen3.8" "latent_upscale_models"; do
    dest="$COMFYUI_DIR/models/$pair"
    [ -d "$dest" ] && [ ! -L "$dest" ] && rm -rf "$dest"
    ln -sfn "$MODEL_ROOT/$pair" "$dest"
done

export PATH="$VENV_DIR/bin:$PATH"

# llama.cpp runtime fallback for QwenH3PromptLocal (cached on container disk)
QN="$COMFYUI_DIR/custom_nodes/ComfyUI_Qwen_H3_Prompt"
if [ -d "$QN" ] && [ ! -f "$QN/runtime_config.json" ]; then
    (cd "$QN" && python3 install_runtime.py --backend cuda13) || echo "entrypoint: Qwen llama runtime install failed" >&2
fi
exec python3 /opt/serverless/handler.py
