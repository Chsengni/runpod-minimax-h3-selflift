# runpod-minimax-h3-selflift — MiniMax H3 SelfLift + sharpness upscaler (normal VAE / 2X VAE) serverless worker (CUDA 13.0)
#
# - Comfy-Org INT8 ConvRot DiT + Qwen3-VL-32B Heretic NVFP4 TE + lightx2v 4-step
#   LoRA + H3 sharpness latent upscaler + Qwen3.8-27B prompt LLM, all on the
#   network volume (~110GB; 96GB Blackwell GPU recommended)
# - Missing volume files are auto-downloaded at worker boot from
#   models-manifest.txt (populate-volume.sh, VOLUME_AUTOPOPULATE=false to skip)
# - runpod serverless handler drives ComfyUI per-job (handler.py)
#
# Build (from repo root):
#   docker build -t chsengni/runpod-minimax-h3-selflift:latest .

FROM huchukato/comfyui-base:cu130

# ──────────────────────────────────────────────────────────────────────────────
# Custom nodes — only what the two SelfLift workflows use
# ──────────────────────────────────────────────────────────────────────────────
ENV GIT_TERMINAL_PROMPT=0
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    /opt/comfyui-baked/custom_nodes/ComfyUI-VideoHelperSuite

# SelfLift + 2xVAE + sharpness-upscaler workflow nodes (commits pinned to the
# versions the workflow was saved with; selflift main renamed model→low_res_model)
RUN pin() { \
    local user="$1" repo="$2" ref="$3"; \
    echo "Cloning $user/$repo @ $ref"; \
    git clone "https://github.com/$user/$repo.git" "/opt/comfyui-baked/custom_nodes/$repo" && \
    git -C "/opt/comfyui-baked/custom_nodes/$repo" checkout -q "$ref"; \
  } && \
    pin slmonker selflift-Avatar v0.1.4-experimental && \
    pin yichengup ComfyUI-YCNodes-MiniMax-H3 146207295c1d2bd6b64a6552484cb6ed8be1c9ef && \
    pin chflame163 ComfyUI_Qwen_H3_Prompt f8ea17991ea39111ef2b2ebdf6ccb631e21e0300 && \
    pin TripleHeadedMonkey ComfyUI-MiniMaxH3_LatentUpscaler 2e568dfe3e4f5e81da178bc845e05dcfbb64d55b


# ──────────────────────────────────────────────────────────────────────────────
# Requirements + runpod serverless SDK
# ──────────────────────────────────────────────────────────────────────────────
# libgl1/libglib2.0-0: opencv-python (VideoHelperSuite) fails to import without
# them → the whole pack never registers → missing_node_type on VHS_* nodes.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN bash -c 'cd /opt/comfyui-baked/custom_nodes && for node_dir in */; do \
        if [ -f "$node_dir/requirements.txt" ]; then \
            echo "Installing requirements for $node_dir..." && \
            pip install --no-cache-dir -r "$node_dir/requirements.txt" || echo "Failed to install requirements for $node_dir"; \
        fi; \
    done' && \
    pip install --no-cache-dir --no-deps "transformers>=5.2.0" && \
    pip install --no-cache-dir 'runpod>=1.12' requests "huggingface_hub[cli]" hf_transfer && \
    pip cache purge

# llama.cpp runtime for QwenH3PromptLocal (prompt rewriter). The build host has
# no GPU, so pick the CUDA 13 backend explicitly; the entrypoint retries at boot.
RUN cd /opt/comfyui-baked/custom_nodes/ComfyUI_Qwen_H3_Prompt && \
    (python3 install_runtime.py --backend cuda13 || echo "llama runtime will be installed at boot")

# ──────────────────────────────────────────────────────────────────────────────
# Model dirs (empty — real models come from the network volume via
# extra_model_paths.yaml written by the entrypoint)
# ──────────────────────────────────────────────────────────────────────────────
RUN mkdir -p /opt/comfyui-baked/models/{latent_upscale_models,vae,vae_approx,diffusion_models,text_encoders,loras,LLM}

ENV HF_TOKEN=""

# ──────────────────────────────────────────────────────────────────────────────
# Serverless entrypoint + handler + workflow templates (API format)
# ──────────────────────────────────────────────────────────────────────────────
COPY handler.py /opt/serverless/handler.py
COPY serverless-entrypoint.sh /opt/serverless/entrypoint.sh
COPY populate-volume.sh /opt/serverless/populate-volume.sh
COPY models-manifest.txt /opt/serverless/models-manifest.txt
COPY workflows-api/mmh3/ /opt/workflows/
RUN chmod +x /opt/serverless/entrypoint.sh /opt/serverless/populate-volume.sh

ENV COMFYUI_DIR="/workspace/runpod-slim/ComfyUI" \
    WORKFLOWS_DIR="/opt/workflows" \
    JOB_TIMEOUT_S="1800" \
    VENV_SUFFIX="cu130"

# Base image carries ENTRYPOINT ["/start.sh"] (pod mode) — a bare CMD would be
# passed to it as args and the runpod handler would never start. Override the
# entrypoint so the container runs the serverless handler directly.
ENTRYPOINT ["/opt/serverless/entrypoint.sh"]
