"""RunPod serverless handler · MiniMax H3 SelfLift + sharpness upscaler (R2VA).

One API-format workflow lives in /opt/workflows:
  MiniMaxH3-SelfLift-Sharpen-R2VA.json        stock MiniMax H3 video VAE

Job input (everything except "prompt" and "images" is optional):
  {
    "workflow":   "normal",                    # or the json filename; default "normal"
    "prompt":     "...",                       # idea / directive text
    "images":     ["https://... or base64"],   # 0-9 reference images  → <Picture 1..9>
    "videos":     ["https://..." | {"url"|"b64": ..., "audio": true}],  # 0-3 reference videos (2-15s) → <Video 1..3>
                                               #   "audio": true also feeds the clip's soundtrack (default true)
    "audios":     ["https://... or base64"],   # 0-3 standalone reference audios → <Audio 1..3>
    "seconds":    10,                          # video length in seconds
    "aspect_ratio": "16:9 (Widescreen)",      # ResolutionSelector
    "megapixels": 1.5,
    "seed":       123,                         # pins sampler seed; random if omitted
    "steps":      8,                           # BasicScheduler steps
    "lora_strength": 0.65,                     # lightx2v turbo LoRA
    "transition_step": 6,                      # SelfLift low-res → high-res switch
    "lowres_scale": 0.5,
    "params":     {"<node_id>:<input>": value},# raw overrides, applied last
    "s3Config":   {"endpointUrl": ..., "accessId": ..., "accessSecret": ...,
                   "bucketName": ..., "region": ...}  # optional: large outputs offload here
  }

Output delivery: videos ≤ OUTPUT_INLINE_MAX_MB (default 10) are returned inline
as base64; anything larger is uploaded to the job's s3Config storage (e.g. the
RunPod network volume S3 API) and returned as a presigned URL. This matters
because RunPod's job-done callback drops results over ~20MB — without the
offload, a long/high-res video would "succeed" but arrive empty.

Env knobs: OUTPUT_INLINE_MAX_MB=10, OUTPUT_INLINE_BUDGET_MB=15 (total inline
budget per job), S3_OUTPUT_PREFIX=outputs, S3_URL_TTL_S=3600.

Returns: {"outputs": [{"filename": "...mp4", "b64"|"url": ...}], "final_prompt": "...", "seed": n}
Health:  {"action": "health"}
"""

import asyncio
import base64
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import runpod

COMFY_DIR = os.environ.get("COMFYUI_DIR", "/workspace/runpod-slim/ComfyUI")
COMFY_URL = "http://127.0.0.1:8188"
WORKFLOWS_DIR = Path(os.environ.get("WORKFLOWS_DIR", "/opt/workflows"))
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "1800"))
# Max seconds to wait for ComfyUI to finish booting before aborting the job.
BOOT_TIMEOUT_S = int(os.environ.get("BOOT_TIMEOUT_S", "600"))
# Max seconds a queued prompt may run without completing before we interrupt
# it and fail fast (hang detection). Should be < platform execution timeout.
STALL_TIMEOUT_S = int(os.environ.get("STALL_TIMEOUT_S", "540"))

# Output delivery: base64 inline under these limits, S3 offload above them.
# RunPod's job-done callback drops results over ~20MB, so keep headroom.
INLINE_MAX = int(os.environ.get("OUTPUT_INLINE_MAX_MB", "10")) * 1024 * 1024
INLINE_BUDGET = int(os.environ.get("OUTPUT_INLINE_BUDGET_MB", "15")) * 1024 * 1024
S3_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "outputs")

WORKFLOWS = {
    "normal": "MiniMaxH3-SelfLift-Sharpen-R2VA.json",
}

# Node ids shared by both workflows
N_LORA = "145"        # LoraLoaderModelOnly
N_LOADERS = ["150", "222", "223"]  # template LoadImage nodes (replaced per job)
MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS = 9, 3, 3
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
N_SECONDS = "132"     # PrimitiveFloat (duration → frame count)
N_RES = "115"         # ResolutionSelector
N_R2V = "136"         # MiniMaxH3ReferenceToVideo
N_SCHED = "124"       # BasicScheduler
N_SAMPLER = "193"     # SelfLiftAvatarH3Sampler


# ── ComfyUI plumbing ──────────────────────────────────────────────────────────

def _http(path, payload=None, timeout=30):
    url = COMFY_URL + path
    try:
        if payload is None:
            return urllib.request.urlopen(url, timeout=timeout).read()
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout).read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"ComfyUI {path} → HTTP {e.code}: {body}") from e


_BOOT_DONE = threading.Event()
_BOOT_ERR = None


def _boot_comfy():
    cmd = ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188",
           "--disable-auto-launch"] + os.environ.get("COMFYUI_EXTRA_ARGS", "").split()
    t0 = time.time()
    print(f"[handler] starting ComfyUI: {' '.join(cmd)} (cwd={COMFY_DIR})", flush=True)
    log = open("/workspace/comfy.log", "ab", buffering=0)
    proc = subprocess.Popen(cmd, cwd=COMFY_DIR, stdout=log, stderr=subprocess.STDOUT)
    print(f"[handler] ComfyUI pid={proc.pid}, full log -> /workspace/comfy.log", flush=True)
    deadline = t0 + 600
    last_note = t0
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"ComfyUI exited during boot (code={rc}) after {time.time()-t0:.0f}s")
        try:
            _http("/system_stats", timeout=5)
            print(f"[handler] ComfyUI ready in {time.time()-t0:.0f}s", flush=True)
            return proc
        except Exception:
            # Periodic heartbeat so a slow boot isn't a black box.
            if time.time() - last_note >= 30:
                print(f"[handler] waiting for ComfyUI... {time.time()-t0:.0f}s elapsed",
                      flush=True)
                last_note = time.time()
            time.sleep(2)
    raise RuntimeError(f"ComfyUI did not become ready in 600s (pid={proc.pid})")


def _boot_in_bg():
    # Boot on a thread so `health` answers even without models / volume.
    global _BOOT_ERR
    try:
        _boot_comfy()
    except Exception as exc:
        _BOOT_ERR = exc
        print(f"[handler] ComfyUI boot failed: {exc}", flush=True)
    finally:
        _BOOT_DONE.set()


async def _ensure_comfy():
    ok = await asyncio.to_thread(_BOOT_DONE.wait, BOOT_TIMEOUT_S)
    if not ok:
        raise TimeoutError(
            f"ComfyUI did not finish booting within {BOOT_TIMEOUT_S}s "
            "(stuck during startup — see /workspace/comfy.log)")
    if _BOOT_ERR is not None:
        raise RuntimeError(f"ComfyUI failed to boot: {_BOOT_ERR} (see /workspace/comfy.log)")


def _media(item, default_ext):
    """item: URL | base64 | data URI | {"url"|"b64": ..., "filename": ...} → (bytes, ext, opts)."""
    opts = item if isinstance(item, dict) else {}
    value = (opts.get("url") or opts.get("b64") or opts.get("data")) if opts else item
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"bad media item: {str(item)[:80]}")
    ext = Path(urllib.parse.urlparse(opts.get("filename") or value).path).suffix.lower() \
        if (opts.get("filename") or value.startswith("http")) else ""
    if value.startswith("http"):
        data = urllib.request.urlopen(value, timeout=300).read()
    else:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        data = base64.b64decode(value)
    return data, (ext if 1 < len(ext) <= 5 else default_ext), opts


def _upload(filename, data):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(COMFY_URL + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read()).get("name", filename)


# ── Workflow patching ─────────────────────────────────────────────────────────

def _load_workflow(name):
    name = WORKFLOWS.get(str(name or "normal").lower(), name)
    path = WORKFLOWS_DIR / name
    if not path.is_file():
        raise RuntimeError(f"workflow not found: {name} (use 'normal')")
    return json.loads(path.read_text())


def _drop_nodes(prompt, dead):
    """Delete nodes and every input that links to them."""
    for nid in dead:
        prompt.pop(nid, None)
    for node in prompt.values():
        inp = node.get("inputs", {})
        for k in [k for k, v in inp.items() if isinstance(v, list) and v and str(v[0]) in dead]:
            del inp[k]


def _set(prompt, nid, key, value):
    if value is not None and nid in prompt:
        prompt[nid]["inputs"][key] = value


def _as_list(v):
    return [] if v is None else (v if isinstance(v, list) else [v])


def _wire_references(prompt, job):
    """Replace the template's 3 LoadImage nodes with per-job loaders:
    up to 9 images, 3 videos (+ their soundtracks) and 3 standalone audios."""
    images, videos, audios = (_as_list(job.get(k)) for k in ("images", "videos", "audios"))
    if len(images) > MAX_IMAGES or len(videos) > MAX_VIDEOS or len(audios) > MAX_AUDIOS:
        raise RuntimeError(f"limits: images ≤{MAX_IMAGES}, videos ≤{MAX_VIDEOS}, audios ≤{MAX_AUDIOS}")
    if not (images or videos or audios):
        raise RuntimeError("need at least one reference: images / videos / audios")

    _drop_nodes(prompt, set(N_LOADERS))
    r2v = prompt[N_R2V]["inputs"]
    tag = uuid.uuid4().hex[:8]

    for i, item in enumerate(images):
        data, ext, _ = _media(item, ".png")
        nid = f"img{i}"
        prompt[nid] = {"class_type": "LoadImage", "inputs": {"image": _upload(f"ref_{tag}_img{i + 1}{ext}", data)}}
        r2v[f"ref_images.ref_image_{i}"] = [nid, 0]

    for i, item in enumerate(videos):
        data, ext, opts = _media(item, ".mp4")
        if ext not in VIDEO_EXT:
            ext = ".mp4"
        nid = f"vid{i}"
        prompt[nid] = {"class_type": "VHS_LoadVideo", "inputs": {
            "video": _upload(f"ref_{tag}_vid{i + 1}{ext}", data),
            "force_rate": 24, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": int(opts.get("max_frames", 360)),   # 15 s @ 24 fps
            "skip_first_frames": int(opts.get("skip_frames", 0)), "select_every_nth": 1,
        }}
        r2v[f"ref_videos.ref_video_{i}"] = [nid, 0]
        if opts.get("audio", True):
            r2v[f"ref_video_audios.ref_video_audio_{i}"] = [nid, 2]

    for i, item in enumerate(audios):
        data, ext, opts = _media(item, ".wav")
        if ext not in AUDIO_EXT:
            ext = ".wav"
        nid = f"aud{i}"
        prompt[nid] = {"class_type": "VHS_LoadAudioUpload", "inputs": {
            "audio": _upload(f"ref_{tag}_aud{i + 1}{ext}", data),
            "start_time": float(opts.get("start", 0)), "duration": float(opts.get("duration", 0)),
        }}
        r2v[f"ref_audios.ref_audio_{i}"] = [nid, 0]
    return len(images), len(videos), len(audios)


def _patch(prompt, job):
    if not job.get("prompt"):
        raise RuntimeError("input.prompt is required")
    _wire_references(prompt, job)

    seed = job.get("seed")
    seed = random.randrange(0, 2**32) if seed is None else int(seed) % (2**32)
    _set(prompt, N_SAMPLER, "seed", seed)

    # No LLM prompt enhancement: the user's text goes straight into H3 — use the
    # <Picture i> / <Video k> / <Audio j> tags yourself.
    prompt[N_R2V]["inputs"]["prompt"] = job["prompt"]

    if job.get("seconds") is not None:
        _set(prompt, N_SECONDS, "value", float(job["seconds"]))
    _set(prompt, N_RES, "aspect_ratio", job.get("aspect_ratio"))
    _set(prompt, N_RES, "megapixels", job.get("megapixels"))
    _set(prompt, N_SCHED, "steps", job.get("steps"))
    _set(prompt, N_LORA, "strength_model", job.get("lora_strength"))
    for key in ("transition_step", "lowres_scale", "cfg", "highres_tiling"):
        _set(prompt, N_SAMPLER, key, job.get(key))

    for key, value in (job.get("params") or {}).items():
        nid, _, widget = key.rpartition(":")
        if nid in prompt and widget:
            prompt[nid]["inputs"][widget] = value
    return prompt, seed


# ── Run + collect ─────────────────────────────────────────────────────────────

def _interrupt(prompt_id):
    for path, payload in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {})):
        try:
            _http(path, payload, timeout=5)
        except Exception as exc:
            print(f"[handler] {path} failed during cancel: {exc}", flush=True)


async def _queue_and_wait(prompt):
    resp = json.loads(await asyncio.to_thread(
        _http, "/prompt", {"prompt": prompt, "client_id": uuid.uuid4().hex}, 60))
    prompt_id = resp["prompt_id"]
    t0 = time.time()
    stall_deadline = t0 + STALL_TIMEOUT_S
    hard_deadline = t0 + JOB_TIMEOUT_S
    try:
        while time.time() < hard_deadline:
            hist = json.loads(await asyncio.to_thread(_http, f"/history/{prompt_id}", None, 30))
            entry = hist.get(prompt_id)
            status = (entry or {}).get("status", {})
            if status.get("status_str") == "error":
                msgs = " | ".join(str(m) for m in status.get("messages", []))
                raise RuntimeError(f"ComfyUI job failed: {msgs[:1500]}")
            if status.get("completed"):
                return entry
            # Hang detection: no completion within STALL_TIMEOUT_S → abort fast
            # instead of hanging until the platform kills the worker.
            if time.time() >= stall_deadline:
                await asyncio.to_thread(_interrupt, prompt_id)
                raise TimeoutError(
                    f"prompt {prompt_id} stalled: no result after {STALL_TIMEOUT_S}s "
                    f"(interrupted; see /workspace/comfy.log)")
            await asyncio.sleep(3)
    except asyncio.CancelledError:
        await asyncio.to_thread(_interrupt, prompt_id)
        raise
    await asyncio.to_thread(_interrupt, prompt_id)
    raise TimeoutError(f"prompt {prompt_id} exceeded {JOB_TIMEOUT_S}s")


def _s3_cfg(job):
    cfg = job.get("s3Config") or (job.get("input") or {}).get("s3Config")
    if isinstance(cfg, dict) and cfg.get("endpointUrl") and cfg.get("accessId"):
        return cfg
    return None


def _s3_client(cfg):
    import boto3  # lazy: only needed when S3 offload is actually used
    from botocore.config import Config
    host = urllib.parse.urlparse(cfg["endpointUrl"]).netloc
    region = cfg.get("region") or host.split(".", 1)[0].removeprefix("s3api-") or "us-east-1"
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpointUrl"],
        aws_access_key_id=cfg["accessId"],
        aws_secret_access_key=cfg["accessSecret"],
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )


def _emit_file(filename, data, job, inline_used):
    """Small outputs inline as b64; large ones upload to the job's s3Config
    storage (e.g. the RunPod network volume S3 API) and return a presigned URL."""
    oversized = len(data) > INLINE_MAX or inline_used[0] + len(data) > INLINE_BUDGET
    s3c = _s3_cfg(job)
    if oversized and s3c:
        try:
            key = f"{S3_PREFIX}/{job.get('id', 'job')}/{filename}"
            client = _s3_client(s3c)
            client.put_object(Bucket=s3c["bucketName"], Key=key, Body=data)
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3c["bucketName"], "Key": key},
                ExpiresIn=int(os.environ.get("S3_URL_TTL_S", "3600")),
            )
            print(f"[handler] {filename} -> s3://{s3c['bucketName']}/{key} ({len(data)/1e6:.1f}MB)", flush=True)
            return {"filename": filename, "url": url}
        except Exception as e:
            print(f"[handler] s3 upload failed for {filename}: {e} — falling back to inline", flush=True)
    elif oversized:
        print(f"[handler] {filename}: {len(data)/1e6:.1f}MB over inline limit, no s3Config — "
              "result may be dropped by RunPod", flush=True)
    inline_used[0] += len(data)
    return {"filename": filename, "b64": base64.b64encode(data).decode()}


def _collect_new_files(since_ts, job, inline_used):
    """Fallback: video files written under ComfyUI's output dir since the job ran."""
    out_dir = Path(COMFY_DIR) / "output"
    try:
        files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.stat().st_mtime >= since_ts)
    except Exception:
        return []
    out = []
    for p in files:
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        out.append(_emit_file(p.name, p.read_bytes(), job, inline_used))
    return out


def _collect(entry, job):
    outputs, inline_used = [], [0]
    # No LLM enhancement: the final prompt is the user's input text.
    final_prompt = job.get("prompt")
    for nid, out in (entry.get("outputs") or {}).items():
        print(f"[handler] output node {nid}: keys={list(out.keys())}", flush=True)
        for key in ("gifs", "videos", "images"):
            for item in out.get(key) or []:
                if not str(item.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov", ".gif")):
                    continue  # skip VHS preview pngs
                q = (f"filename={urllib.parse.quote(item['filename'])}"
                     f"&subfolder={urllib.parse.quote(item.get('subfolder', ''))}&type={item.get('type', 'output')}")
                data = _http(f"/view?{q}", timeout=300)
                outputs.append(_emit_file(item["filename"], data, job, inline_used))
    return outputs, final_prompt, inline_used


async def handler(job):
    inputs = job.get("input") or {}
    if inputs.get("action") == "health":
        return {
            "status": "ok",
            "comfyui": "failed" if _BOOT_ERR else ("ready" if _BOOT_DONE.is_set() else "booting"),
            "comfyui_error": str(_BOOT_ERR) if _BOOT_ERR else None,
            "workflows": sorted(p.name for p in WORKFLOWS_DIR.glob("*.json")),
            "volume_mounted": os.path.isdir("/runpod-volume"),
        }
    await _ensure_comfy()
    wf = _load_workflow(inputs.get("workflow"))
    prompt, seed = await asyncio.to_thread(_patch, wf, inputs)
    print(f"[handler] job {job.get('id')}: {len(prompt)} nodes, seed {seed}", flush=True)
    started = time.time()
    comfy_log = Path("/workspace/comfy.log")
    log_mark = comfy_log.stat().st_size if comfy_log.is_file() else 0
    entry = await _queue_and_wait(prompt)
    outputs, final_prompt, inline_used = await asyncio.to_thread(_collect, entry, job)
    try:  # surface ComfyUI warnings/errors from this job in the worker log
        with open(comfy_log, "rb") as f:
            f.seek(log_mark)
            interesting = [l for l in f.read().decode(errors="replace").splitlines()
                           if re.search(r"warn|error|missing|unexpected|fail|dtype|clip|traceback", l, re.I)]
            for l in interesting[-40:]:
                print(f"[comfy] {l}", flush=True)
    except Exception:
        pass
    if not outputs:
        outputs = await asyncio.to_thread(_collect_new_files, started, job, inline_used)
        print(f"[handler] fallback sweep — {len(outputs)} file(s) in output/", flush=True)
    print(f"[handler] done in {time.time() - started:.1f}s — {len(outputs)} video(s)", flush=True)
    if not outputs:
        raise RuntimeError("workflow finished but produced no video (see /workspace/comfy.log)")
    return {"outputs": outputs, "final_prompt": final_prompt, "seed": seed}


# Module scope (not under `if __name__ == "__main__"`): RunPod's pre-deploy
# check statically scans for unindented `import runpod` and
# `runpod.serverless.start(` — an indented call is invisible to it.
threading.Thread(target=_boot_in_bg, daemon=True).start()

runpod.serverless.start({"handler": handler})
