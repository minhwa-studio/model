# -*- coding: utf-8 -*-
"""
민화 수채화 스타일 (객체 마스크 없이, 엣지맵만으로 구도 고정)
- LoRA: ./sd15_lora_minhwa/checkpoint-15000/pytorch_lora_weights.safetensors
- HF: 로그인만 유지, 로딩은 캐시(local_files_only=True)
"""

import os
from pathlib import Path
from urllib.parse import urlparse

import torch, numpy as np, cv2
from PIL import Image
import dotenv
from huggingface_hub import login
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)

# -----------------------------
# 경로/설정
# -----------------------------
SOURCE_IMAGE = "./inputs/dog.jpg"
OUT_PATH     = "./outputs/minhwa_watercolor.png"

LORA_FILE = Path("./sd15_lora_minhwa/checkpoint-15000/pytorch_lora_weights.safetensors")
assert LORA_FILE.exists(), f"LoRA 없음: {LORA_FILE}"

SEED, STEPS, CFG = 42, 38, 6.0
ADAPTER_WEIGHT   = 0.70          # LoRA 약화(주제 덮어쓰기 방지)
CN_SCALE         = 1.10          # ControlNet 강하게(구도 고정)
GUIDE_START, GUIDE_END = 0.0, 1.0

# ---- 전역 엣지 제어(마스크 없음) ----
TARGET_EDGE_DENSITY = 0.0045     # 0.45% 목표(필요시 0.003~0.006 사이로 조절)
DENSITY_TOL         = 0.0015
CANNY_LOW_INIT, CANNY_HIGH_INIT, CANNY_MAX_ITER = 25, 70, 8

# 노이즈 정리(자잘한 조각 제거/얇게)
MIN_COMPONENT_AREA = 140
EDGE_ERODE_ITERS   = 1

# 수채화 스무딩(질감 억제)
MEANSHIFT_SP, MEANSHIFT_SR = 12, 24
BILATERAL_D,  BILATERAL_SC, BILATERAL_SS = 7, 45, 9

# -----------------------------
# HF 로그인(캐시만)
# -----------------------------
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN, "HUGGINGFACE_TOKEN(.env)에 토큰을 넣어주세요."
login(token=HF_TOKEN, add_to_git_credential=False)
os.environ["HF_HUB_OFFLINE"] = "1"
print("✅ HF login ok (cache only)")

use_cuda = torch.cuda.is_available()
precision = torch.float16 if use_cuda else torch.float32

def smart_load_image(p):
    pr = urlparse(str(p))
    if pr.scheme in ("http","https"):
        raise SystemExit("URL은 오프라인 모드 불가. 로컬 경로 사용.")
    path = Path(p)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    if not path.exists():
        d = Path(__file__).resolve().parent / "inputs"
        cs = [q for q in d.glob("*.*") if q.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
        assert cs, "inputs 폴더에 이미지 넣어주세요."
        print(f"⚠ SOURCE_IMAGE 없음 → {cs[0].name} 사용")
        path = cs[0]
    return Image.open(path).convert("RGB")

# -----------------------------
# 엣지맵(전역 밀도 기반)
# -----------------------------
def auto_canny_full(gray, low=CANNY_LOW_INIT, high=CANNY_HIGH_INIT,
                    target=TARGET_EDGE_DENSITY, tol=DENSITY_TOL, max_iter=CANNY_MAX_ITER):
    total = gray.size
    l, h = low, high
    for _ in range(max_iter):
        e = cv2.Canny(gray, l, h)
        dens = float(np.count_nonzero(e)) / float(total)
        if   dens > target + tol: l = min(int(l*1.20)+1, 170); h = min(int(h*1.20)+1, 250)
        elif dens < target - tol: l = max(int(l*0.85),   5);   h = max(int(h*0.85),   20)
        else: break
    return cv2.Canny(gray, l, h)

def post_clean_edges(edges):
    out = edges.copy()
    if EDGE_ERODE_ITERS > 0:
        out = cv2.erode(out, np.ones((3,3), np.uint8), iterations=EDGE_ERODE_ITERS)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((out>0).astype(np.uint8), connectivity=8)
    clean = np.zeros_like(out)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255
    return clean

def build_edge_input(pil):
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # 수채화 스무딩 → 배경/털 텍스처 억제
    ms  = cv2.pyrMeanShiftFiltering(bgr, sp=MEANSHIFT_SP, sr=MEANSHIFT_SR)
    sm  = cv2.bilateralFilter(ms, d=BILATERAL_D, sigmaColor=BILATERAL_SC, sigmaSpace=BILATERAL_SS)
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    # 전역 밀도 기반 Canny
    e = auto_canny_full(gray)
    e = post_clean_edges(e)
    e3 = np.repeat(e[:, :, None], 3, axis=2)
    return Image.fromarray(e3).convert("RGB")

def balance_tone(pil, desat=0.95, gamma=1.0):
    bgr=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR).astype(np.float32)/255.0
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV); hsv[...,1]*=float(desat)
    bgr=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR); bgr=np.clip(bgr**float(gamma),0,1)
    return Image.fromarray(cv2.cvtColor((bgr*255).astype(np.uint8),cv2.COLOR_BGR2RGB))

# -----------------------------
# 파이프라인(캐시)
# -----------------------------
try:
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                 torch_dtype=precision, local_files_only=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=precision,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                        torch_dtype=precision, local_files_only=True)
    pipe.vae = vae
except Exception as e:
    raise SystemExit("❌ 캐시에 모델이 없습니다. 인터넷 가능한 곳에서 한 번 받아 캐시 복사하세요.\n원인: " + str(e))

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing(); pipe.enable_attention_slicing()
pipe.load_lora_weights(str(LORA_FILE.parent), weight_name=LORA_FILE.name, adapter_name="minhwa")
pipe.set_adapters(["minhwa"], adapter_weights=[ADAPTER_WEIGHT])
if use_cuda: pipe.to("cuda")

# -----------------------------
# 프롬프트(주제 명시 + 인물 네거티브)
# -----------------------------
prompt = (
    "a dog sitting with a twig in mouth, "
    "<minhwastyle>, traditional Korean minhwa watercolor (damchae), "
    "soft ink outlines, light washes on hanji paper, muted pastel palette, calm mood"
)
negative_prompt = (
    "human, woman, geisha, portrait of person, photo, photorealistic, hdr, 3d render, "
    "oil painting, western style, neon, vivid, high saturation, harsh contrast, plastic, metallic, noise, artifacts"
)

# -----------------------------
# 생성
# -----------------------------
src_pil  = smart_load_image(SOURCE_IMAGE)
edge_pil = build_edge_input(src_pil)        # ★ 마스크 없이 전역 엣지
W, H = edge_pil.width, edge_pil.height

gen = torch.Generator(device="cuda" if use_cuda else "cpu").manual_seed(SEED)
img = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=edge_pil,
    num_inference_steps=STEPS,
    guidance_scale=CFG,
    controlnet_conditioning_scale=CN_SCALE,
    control_guidance_start=GUIDE_START,
    control_guidance_end=GUIDE_END,
    guess_mode=False,
    width=W, height=H,
    generator=gen,
).images[0]

img = balance_tone(img, desat=0.95, gamma=1.0)
out = Path(OUT_PATH); out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): out.unlink()
img.save(out)
print("✔ Saved:", out)