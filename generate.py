# -*- coding: utf-8 -*-
import os
from pathlib import Path

import dotenv
import torch
import numpy as np
import cv2
from PIL import Image

from huggingface_hub import login
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,          # 남겨두지만 본 프리셋은 EulerAncestral로 교체
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image


# ==============================
# 0) 설정 (필요시 변경)
# ==============================
SOURCE_IMAGE = "./inputs/img1.jpg"
OUT_DIR = Path("./outputs")

# 실행 스위치
RUN_MID_PRESET = True   # 중간 톤 프리셋 1장
RUN_GRID       = True   # 중간대 그리드 여러 장(아래 파라미터 참고)

# 재현성 고정 시드
SEED = 42

# ==============================
# 1) Hugging Face 로그인
# ==============================
dotenv.load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print(":x: 'HUGGINGFACE_TOKEN'/.env 가 없습니다. 예: HUGGINGFACE_TOKEN=hf_xxx")
    raise SystemExit(1)
login(token=hf_token)


# ==============================
# 2) 기본 장치/정밀도 & 모델 로드
# ==============================
use_cuda = torch.cuda.is_available()
precision = torch.float16 if use_cuda else torch.float32  # diffusers==0.27.2는 torch_dtype 사용

# ControlNet (Canny)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=precision,
)

# SD 1.5 + ControlNet 파이프라인
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=precision,
)

# 부드러운 톤을 위한 VAE 교체
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=precision)
pipe.vae = vae
pipe.enable_vae_slicing()

# 스케줄러: 미드톤/대비 과다 방지를 위해 EulerAncestral 권장
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 메모리 최적화
pipe.enable_attention_slicing()
if use_cuda:
    pipe.to("cuda")

# LoRA 로드 (peft 필요)
pipe.load_lora_weights(
    "./sd15_lora_minhwa/checkpoint-15000",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="minhwa",
)
pipe.set_adapters(["minhwa"], adapter_weights=[1.00])  # 루프에서 변경 가능


# ==============================
# 3) 유틸 함수
# ==============================
def get_canny_image(image_path, low_threshold=20, high_threshold=60):
    """원본 이미지를 Canny 엣지(3채널 RGB)로 변환"""
    img = load_image(image_path)    # PIL
    arr = np.array(img)             # HWC, uint8
    edges = cv2.Canny(arr, low_threshold, high_threshold)
    edges = np.repeat(edges[:, :, None], 3, axis=2)
    return Image.fromarray(edges).convert("RGB")


def balance_tone(pil_img, desat=0.93, gamma=0.96, clahe_clip=1.2):
    """
    아주 약한 톤 보정:
      - desat: 채도 93%만 유지(7% 감소)
      - gamma: 0.96 → 미드톤 살짝 리프트(밝기↑)
      - clahe_clip: CLAHE로 L채널 미드톤 조금 올림
    """
    if desat is None and gamma is None and clahe_clip is None:
        return pil_img

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

    if desat is not None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(desat), 0, 1)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if clahe_clip is not None:
        lab = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

    if gamma is not None:
        bgr = np.clip(bgr ** float(gamma), 0, 1)

    rgb = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_overwrite(pil_img: Image.Image, path: Path):
    """같은 파일명이 있으면 삭제 후 저장(덮어쓰기 보장)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    pil_img.save(path)


# ==============================
# 4) 프롬프트 (중간 톤 지시)
# ==============================
prompt = (
    "<minhwastyle>, traditional Korean minhwa painting, joseon style, "
    "soft pastel tones, muted colors, low saturation, soft contrast, balanced exposure, "
    "watercolor ink wash texture, calm and airy mood, elegant korean aesthetics"
)
negative_prompt = (
    "photo, realistic, 3d render, neon, vivid, high saturation, intense colors, "
    "harsh contrast, crushed blacks, blown highlights, dark, dim, muddy colors"
)

# 공통
OUT_DIR.mkdir(parents=True, exist_ok=True)
generator = torch.Generator(device="cuda" if use_cuda else "cpu").manual_seed(SEED)


# ==============================
# 5) A안 — 중간 톤 프리셋 1장
# ==============================
if RUN_MID_PRESET:
    CFG = 6.0
    STEPS = 36
    CN = 0.46
    LOW, HIGH = 22, 64
    LORA_W = 1.00

    base_image = get_canny_image(SOURCE_IMAGE, LOW, HIGH)
    pipe.set_adapters(["minhwa"], adapter_weights=[LORA_W])

    img = pipe(
        prompt=prompt,
        image=base_image,
        negative_prompt=negative_prompt,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        controlnet_conditioning_scale=CN,
        generator=generator,
    ).images[0]

    img = balance_tone(img, desat=0.93, gamma=0.96, clahe_clip=1.2)
    save_overwrite(img, OUT_DIR / "minhwa_mid_preset.png")
    print("✔ Saved:", OUT_DIR / "minhwa_mid_preset.png")


# ==============================
# 6) B안 — 중간대 그리드(기본 18장)
#    12장으로 줄이려면 CN_SCALES를 2개로 줄이면 됨.
# ==============================
if RUN_GRID:
    LORA_STRENGTHS = [0.95, 1.00, 1.05]                # LoRA 강도
    CN_SCALES      = [0.42, 0.46, 0.50]                # ControlNet 스케일(중간대)
    CANNY_SETS     = [(20, 60), (25, 70)]              # Canny 두 구간 → 3*3*2 = 18장

    NUM_STEPS      = 36
    GUIDANCE_SCALE = 6.0
    DESAT_FACTOR   = 0.93
    GAMMA          = 0.96
    CLAHE_CLIP     = 1.2

    total = len(LORA_STRENGTHS) * len(CN_SCALES) * len(CANNY_SETS)
    idx = 0

    for (low, high) in CANNY_SETS:
        subdir = OUT_DIR / f"grid_canny_{low}-{high}"
        subdir.mkdir(parents=True, exist_ok=True)

        base_image = get_canny_image(SOURCE_IMAGE, low_threshold=low, high_threshold=high)

        for w in LORA_STRENGTHS:
            pipe.set_adapters(["minhwa"], adapter_weights=[w])

            for cn in CN_SCALES:
                idx += 1
                print(f"[{idx:02d}/{total}] Canny={low}-{high}, LoRA={w:.2f}, CN={cn:.2f}")

                img = pipe(
                    prompt=prompt,
                    image=base_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    controlnet_conditioning_scale=cn,
                    generator=generator,  # 고정 시드 → 조건 차이만 반영
                ).images[0]

                img = balance_tone(img, desat=DESAT_FACTOR, gamma=GAMMA, clahe_clip=CLAHE_CLIP)

                fname = subdir / f"minhwa_c{low}-{high}_w{w:.2f}_cn{cn:.2f}.png"
                save_overwrite(img, fname)
                print("   -> Saved:", fname)

    print(f"✔ Grid done. dir = {OUT_DIR.resolve()}")