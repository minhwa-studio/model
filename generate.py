import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

# ------------------------------------------------
# ControlNet 및 LoRA 모델 로드
# ------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# 2000 스텝까지 학습된 LoRA 가중치 적용
pipe.load_lora_weights("./sd15_lora_minhwa/checkpoint-2000", weight_name="pytorch_lora_weights.safetensors")

# 스케줄러 설정 (일반적으로 UniPCMultistepScheduler를 사용)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# ------------------------------------------------
# Canny 엣지 맵 생성 함수 (강도 조절)
# ------------------------------------------------
def get_canny_image(image_path):
    image = load_image(image_path)
    image = np.array(image)
    
    # Canny 엣지 맵의 강도 조절 (threshold를 낮춰 엣지를 약하게)
    low_threshold = 50
    high_threshold = 100
    
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

# 이미지 생성 파라미터 및 스크립트 실행

source_image_path = "./inputs/img1.jpg"
base_image = get_canny_image(source_image_path)

# 프롬프트 설정
prompt = "<minhwastyle> painting, oriental ink painting, watercolor, vibrant colors, subtle brush strokes, harmony, a bird on a branch, detailed background"
negative_prompt = "japanese style, chinese style, western painting, oil painting, impressionist, very dark, very bright, saturated colors, photography, bad art, blurry, ugly, duplicate"

# 이미지 생성
print("이미지 생성 시작...🎥")
generated_image = pipe(
    prompt,
    image=base_image,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8
).images[0]

# 생성된 이미지 저장
generated_image.save("./outputs/generated_minhwa_image.png")
print("\n이미지 생성 완료. 'generated_minhwa_image.png' 파일로 저장되었습니다.🎁")