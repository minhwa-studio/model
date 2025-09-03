from diffusers import StableDiffusionPipeline

# ① 베이스 모델 ID와 저장 경로 지정
repo_id = "runwayml/stable-diffusion-v1-5"
save_dir = r"C:\Users\Administrator\Desktop\minhwa\model\base"

# ② 모델 다운로드 및 저장
pipe = StableDiffusionPipeline.from_pretrained(
    repo_id,
    torch_dtype="auto"
)
pipe.save_pretrained(save_dir)

print(f"✅ SD v1.5 base model saved to: {save_dir}")
