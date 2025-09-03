from diffusers import ControlNetModel

repo_id = "lllyasviel/sd-controlnet-canny"
save_dir = r"C:\Users\Administrator\Desktop\minhwa\model\controlnet"

controlnet = ControlNetModel.from_pretrained(
    repo_id,
    torch_dtype="auto"
)
controlnet.save_pretrained(save_dir)

print(f"✅ ControlNet model saved to: {save_dir}")
