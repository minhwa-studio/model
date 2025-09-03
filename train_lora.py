import os, sys, subprocess, urllib.request, ssl
from pathlib import Path
from huggingface_hub import login
import dotenv

# ----------------------------
# 0) 환경/토큰
# ----------------------------
dotenv.load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print(":x: 'HUGGINGFACE_TOKEN'이 없습니다. .env에 추가하세요.")
    sys.exit(1)
login(token=hf_token)

ROOT = Path(__file__).resolve().parent
train_script = ROOT / "train_dreambooth_lora.py"

# ----------------------------
# 1) 학습 스크립트 다운로드
# ----------------------------
if not train_script.exists():
    print("LoRA 학습 스크립트를 다운로드합니다...")
    url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py"
    # 인증서 문제 우회(사내망 등)
    ssl_ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, context=ssl_ctx) as r, open(train_script, "wb") as f:
            f.write(r.read())
    except Exception as e:
        print(f":x: 다운로드 실패: {e}")
        sys.exit(1)
    print("다운로드 완료 ✅")

# ----------------------------
# 2) resume 체크포인트 존재 여부
# ----------------------------
resume_dir = ROOT / "sd15_lora_minhwa" / "checkpoint-15000"
resume_args = []
if resume_dir.exists():
    resume_args = [f"--resume_from_checkpoint={resume_dir.name}"]

# ----------------------------
# 3) accelerate로 학습 실행
# ----------------------------
print("스타일 학습을 시작합니다... 🚀")
accelerate_cmd = [
    sys.executable, "-m", "accelerate", "launch",
    "--mixed_precision=fp16",              # Windows/NVIDIA에서 fp16 권장
    str(train_script),
    "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
    "--instance_data_dir=./minhwa/data",
    "--output_dir=./sd15_lora_minhwa",
    "--instance_prompt", "<minhwastyle>",
    "--resolution=512",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=8",
    "--learning_rate=5e-5",
    "--lr_scheduler=cosine",
    "--lr_warmup_steps=0",
    "--max_train_steps=52220",
    "--checkpointing_steps=1000",
    "--seed=2025",
    "--gradient_checkpointing",
]
accelerate_cmd += resume_args  # checkpoint-15000이 있을 때만 추가

try:
    subprocess.run(accelerate_cmd, check=True, cwd=str(ROOT))
    print("학습 완료 ✅")
except FileNotFoundError as e:
    print(":x: venv에 accelerate가 없거나 PATH 문제입니다.")
    print("해결: venv 활성화 후 아래 설치\n  python -m pip install accelerate==0.31.0")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f":x: 학습 중 오류: {e}")
    sys.exit(1)
