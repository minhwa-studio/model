import subprocess
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login
import dotenv
import os
# 허깅페이스 로그인
# .env 파일 로드
dotenv.load_dotenv()
# Hugging Face 토큰을 환경 변수에서 가져옵니다.
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print(":x: 오류: 환경 변수 'HUGGINGFACE_TOKEN'을 찾을 수 없습니다.")
    print(".env 파일에 'HUGGINGFACE_TOKEN=your_token_here' 형식으로 추가하거나 환경 변수를 설정해주세요.")
    exit(1)
try:
    from huggingface_hub import login
    login(token=hf_token)
except ImportError:
    print(":x: 오류: huggingface_hub 라이브러리가 설치되지 않았습니다. 'pip install huggingface_hub'를 실행해주세요.")
    exit(1)
# ------------------------------------------------
# LoRA 학습
# ------------------------------------------------
print("\nLoRA 학습 스크립트를 다운로드합니다...")
# '-k' 또는 '--insecure' 옵션을 추가하여 인증서 오류를 무시합니다.
subprocess.run(["curl", "-O", "-k", "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py"], check=True)
print("스크립트 다운로드 완료. :흰색_확인_표시:")
print("\n스타일 학습을 시작합니다... :로켓:")
try:
    subprocess.run([
        "accelerate", "launch",
        "--mixed_precision=bf16",
        "train_dreambooth_lora.py",
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        "--instance_data_dir=./minhwa/data",
        "--output_dir=./sd15_lora_minhwa",
        "--instance_prompt", '<minhwastyle>',
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
        "--resume_from_checkpoint=checkpoint-5000"
    ], check=True)
    print("\n학습이 성공적으로 완료")
except FileNotFoundError:
    print(":x: 오류: 'accelerate' 또는 'train_dreambooth_lora.py'를 찾을 수 없습니다. ")
    exit(1)
except subprocess.CalledProcessError as e:
    print(f":x: 학습 중 오류가 발생했습니다: {e}")
    exit(1)
