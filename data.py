import os
import zipfile
from PIL import Image
import tqdm

# .env 파일에서 환경 변수를 로드하는 함수를 추가합니다.
def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("❌ 경고: 'python-dotenv' 라이브러리가 설치되지 않았습니다. 'pip install python-dotenv'를 실행해주세요.")
        print("환경 변수를 수동으로 설정해야 할 수 있습니다.")
        
# ------------------------------------------------
# 1. 환경 설정 및 데이터셋 압축 해제
# ------------------------------------------------
print("환경 설정 및 다운로드된 데이터셋 압축 해제를 시작합니다. 🖼️")

# .env 파일 로드
load_env()

# Hugging Face 토큰을 환경 변수에서 가져옵니다.
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("❌ 오류: 환경 변수 'HUGGINGFACE_TOKEN'을 찾을 수 없습니다.")
    print(".env 파일에 'HUGGINGFACE_TOKEN=your_token_here' 형식으로 추가하거나 환경 변수를 설정해주세요.")
    exit(1)

try:
    from huggingface_hub import login
    login(token=hf_token)
except ImportError:
    print("❌ 오류: huggingface_hub 라이브러리가 설치되지 않았습니다. 'pip install huggingface_hub'를 실행해주세요.")
    exit(1)


# 처리할 ZIP 파일 목록과 압축 해제할 폴더를 지정하세요.
zip_files_path = ["minhwa1.zip", "minhwa2.zip", "minhwa3.zip"]
extract_dir = "minhwa/original"  # 원본 이미지는 'minhwa/original'에 압축 해제
final_data_dir = "minhwa/data"   # 모든 이미지를 'minhwa/data'에 저장

# 'minhwa' 폴더가 없으면 생성
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(final_data_dir, exist_ok=True)

# 여러 ZIP 파일 압축 해제
for zip_file_path in zip_files_path:
    try:
        if not os.path.exists(zip_file_path):
            print(f"❌ 오류: '{zip_file_path}' 파일을 찾을 수 없습니다. 직접 다운로드하여 현재 폴더에 넣어주세요.")
            continue
            
        print(f"\n데이터셋({zip_file_path}) 압축을 해제합니다...")
        
        # 'encoding' 옵션을 제거하고 파일명을 수동으로 디코딩하여 한글 깨짐 문제 해결
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # zip 파일 내의 모든 파일을 지정된 폴더로 이동
            for member in tqdm.tqdm(zip_ref.infolist(), desc=f"'{zip_file_path}' 압축 해제 중"):
                try:
                    # 파일명 인코딩을 수동으로 처리
                    # cp437 인코딩을 euc-kr로 디코딩
                    member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
                    zip_ref.extract(member, extract_dir)
                except Exception as e:
                    # tqdm.tqdm.write() 대신 print() 사용
                    print(f"  경고: '{member.filename}' 압축 해제 중 문제 발생 - {e}. 건너뜁니다.")

        print("압축 해제 완료. ✅")

    except zipfile.BadZipFile:
        print(f"❌ 오류: '{zip_file_path}' 파일이 올바른 ZIP 형식이 아닙니다. 파일이 손상되었을 수 있습니다.")
    except Exception as e:
        print(f"❌ 오류: '{zip_file_path}' 압축 해제 중 문제 발생 - {e}")

# ------------------------------------------------
# 2. 이미지 리사이징 및 증강
# ------------------------------------------------
def process_and_augment_images(root_dir, output_dir, max_size=512):
    print("\n이미지 리사이징 및 증강 작업을 시작합니다... 🖼️")
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                all_files.append(os.path.join(dirpath, filename))

    if not all_files:
        print(f"❌ 오류: '{root_dir}' 디렉터리에서 이미지 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")
        return

    # tqdm 진행률 바에 리사이징된 파일 정보를 표시합니다.
    with tqdm.tqdm(total=len(all_files), desc=f"리사이징 및 증강 중") as pbar:
        for file_path in all_files:
            try:
                with Image.open(file_path) as img:
                    # 1. 리사이징
                    # 비율 유지 리사이징
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    # 2. 증강된 이미지 생성 및 저장
                    base_filename, ext = os.path.splitext(os.path.basename(file_path))
                    
                    # 원본 이미지 저장
                    img.save(os.path.join(output_dir, f"{base_filename}_original{ext}"), quality=95)
                    
                    # 좌우 반전
                    img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_flipped.save(os.path.join(output_dir, f"{base_filename}_h_flipped{ext}"), quality=95)

                    # 상하 반전
                    img_flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
                    img_flipped_v.save(os.path.join(output_dir, f"{base_filename}_v_flipped{ext}"), quality=95)
                    
                    # 90도 회전
                    img_rot90 = img.rotate(90, expand=True)
                    img_rot90.save(os.path.join(output_dir, f"{base_filename}_rot90{ext}"), quality=95)
                    
                    # 180도 회전
                    img_rot180 = img.rotate(180, expand=True)
                    img_rot180.save(os.path.join(output_dir, f"{base_filename}_rot180{ext}"), quality=95)
                    
                    # 270도 회전
                    img_rot270 = img.rotate(270, expand=True)
                    img_rot270.save(os.path.join(output_dir, f"{base_filename}_rot270{ext}"), quality=95)

            except Exception as e:
                # tqdm.tqdm.write() 대신 print() 사용
                print(f"❌ 오류: {file_path} 처리 중 문제 발생: {e}")
            
            pbar.update(1)

    print("\n모든 이미지 리사이징 및 증강 작업이 완료되었습니다. ✅")

# 캡셔닝을 수행할 이미지 디렉터리를 지정합니다.
image_directory_to_caption = "minhwa/original"
final_data_directory = "minhwa/data"

if os.path.isdir(image_directory_to_caption):
    process_and_augment_images(image_directory_to_caption, final_data_directory, max_size=512)
else:
    print(f"오류: 원본 이미지 디렉토리 '{image_directory_to_caption}'를 찾을 수 없습니다. 스크립트를 먼저 실행하여 데이터를 준비하세요.")