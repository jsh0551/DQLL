import os
import shutil
import json
from tqdm import tqdm

# 원본 폴더와 대상 폴더 경로 설정
for dir in ['train','test']:
    src_folder = f'data/TUSimple/MyTuSimpleLane/{dir}/img'
    dst_folder = f'data/ultralytics/images/{dir}'

    # 대상 폴더가 없으면 생성
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 원본 폴더 내의 파일을 대상 폴더로 복사
    for file_name in os.listdir(src_folder):
        src_file_path = os.path.join(src_folder, file_name)
        dst_file_path = os.path.join(dst_folder, file_name)
        
        # 파일 복사 (src_file_path에서 dst_file_path로)
        shutil.copy(src_file_path, dst_file_path)
        
    print(f"파일 복사 완료: {src_folder} -> {dst_folder}")

for dir in ['train','test']:
    file_path = f'data/TUSimple/MyTuSimpleLane/{dir}/bbox'
    dst_folder = f'data/ultralytics/labels/{dir}'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    file_names = os.listdir(file_path)
    pbar = tqdm(total=len(file_names))
    for fn in file_names:
        json_file_path = os.path.join(file_path, fn)
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        transformed_data = []

        for item in data:
            x1,y1,x2,y2 = item['points']
            cx,cy = (x1+x2)/2, (y1+y2)/2
            w,h = x2-x1, y2-y1
            transformed_points = [
                cx / 1280,
                cy / 720,
                w / 1280,
                h / 720
            ]
            
            # 변환된 데이터 포맷에 맞게 저장
            transformed_data.append([item["class"], *transformed_points])

        # 변환된 데이터를 test.txt 파일에 저장
        fn = fn.split('.')[0]
        with open(os.path.join(dst_folder,f'{fn}.txt'), 'w') as file:
            for line in transformed_data:
                file.write(' '.join(map(str, line)) + '\n')
        pbar.update(1)
    pbar.close()