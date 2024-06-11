import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_PATH = 'data/merged_data'
json_path = os.path.join(DATA_PATH, 'data.json')
half_image_path = os.path.join(DATA_PATH, 'half_images')
mask_path = os.path.join(DATA_PATH, 'mask')
mask_color_path = os.path.join(DATA_PATH, 'mask_color')
line_path = os.path.join(DATA_PATH, 'line')
bbox_path = os.path.join(DATA_PATH, 'bbox')
th_H = 0.04

def get_bbox(img, mask_color):
    bboxes, contours, contours_color, cropped_imgs = [], [], [], []
    cons, hier = cv2.findContours(mask_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    height = img.shape[0]
    for con in cons:
        x, y, w, h = cv2.boundingRect(con)
        if h < height*th_H:
            continue
        temp = np.zeros(img.shape)
        cv2.drawContours(temp, [con], 0, (0, 0, 255), 5)
        tmp_con = np.zeros((img.shape[0], img.shape[1]))
        tmp_conc = np.zeros((img.shape[0], img.shape[1]))
        tmp_con[temp[:, :, 2] == 255] = 1
        tmp_conc[temp[:, :, 2] == 255] = 255
        tmp_con = tmp_con[y:y+h, x:x+w].astype('uint8')
        tmp_conc = tmp_conc[y:y+h, x:x+w].astype('uint8')
        bbox = np.array([x, y, w, h]).astype(np.int16).tolist()
        cropped_img = img[y:y+h,x:x+w,:]
        bboxes.append(bbox)
        contours.append(tmp_con)
        contours_color.append(tmp_conc)
        cropped_imgs.append(cropped_img)
    return bboxes, contours, contours_color, cropped_imgs

def split_file_name(file_name, join = False):
    name = file_name[:-4]
    tmp = name.split('_')
    tmp_date, tmp_time = tmp[:2]
    tmp_info = tmp[2:]
    if join:
        return name
    return tmp_date, tmp_time, tmp_info

def add_word_to_file_name(file_name, word=None):
    tdate, ttime, tinfo = split_file_name(file_name)
    tinfo = '_'.join(tinfo)
    new_name = f'{tdate}_{ttime}_{tinfo}_{word}.png'
    return new_name

def save_line(file_name, contours, contours_color, cropped_imgs):
    con_names = []
    for i, (con, conc, cropped) in enumerate(zip(contours, contours_color, cropped_imgs)):
        rsz_con, rsz_conc, rsz_cropped = cv2.resize(con,(100,100)), cv2.resize(conc,(100,100)), cv2.resize(cropped,(100,100))
        con_name = add_word_to_file_name(file_name, i)
        con_mask_name = add_word_to_file_name(con_name, 'mask')
        con_mask_color_name = add_word_to_file_name(con_mask_name, 'color')
        con_only_name = split_file_name(con_name, True)
        con_path = os.path.join(line_path, 'ori', con_mask_name)
        conc_path = os.path.join(line_path, 'ori', con_mask_color_name)
        cropped_path = os.path.join(line_path, 'ori', con_name)
        rsz_con_path = os.path.join(line_path, 'resize', con_mask_name)
        rsz_conc_path = os.path.join(line_path, 'resize', con_mask_color_name)
        rsz_cropped_path = os.path.join(line_path, 'resize', con_name)
        cv2.imwrite(con_path, con)
        cv2.imwrite(conc_path, conc)
        cv2.imwrite(cropped_path, cropped)
        cv2.imwrite(rsz_con_path, rsz_con)
        cv2.imwrite(rsz_conc_path, rsz_conc)
        cv2.imwrite(rsz_cropped_path, rsz_cropped)
        con_names.append(con_only_name)
    return con_names

if __name__ == '__main__':
    os.makedirs(half_image_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(mask_color_path, exist_ok=True)
    os.makedirs(os.path.join(line_path,'ori'), exist_ok=True)
    os.makedirs(os.path.join(line_path,'resize'), exist_ok=True)
    os.makedirs(bbox_path, exist_ok=True)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    items = json_data['items']
    pbar = tqdm(total=len(items))
    all_con_names = []
    for item in items:
        h,w = item['image']['height'], item['image']['width']
        file_name = item['image']['file_name']
        file_path = os.path.join(DATA_PATH,item['image']['path'])
        only_name = split_file_name(file_name, True)
        img = cv2.imread(file_path)
        mask_img = np.zeros((h,w,3))
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        mask_color = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for poly in item['annotations']:
            label = poly['label']
            points = np.array(poly['points']).astype(np.uint16)
            for i in range(len(points)-1):
                p1 = points[i]
                p2 = points[i+1]
                cv2.line(mask_img, p1, p2, (255,255,255), thickness=4)
        # full
        mask[mask_img[:, :, 1] == 255] = 1
        mask_color[mask_img[:, :, 1] == 255] = 255
        bboxes, contours, contours_color, cropped_imgs = get_bbox(img, mask_color)
        # half
        half_img = img[h//2:,:,:]
        mask = mask[h//2:,:]
        mask_color = mask_color[h//2:,:]
        half_bboxes, half_contours, half_contours_color, half_cropped_imgs = get_bbox(half_img, mask_color)
        contours += half_contours
        contours_color += half_contours_color
        cropped_imgs += half_cropped_imgs
        if bboxes:
            bbox_info = {'file_name':file_name, 'bbox':half_bboxes}
            with open(os.path.join(bbox_path, f'{only_name}.json'), 'w') as f:
                json.dump(bbox_info, f)
            con_names = save_line(file_name, contours, contours_color, cropped_imgs)
            all_con_names += con_names
            cv2.imwrite(os.path.join(half_image_path,file_name), half_img)
            cv2.imwrite(os.path.join(mask_path,file_name), mask)
            cv2.imwrite(os.path.join(mask_color_path,file_name), mask_color)
        pbar.update(1)
    pbar.close()
    # split train/val
    train_list, val_list = train_test_split(all_con_names, test_size=0.3, random_state=42, shuffle=True)
    with open(os.path.join(DATA_PATH, 'train_line_list.json'), 'w') as f:
        json.dump(train_list, f)
    with open(os.path.join(DATA_PATH, 'val_line_list.json'), 'w') as f:
        json.dump(val_list, f)
    print(f'train line datasets : {len(train_list)}')
    print(f'val line datasets : {len(val_list)}')