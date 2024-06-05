import json
import os
import cv2
import numpy as np
from tqdm import tqdm

DATA_PATH = 'data/merged_data'
json_path = os.path.join(DATA_PATH, 'data.json')
mask_path = os.path.join(DATA_PATH, 'mask')
mask_color_path = os.path.join(DATA_PATH, 'mask_color')
line_path = os.path.join(DATA_PATH, 'line')
line_mask_path = os.path.join(DATA_PATH, 'line_mask')
bbox_path = os.path.join(DATA_PATH, 'bbox')
th_H = 0.04

def get_bbox(img, mask_color):
    bboxes, contours, cropped_imgs = [], [], []
    cons, hier = cv2.findContours(mask_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    height = img.shape[0]
    for con in cons:
        x, y, w, h = cv2.boundingRect(con)
        if h < height*th_H:
            continue
        temp = np.zeros(img.shape)
        cv2.drawContours(temp, [con], 0, (0, 0, 255), -1)
        tmp_con = np.zeros((img.shape[0], img.shape[1]))
        tmp_con[temp[:, :, 2] == 255] = 255
        tmp_con = tmp_con[y:y+h, x:x+w].astype('uint8')
        bbox = np.array([x, y, w, h]).astype(np.int16).tolist()
        cropped_img = img[y:y+h,x:x+w,:]
        bboxes.append(bbox)
        contours.append(tmp_con)
        cropped_imgs.append(cropped_img)
    return bboxes, contours, cropped_imgs

def split_file_name(file_name, join = False):
    name = file_name[:-4]
    tmp_date, tmp_time, tmp_frame = name.split('_')
    if join:
        return f'{tmp_date}_{tmp_time}_{tmp_frame}'
    return tmp_date, tmp_time, tmp_frame

def add_word_to_file_name(file_name, word=None):
    tdate, ttime, tframe = split_file_name(file_name)
    new_name = f'{tdate}_{ttime}_{tframe}_{word}.png'
    return new_name

def save_line(file_name, contours, cropped_imgs):
    con_names = []
    for i, (con, cropped) in enumerate(zip(contours, cropped_imgs)):
        con_name = add_word_to_file_name(file_name, i)
        # con_only_name = split_file_name(con_name, True)
        con_path = os.path.join(line_mask_path, con_name)
        cropped_path = os.path.join(line_path, con_name)
        cv2.imwrite(con_path, con)
        cv2.imwrite(cropped_path, cropped)
        con_names.append(con_name)
    return con_names

if __name__ == '__main__':
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(mask_color_path, exist_ok=True)
    os.makedirs(line_path, exist_ok=True)
    os.makedirs(line_mask_path, exist_ok=True)
    os.makedirs(bbox_path, exist_ok=True)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    items = json_data['items']
    pbar = tqdm(total=len(items))
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
        bboxes, contours, cropped_imgs = get_bbox(img, mask_color)
        # half
        mask[:h//2,:] = 0
        mask_color[:h//2,:] = 0
        half_bboxes, half_contours, half_cropped_imgs = get_bbox(img, mask_color)
        contours += half_contours
        cropped_imgs += half_cropped_imgs
        if bboxes:
            bbox_info = {'file_name':file_name, 'bbox':half_bboxes}
            with open(os.path.join(bbox_path, f'{only_name}.json'), 'w') as f:
                json.dump(bbox_info, f)
            con_names = save_line(file_name, contours, cropped_imgs)
            cv2.imwrite(os.path.join(mask_path,file_name), mask)
            cv2.imwrite(os.path.join(mask_color_path,file_name), mask_color)
        pbar.update(1)
    pbar.close()