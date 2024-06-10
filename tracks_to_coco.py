import os
import json
import random
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from tracks_mask_bbox import split_file_name, add_word_to_file_name

DATA_PATH = 'data/merged_data'

# id, image_id, category_id, segmentation, area(seg), bbox
def get_annotations(bbox_path, image_id, id):
    annotation = []
    with open(bbox_path) as f:
        bbox_json = json.load(f)
    for bbox in bbox_json['bbox']:
        tmp_dict = dict()
        tmp_dict['id'] = id
        tmp_dict['image_id'] = image_id
        tmp_dict['category_id'] = 0 # maybe need change
        x,y,w,h = bbox
        coco_bbox = [x, y, w, h]
        tmp_dict['bbox'] = coco_bbox
        tmp_dict['iscrowd'] = 0
        tmp_dict['area'] = w*h
        annotation.append(tmp_dict)
        id += 1
    return annotation, id

def split_train_val(json_data, val_size=0.25):
    train_data,train_list = dict(), []
    val_data,val_list = dict(), []
    train_list, val_list = train_test_split(json_data['items'], test_size=val_size, random_state=42, shuffle=True)

    train_data['items'] = train_list
    val_data['items'] = val_list
    return train_data, val_data

# id, width, height, file_name
def get_json(json_data):
    images = []
    annotations = []
    annot_id = 0
    pbar = tqdm(total=len(json_data['items']))
    for i,item in enumerate(json_data['items']):
        annot = item['annotations']
        pbar.update(1)
        if not annot:
            continue
        file_name = item['image']['file_name']
        bbox_file_name = split_file_name(file_name, True)
        bbox_path = os.path.join(DATA_PATH, 'bbox', f'{bbox_file_name}.json')
        width, height = item['image']['width'], item['image']['height']
        tmp_dict = dict()
        tmp_dict['id'] = i
        tmp_dict['file_name'] = file_name
        tmp_dict['width'] = width
        tmp_dict['height'] = height//2
        images.append(tmp_dict)
        tmp_annotation, annot_id = get_annotations(bbox_path, i, annot_id)
        annotations.extend(tmp_annotation)
    pbar.close()
    coco_json = dict()
    coco_json['images'] = images
    coco_json['annotations'] = annotations
    coco_json['categories'] = [{"id":0, "name":"line"}] # maybe need change
    return coco_json



if __name__ == "__main__":
    os.makedirs(os.path.join(DATA_PATH, "coco/annotations"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "coco/images/train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "coco/images/val"), exist_ok=True)

    json_path = f'{DATA_PATH}/data.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    train_data, val_data = split_train_val(json_data)

    # convert to coco
    train_coco_json = get_json(train_data)
    val_coco_json = get_json(val_data)

    for images in train_coco_json['images']:
        file_name = images['file_name']
        src_file_path = os.path.join(DATA_PATH, 'half_images',file_name)
        dst_file_path = os.path.join(DATA_PATH, 'coco/images', 'train', file_name)
        shutil.copy(src_file_path, dst_file_path)

    for images in val_coco_json['images']:
        file_name = images['file_name']
        src_file_path = os.path.join(DATA_PATH, 'half_images',file_name)
        dst_file_path = os.path.join(DATA_PATH, 'coco/images', 'val', file_name)
        shutil.copy(src_file_path, dst_file_path)

    with open(f'{DATA_PATH}/coco/annotations/train.json', 'w') as f:
        json.dump(train_coco_json, f, indent=2)
    with open(f'{DATA_PATH}/coco/annotations/val.json', 'w') as f:
        json.dump(val_coco_json, f, indent=2)