import os
import json
from PIL import Image

data_path = 'data/TUSimple/MyTuSimpleLane'
os.makedirs(f"{data_path}/coco/annotations", exist_ok=True)
train_path = os.path.join(data_path,'train')
test_path = os.path.join(data_path,'test')

# id, image_id, category_id, segmentation, area(seg), bbox
def get_annotations(bbox_path, file_name, image_id, id):
    annotation = []
    file_name = file_name.split('.')[0] + '.json'
    file_path = os.path.join(bbox_path, file_name)
    with open(file_path) as f:
        bbox_json = json.load(f)
    for jd in bbox_json:
        tmp_dict = dict()
        tmp_dict['id'] = id
        tmp_dict['image_id'] = image_id
        tmp_dict['category_id'] = jd['class']
        x1,y1,x2,y2 = jd['points']
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]
        tmp_dict['bbox'] = coco_bbox
        tmp_dict['is_crowd'] = 0
        tmp_dict['area'] = 0
        annotation.append(tmp_dict)
        id += 1
    return annotation, id

# id, width, height, file_name
def get_json(data_path):
    img_path = os.path.join(data_path,'img')
    bbox_path = os.path.join(data_path,'bbox')
    images = []
    annotations = []
    img_names = os.listdir(img_path)
    annot_id = 0
    for i,n in enumerate(img_names):
        tmp_dict = dict()
        tmp_dict['id'] = i
        file_path = os.path.join(img_path, n)
        tmp_dict['file_name'] = n
        img = Image.open(file_path)
        width, height = img.size
        tmp_dict['width'] = width
        tmp_dict['height'] = height
        images.append(tmp_dict)
        tmp_annotation, annot_id = get_annotations(bbox_path, n, i, annot_id)
        annotations.extend(tmp_annotation)
    coco_json = dict()
    coco_json['images'] = images
    coco_json['annotations'] = annotations
    coco_json['categories'] = [{"id":0, "name":"right down", "supercategory":"Lane"},{"id":1, "name":"right up", "supercategory":"Lane"}]
    return coco_json

train_coco_json = get_json(train_path)
test_coco_json = get_json(test_path)

with open(f'{data_path}/coco/annotations/TuSimple_train.json', 'w') as f:
    json.dump(train_coco_json, f, indent=2)

with open(f'{data_path}/coco/annotations/TuSimple_test.json', 'w') as f:
    json.dump(test_coco_json, f, indent=2)