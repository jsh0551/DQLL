import numpy as np
import os
import cv2
import json
from tqdm import tqdm
from PIL import Image
from os.path import join as pathJoin
from os import listdir
from scipy.spatial import cKDTree

def get_image_list(data_dir):
    img_paths = listdir(data_dir)
    img_paths = [pathJoin(data_dir, path) for path in img_paths]
    base_list, rgb_list, bin_list, seg_list = [], [], [], []
    for path in img_paths:
        if ('bin768.png' in path):
            base_path = "_".join(path.split('_')[:-1])
            base_list.append(base_path)
            rgb_list.append(base_path+'.png')
            bin_list.append(base_path+'_bin768.png')
            seg_list.append(base_path+'_seg.png')
    return base_list, rgb_list, bin_list, seg_list

# 이미지 생성
def interpolate_seg(seg_mask):
    filled_idx = np.where(seg_mask > 0)
    points = list(zip(filled_idx[1], filled_idx[0]))
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=3)
    for i, j in pairs:
        start_point = tuple(points[i])
        end_point = tuple(points[j])
        cv2.line(seg_mask, start_point, end_point, 1, 2)
    return seg_mask

def post_process_mask(bin_masks, base_list, root_path, blank_ratio = 0.65, min_h=20, min_w=20):
    pp_masks, color_pp_masks, bboxes, box_masks, classes = [], [], [], [], []
    pbar = tqdm(total=len(base_list))
    for mask, base in zip(bin_masks, base_list):
        base_name = os.path.basename(base)
        height, _ = mask.shape
        blank = int(height * blank_ratio)
        mask[:blank, : ] = 0
        cons, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        sort_cons, bbox, box_mask, cls = [], [], [], []
        # get box and box index 
        for con in cons:
            x, y, w, h = cv2.boundingRect(con)
            if h<=min_h or w<=min_w:
                continue
            x1, y1, x2, y2 = x, y, x+w, y+h
            sort_cons.append(con)
            bbox.append((x1,y1,x2,y2))
        pp_mask = np.zeros_like(mask)
        # draw contour and get class
        for con, b in zip(sort_cons, bbox):
            # box mask
            x1,y1,x2,y2 = b
            tmp_mask = np.zeros_like(mask)
            cv2.drawContours(tmp_mask, [con], 0, (1), -1)
            bmsk = tmp_mask[y1:y2, x1:x2]
            # image mask
            cv2.drawContours(pp_mask, [con], 0, (1), -1)
            [vx, vy, xx, yy] = cv2.fitLine(con, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = -float(vy)/float(vx)
            if slope <= 0:
                cl = 0
            else:
                cl = 1
            cls.append(cl)
            box_mask.append(bmsk)
        color_pp_mask = pp_mask * 255
        cv2.imwrite(pathJoin(root_path, 'mask',base_name+'.png'), pp_mask)
        cv2.imwrite(pathJoin(root_path, 'mask_color', base_name+'.png'), color_pp_mask)
        pp_masks.append(pp_mask)
        color_pp_masks.append(color_pp_mask)
        bboxes.append(bbox)
        box_masks.append(box_mask)
        classes.append(cls)
        pbar.update(1)
    pbar.close()
    return pp_masks, color_pp_masks, bboxes, box_masks, classes

def get_mask_box_cls(base_list, root_path):
    bin_masks = []
    print("...preparing mask image")
    for base in base_list:
        seg_img = cv2.imread(base+'_seg.png')
        seg_mask = np.zeros_like(seg_img[:, :, 0])
        seg_mask[(seg_img[:,:,2]==157)&(seg_img[:,:,1]==234)&(seg_img[:,:,0]==50)] = 1
        seg_mask = interpolate_seg(seg_mask)

        tmp_bin = cv2.imread(base+'_bin768.png')
        tmp_bin = cv2.resize(tmp_bin, (1280,720))
        tmp_bin = cv2.cvtColor(tmp_bin, cv2.COLOR_BGR2GRAY)

        bin_mask = np.zeros_like(tmp_bin)
        bin_mask[tmp_bin>130] = 1
        bin_mask[seg_mask > 0] = 1
        bin_masks.append(bin_mask)
    print("...post processing mask image")
    pp_masks, color_pp_masks, bboxes, box_masks, classes = post_process_mask(bin_masks, base_list, root_path)


    return pp_masks, color_pp_masks, bboxes, box_masks, classes

# DRL : contour image, mask, mask_color, json(class, gt)로 저장. ori 생성 후 resize 생성
def get_gt_regt(box_mask, rebox_mask):
    initY = []
    h = box_mask.shape[0]
    for i in range(5):
        initY.append(int((i+1)*(h/6)))
    gt = []
    for y in initY:
        xx = box_mask[y, :]
        xx = np.where(xx == 1)
        x = int((np.max(xx)+np.min(xx))/2)
        gt.append(x)

    initY = [11, 31, 51, 71, 91]
    regt = []
    for y in initY:
        xx = rebox_mask[y, :]
        xx = np.where(xx == 1)
        x = int((np.max(xx)+np.min(xx))/2)
        regt.append(x)
    return gt, regt

def get_bbox_DRL(root_path, base_list, masks, color_masks, bboxes, box_masks, classes):
    box_idx = 0
    print("...get bbox and DRL")
    pbar = tqdm(total=1)
    for idx ,(box, box_mask, cls) in enumerate(zip(bboxes, box_masks, classes)):
        file_name = os.path.basename(base_list[idx])
        bgr_img = cv2.imread(base_list[idx] + '.png')
        cv2.imwrite(pathJoin(root_path, 'img', f'{file_name}.png'), bgr_img)
        
        bbox_list = []
        for b, bmsk, cl in zip(box, box_mask, cls):
            x1, y1, x2, y2 = b
            box_img = bgr_img[y1:y2, x1:x2, :]
            rebox_img = cv2.resize(box_img, (100,100))
            re_bmsk = cv2.resize(bmsk, (100,100))
            bcmsk = bmsk * 255
            re_bcmsk = re_bmsk * 255
            gt, regt = get_gt_regt(bmsk, re_bmsk)
            bbox = {"points":[x1, y1, x2, y2], "class":cl}
            gt_data = {"class":cl, "gt":gt}
            regt_data = {"class":cl, "gt":regt}

            box_img_name, gt_json_name = f'{file_name}_{box_idx}.png', f'{file_name}_{box_idx}.json'
            box_mask_name, box_cmask_name = f'{file_name}_{box_idx}_mask.png', f'{file_name}_{box_idx}_mask_color.png'
            # save file
            cv2.imwrite(pathJoin(root_path, 'DRL/ori', box_img_name), box_img)
            cv2.imwrite(pathJoin(root_path, 'DRL/ori', box_mask_name), bmsk)
            cv2.imwrite(pathJoin(root_path, 'DRL/ori', box_cmask_name), bcmsk)
            with open(pathJoin(root_path, 'DRL/ori', gt_json_name), 'w') as f:
                json.dump(gt_data, f)
            cv2.imwrite(pathJoin(root_path, 'DRL/resize', box_img_name), rebox_img)
            cv2.imwrite(pathJoin(root_path, 'DRL/resize', box_mask_name), re_bmsk)
            cv2.imwrite(pathJoin(root_path, 'DRL/resize', box_cmask_name), re_bcmsk)
            with open(pathJoin(root_path, 'DRL/resize', gt_json_name), 'w') as f:
                json.dump(regt_data, f)
            box_idx += 1
            bbox_list.append(bbox)
        with open(pathJoin(root_path, 'bbox', f'{file_name}.json'), 'w') as f:
            json.dump(bbox_list, f)
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    base_list, rgb_list, bin_list, seg_list = [], [], [], []
    out_dirs = [pathJoin(os.getcwd(), dir_name) for dir_name in os.listdir('.') if '_out' in dir_name]
    image_dirs = []
    for out_dir in out_dirs:
        tmp_dirs = os.listdir(out_dir)
        tmp_dirs = [pathJoin(out_dir, tmp) for tmp in tmp_dirs]
        image_dirs += tmp_dirs
    for img_dir in image_dirs:
        tmp_base, tmp_rgb, tmp_bin, tmp_seg = get_image_list(img_dir)
        base_list.append(tmp_base)
        rgb_list.append(tmp_rgb)
        bin_list.append(tmp_bin)
        seg_list.append(tmp_seg)
    root_path = 'data/carla'
    os.makedirs(root_path, exist_ok=True)
    for dir_name in ['bbox','DRL','DRL/ori','DRL/resize','img','mask','mask_color']:
        os.makedirs(pathJoin(root_path, dir_name), exist_ok=True)
    
    pp_masks, color_pp_masks, bboxes, box_masks, classes = get_mask_box_cls(base_list, root_path)
    print('mask process done!')
    get_bbox_DRL(root_path, base_list, pp_masks, color_pp_masks, bboxes, box_masks, classes)
    print('done!')