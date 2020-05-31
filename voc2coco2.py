import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
import random
import shutil
def get_ann_paths(ann_dir_path, pic_dir_path, save_train_pic_path, save_val_pic_path, trainval_txt):
    # ann_path = {}
    ann_ids = []
    ann_paths_train = []
    ann_paths_val = []
    with open(trainval_txt, 'r') as f:
        for trainval_line in f.readlines():
            # trainval_data = trainval_line.strip('\n') + '.xml'
            trainval_data = trainval_line.strip('\n')
            ann_ids.append(trainval_data)
    random.seed(1)
    train_ids = random.sample(ann_ids, int(0.8*len(ann_ids)))
    val_ids = list(set(ann_ids) - set(train_ids))
    print('Start move train figure!')
    for train_aid in  tqdm(train_ids):
        old_train_pic_path = os.path.join(pic_dir_path, train_aid + '.png')
        new_train_pic_path = os.path.join(save_train_pic_path, train_aid + '.png')
        shutil.copyfile(old_train_pic_path, new_train_pic_path)
        ann_paths_train.append(os.path.join(ann_dir_path, train_aid + '.xml'))
    print('Start move val figure!')
    for val_aid in tqdm(val_ids):
        old_val_pic_path = os.path.join(pic_dir_path, val_aid + '.png')
        new_val_pic_path = os.path.join(save_val_pic_path, val_aid + '.png')
        shutil.copyfile(old_val_pic_path, new_val_pic_path)
        ann_paths_val.append(os.path.join(ann_dir_path, val_aid + '.xml'))
    # pic_paths_train = [os.path.join(pic_dir_path, aid + '.png') for aid in train_ids]
    # pic_paths_val = [os.path.join(pic_dir_path, aid + '.png') for aid in val_ids]
    # ann_paths_train = [os.path.join(ann_dir_path, aid + '.xml') for aid in train_ids]
    # ann_paths_val = [os.path.join(ann_dir_path, aid + '.xml') for aid in val_ids]
    # ann_paths_train = random.sample(ann_paths, int(0.8*len(ann_paths)))
    # ann_paths_val = list(set(ann_paths) - set(ann_paths_train))
    # print(len(ann_paths_train), len(ann_paths_val))
    # if not return_train: return ann_paths_val
    ann_path = {"ann_paths_train":ann_paths_train, "ann_paths_val":ann_paths_val}
    return ann_path

def get_image_info(annotation_root, ann_img_id):
    filename = annotation_root.findtext('filename')
    img_name = os.path.basename(filename)
    # img_id = os.path.splitext(img_name)[0].split('_')[1]
    img_id = ann_img_id
    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

def get_coco_annotation_from_obj(label_info, obj):
    label = obj.findtext('name')
    label = label.capitalize()
    assert label in label_info, f"Error: {label} is not in label_info !"
    category_id = label_info[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin'))
    ymin = int(bndbox.findtext('ymin')) 
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  
    }
    return ann

def convert_xmls_to_cocojson(ann_paths, output_json_path, label_info):
    output_json_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    bnd_id = 0  # start_bbx_id
    img_id = 0
    print('Start converting  json!')
    for ann_file_path in tqdm(ann_paths):
        ann_tree = ET.parse(ann_file_path)
        ann_root = ann_tree.getroot()
        img_id += 1
        img_info = get_image_info(annotation_root=ann_root, ann_img_id = img_id)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(label_info, obj=obj)
            bnd_id = bnd_id + 1
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            

    for label, label_id in label_info.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_json_path, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


if __name__ == "__main__":
    # annotation_paths= '/home/xjma/Downloads/MOCOD/FreeAnchor/tools/car_0001.xml'  
    OUTPUT_TRAIN_JSON_PATH = '/home/xjma/Downloads/MOCOD/coco/annotations/train2020.json'    # save train.json 
    OUTPUT_VAL_JSON_PATH = '/home/xjma/Downloads/MOCOD/coco/annotations/val2020.json'     # save val.json
    OUTPUT_TRAIN_PIC_PATH = '/home/xjma/Downloads/MOCOD/coco/train2020'
    OUTPUT_VAL_PIC_PATH = '/home/xjma/Downloads/MOCOD/coco/val2020'

    ANNOTATION_PATH = '/home/xjma/Downloads/MOCOD/Annotations'  # change path to Annotations dir 
    TRAINVAL = '/home/xjma/Downloads/MOCOD/Main/trainval.txt'  # change path to trainval.txt
    PIC_PATH = '/home/xjma/Downloads/MOCOD/JPEGImages'
    # ann_paths_train = get_ann_paths(ANNOTATION_PATH, PIC_PATH, OUTPUT_TRAIN_PIC_PATH, OUTPUT_VAL_PIC_PATH,TRAINVAL, return_train=True)
    # ann_paths_test = get_ann_paths(ANNOTATION_PATH,PIC_PATH, OUTPUT_TRAIN_PIC_PATH, OUTPUT_VAL_PIC_PATH, TRAINVAL, return_train=False)
    ann_paths = get_ann_paths(ANNOTATION_PATH,PIC_PATH, OUTPUT_TRAIN_PIC_PATH, OUTPUT_VAL_PIC_PATH, TRAINVAL)
    ann_paths_train = ann_paths["ann_paths_train"]
    ann_paths_val = ann_paths["ann_paths_val"]
    convert_xmls_to_cocojson(ann_paths_train, output_json_path=OUTPUT_TRAIN_JSON_PATH, label_info={"Car":1,"Human":2,"Ship":3,"Plane":4})
    convert_xmls_to_cocojson(ann_paths_val, output_json_path=OUTPUT_VAL_JSON_PATH, label_info={"Car":1,"Human":2,"Ship":3,"Plane":4})
   

