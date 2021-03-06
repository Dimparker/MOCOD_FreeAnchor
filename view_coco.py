from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import os
from skimage.io import imsave
import numpy as np

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
img_path = '/home/xjma/Downloads/MOCOD/coco/train2020'
annFile = '/home/xjma/Downloads/MOCOD/coco/annotations/train2020.json'

if not os.path.exists('anno_image_coco/'):
    os.makedirs('anno_image_coco/')

def draw_rectangle(coordinates, image, image_name):
    for coordinate in coordinates:
        left = np.rint(coordinate[0])
        right = np.rint(coordinate[1])
        top = np.rint(coordinate[2])
        bottom = np.rint(coordinate[3])
        # 左上角坐标, 右下角坐标 
        cv2.rectangle(image,
                      (int(left), int(right)),
                      (int(top), int(bottom)),
                      (0, 255, 0),
                      2)
    imsave('anno_image_coco/'+image_name, image)


coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: {}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
img_path = '/home/xjma/Downloads/MOCOD/coco/train2020'
img_list = os.listdir(img_path)
for i in range(20):
    imgIds = i+1
    img = coco.loadImgs(imgIds)[0]
    image_name = img['file_name']
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIds)
    # print(anns)
    coordinates = []
    img_raw = cv2.imread(os.path.join(img_path, image_name))
    for j in range(len(anns)):
        coordinate = []
        coordinate.append(anns[j]['bbox'][0])
        coordinate.append(anns[j]['bbox'][1]+anns[j]['bbox'][3])
        coordinate.append(anns[j]['bbox'][0]+anns[j]['bbox'][2])
        coordinate.append(anns[j]['bbox'][1])
        coordinates.append(coordinate)
    draw_rectangle(coordinates, img_raw, image_name)