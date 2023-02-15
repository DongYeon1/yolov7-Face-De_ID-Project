# pip install -r yolov7/requirements.txt
# pip install -r yolov7/requirements_gpu.txt

# python train.py --workers 1 --device 0 --batch-size 16 --epochs 100 --img 640 640 --hyp data/hyp.scratch.custom.yaml --name yolov7-custom --weights yolov7.pt


import json
from pathlib import Path
from PIL import Image
import pandas
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

def bbox_to_yolo(image_size, bbox):
    dw = 1. / image_size[0]
    dh = 1. / image_size[1]
    x = (bbox['x'] + bbox['x'] + bbox['width']) / 2.0
    y = (bbox['y'] + bbox['y'] + bbox['height']) / 2.0
    w = bbox['width']
    h = bbox['height']
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    ret_string = "0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
    return ret_string


def writing_txt(coco_paths, img_paths, label_paths, train_or_test, jpg_lists, json_lists):
    g = open(str(coco_paths) + "/" + train_or_test + ".txt", 'w')
    for jpg_name, label_name in tqdm(zip(jpg_lists, json_lists)):
        if jpg_name[0:-4] != label_name[0:-5]:
            raise Exception('Does not match pairing -> ' + jpg_name + "!=" + label_name)
        g.write(str(img_paths) + "/" + jpg_name + "\n")  # 전체 txt 리스트에 파일명 적기
        image_size = Image.open(str(img_paths) + "/" + jpg_name).size
        json_file = json.load(Path(str(label_paths) + '/' + label_name).open(mode='r'))
        try:
            img_info_dict = json_file['annotationsData']['image']
        except TypeError as e:
            print(e)
            jpg_lists.remove(jpg_name)
            json_lists.remove(label_name)
            print(jpg_name + ' and ' + label_name + ' are deleted from jpg_list and json_list')
            continue
        f = open(str(coco_paths) + "/labels/" + train_or_test + "/" + jpg_name[0:-4] + ".txt", 'w')
        for image_info in img_info_dict:
            bbox = image_info['coordinate']
            f.write(bbox_to_yolo(image_size, bbox))
        f.close()
    g.close()

    return 0


def create_json_to_txt(coco_paths, img_paths, label_paths):
    jpg_list = os.listdir(img_paths)
    jpg_list.remove('@eaDir')
    json_list = os.listdir(label_paths)
    jpg_list.sort()
    json_list.sort()
    small_jpg_list = jpg_list[0:30000]
    small_json_list = json_list[0:30000]

    #merged_list = list(zip(jpg_list, json_list))

    jpg_train, jpg_valid, json_train, json_valid = train_test_split(small_jpg_list, small_json_list, test_size=0.2,
                                                                    shuffle=False)

    train_or_test = "train"
    writing_txt(coco_paths, img_paths, label_paths, train_or_test, jpg_train, json_train)
    train_or_test = "val"
    writing_txt(coco_paths, img_paths, label_paths, train_or_test, jpg_valid, json_valid)

    return 0

Image_Path = Path('/mnt/development/ML_team/facial_img_json/img')
Label_Path = Path('/mnt/development/ML_team/facial_img_json/json')
coco_Path = Path('/root/home/development/users/dongyeon/datamaker/final_yolov7/coco')

create_json_to_txt(coco_Path, Image_Path, Label_Path)

# path_to_txt = Path('/home/development/users/dongyeon/datamaker/yolov7/coco/train.txt')

# jpg_list = os.listdir(Train_Image_Path)
# json_list = os.listdir(Train_Label_Path)

# with open('/home/development/users/dongyeon/datamaker/yolov7/coco/train.txt', 'r') as r:
#     for line in sorted(r):
#         print(line, end='')

# image_object = Image.open('/home/development/users/dongyeon/datamaker/yolov7/coco/images/train/airport_inside_airport_inside_0007.jpg')
# image_size = image_object.size

# list = json.load(Path(str(Train_Label_Path)+'/'+json_list[0]).open(mode='r'))
# for bbox in list:
# print(bbox_to_yolo(image_size,bbox))

# ret = "0 0.3 0.3 0.3"


# empty 파일 갯수 세는 코드
# train_Label_Path = Path('/home/development/users/dongyeon/datamaker/yolov7/coco/labels/val_json')
# json_list = os.listdir(train_Label_Path)
# cnt = 0
# for file_name in json_list:
#     bbox_list = json.load(Path(str(train_Label_Path)+'/'+file_name).open(mode='r'))
#     if len(bbox_list) == 0:
#        cnt += 1
# print(cnt)
