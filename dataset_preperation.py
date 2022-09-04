import os
import cv2
import json
import shutil

from tqdm import tqdm
from config import csp_cfg
from Primary.data_generator.cityperson import CityPersonDataset
from Primary.data_generator.datagen import create_visible_bbox


def Copy_and_Save (img_path,set_file):

    if isinstance(img_path,str) and isinstance(set_file,str):
        print("Images -")
        subfiles=os.listdir(img_path)
        for subf in subfiles:
            set_path = os.path.join(set_file,subf)
            if os.path.exists(set_path) != True:
                os.makedirs(set_path)
            print(f"\n{subf} - ")
            original_dir=os.path.join(img_path,subf)
            city_files = os.listdir(original_dir)
            for city in city_files:
                img_dir = os.path.join(original_dir, city)
                img_names = os.listdir(img_dir)
                print(f"\n{city} :" )
                for i in tqdm(range(len(img_names))):
                    image = cv2.imread(os.path.join(img_dir, img_names[i]))
                    cv2.imwrite(os.path.join(set_path,img_names[i]), image)
    else:
        raise ValueError("img_path and set_file must be a string !")



def Move_Json_Files(json_path,set_file):

    if isinstance(json_path, str) and isinstance(set_file, str):
        print(" Annotations : \n")
        subfiles = os.listdir(json_path)
        for subf in subfiles:
            set_path = os.path.join(set_file, subf)
            if os.path.exists(set_path) != True:
                os.makedirs(set_path)
            print(f"\n{subf} - ")
            original_dir = os.path.join(json_path, subf)
            city_files = os.listdir(original_dir)
            for city in city_files:
                json_dir = os.path.join(original_dir, city)
                json_names = os.listdir(json_dir)
                print(f"\n{city} :")
                for i in tqdm(range(len(json_names))):
                    json_file = os.path.join(json_dir, json_names[i])
                    shutil.copy2(src=json_file, dst=set_path)
    else:
        raise ValueError("img_path and set_file must be a string !")

def delete_invalid_images_labels(Img_Path,Label_Path,subset="None"):
    Img_list = os.listdir(Img_Path)
    Label_list = os.listdir(Label_Path)
    for i in range(len(Label_list)):

        if Img_list[i].split('_')[:-1] == Label_list[i].split('_')[:-1]:
            f = open(os.path.join(Label_Path, Label_list[i]))
            data = json.load(f)
            data = CityPersonDataset(data=data)
            boxes = create_visible_bbox(data,subset=subset)
            if data.get_num_objects() == 0 or len(boxes) == 0:
                f.close()
                print("{} and {} are deleted!".format(Img_list[i], Label_list[i]))
                os.remove(os.path.join(Img_Path, Img_list[i]))
                os.remove(os.path.join(Label_Path, Label_list[i]))
        else:
            raise Exception('Image and Annotation file are not the same!')

if __name__ == '__main__':

    IMG_PATH = csp_cfg['IMG_PATH']
    JSON_PATH = csp_cfg['JSON_ANNOTATION']
    MAIN_DIR = csp_cfg['MAIN_DIR']
    SET_IMG_PATH = csp_cfg['SET_IMG_PATH']
    SET_JSON_PATH = csp_cfg['SET_JSON_PATH']
    SUBSET = csp_cfg["SUBSET"]

    SET_IMG_PATH = os.path.join(MAIN_DIR,SET_IMG_PATH)
    SET_JSON_PATH = os.path.join(MAIN_DIR, SET_JSON_PATH)
    Copy_and_Save(img_path=IMG_PATH,set_file=SET_IMG_PATH)
    Move_Json_Files(json_path=JSON_PATH, set_file=SET_JSON_PATH)
    delete_invalid_images_labels(Img_Path=os.path.join(SET_IMG_PATH,"train"),Label_Path=os.path.join(SET_JSON_PATH,"train"),subset=SUBSET)
    delete_invalid_images_labels(Img_Path=os.path.join(SET_IMG_PATH, "val"),Label_Path=os.path.join(SET_JSON_PATH, "val"),subset=SUBSET)