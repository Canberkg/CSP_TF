import os
import json
import cv2

import numpy as np
import tensorflow as tf


from Primary.data_generator.data_augmentation import image_augmentation
from Primary.data_generator.cityperson import CityPersonDataset

def resize_boxes(resized_w,resized_h,gt_boxes):

    ratio_h = resized_h/1024
    ratio_w = resized_w/2048
    gt_boxes[:, 0:4:2] *= ratio_w
    gt_boxes[:, 1:4:2] *= ratio_h

    return gt_boxes

def create_visible_bbox(data,subset="default"):
    visbboxes=[]

    for i in range(data.get_num_objects()):

        x_min=data.get_bounding_box(i)[0]
        y_min=data.get_bounding_box(i)[1]
        w=data.get_bounding_box(i)[2]
        h=data.get_bounding_box(i)[3]
        class_label=data.get_label(idx=i)
        h_unresized = h

        w_visible = data.get_visibile_box(i)[2]
        h_visible = data.get_visibile_box(i)[3]

        occlusion_rate = 1-((w_visible * h_visible) / (w * h))

        if subset == "reasonable":
            if class_label == "pedestrian" and h_unresized >= 50 and occlusion_rate <= 0.35:
                x_max = x_min + w
                y_max = y_min + h
                visbboxes.append([x_min,y_min,x_max,y_max])
        elif subset == "small":
            if class_label == "pedestrian" and h_unresized >= 50 and h_unresized <= 50 and occlusion_rate <= 0.35:
                x_max = x_min + w
                y_max = y_min + h
                visbboxes.append([x_min, y_min, x_max, y_max])
        elif subset == "highly_occluded":
            if class_label == "pedestrian" and h_unresized >= 50 and occlusion_rate >= 0.35 and occlusion_rate <= 0.8:
                x_max = x_min + w
                y_max = y_min + h
                visbboxes.append([x_min, y_min, x_max, y_max])
        elif subset == "default":
            if class_label == "pedestrian":
                x_max = x_min + w
                y_max = y_min + h
                visbboxes.append([x_min, y_min, x_max, y_max])

    if len(visbboxes) == 0:
        return visbboxes
    else:
        return np.stack(visbboxes,axis=0)

class data_gen(tf.keras.Sequential):
    """Data Generator Class

    Attributes :
        Img_Path      : Path of Image Directory
        Label_Path    : Path of Label Directory
        Img_Width     : Width of input accepted by the network (Int)
        Img_Height    : Height of input accepted by the network (Int)
        Batch_Size    : Length of a  Batch
        Num_Classes   : Number of Classes
        Subset        : Subset of CityPersons (Reasonable : "reasonable", Small : "small", Highly occluded : "highly_occluded", No Subset : "None")
        Shuffle       : Whether Shuffle the data or not (boolean) , False by default
        Augmentation  : Perform Augmentation or Not (boolean), False by default

    """
    def __init__(self,Img_Path,Label_Path,Img_Width,Img_Height,Batch_Size,Num_Classes,Subset="None",Augmentation=False,Shuffle=False):
        super(data_gen, self).__init__()
        self.Img_Path = Img_Path
        self.Label_Path = Label_Path
        self.Img_list = os.listdir(self.Img_Path)
        self.Label_list = os.listdir(self.Label_Path)
        self.Img_Width = Img_Width
        self.Img_Height = Img_Height
        self.Batch_Size = Batch_Size
        self.Num_Classes = Num_Classes
        self.indices = range(0, len(self.Img_list) - (len(self.Img_list) % self.Batch_Size))
        self.index = np.arange(0, len(self.Img_list) - (len(self.Img_list) % self.Batch_Size))
        self.subset=Subset
        self.Augmentation = Augmentation
        self.shuffle = Shuffle
        self.radius=2

    def num_images(self):
        """Find the total number of image
        Params:
        Return:
           Total number of image (int)
        """
        return len(self.Img_list) - (len(self.Img_list) % self.Batch_Size)

    def on_epoch_end(self):
        """Apply after completion of each epoch
        Params:
        Return:
            Shuffled indices if Shuufle is True (int)
        """
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        """Find the original width of image
        Params:
        Return:
            Total number of batches (int)
        """
        return (len(self.Img_list) // self.Batch_Size)

    def gaussian(self, kernel):
        sigma = ((kernel - 1) * 0.3 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    def get_label(self,gt_boxes,img_shape):


        center_map = np.zeros(shape=(int(img_shape[0] / 4), int(img_shape[1] / 4),3))
        scale_map  = np.zeros(shape=(int(img_shape[0] / 4), int(img_shape[1] / 4),3))
        offset_map = np.zeros(shape=(int(img_shape[0] / 4), int(img_shape[1] / 4),3))
        center_map[:,:,1]=1

        if len(gt_boxes)!=0:

            for box in gt_boxes:
                x1,y1,x2,y2 = box
                x1, y1, x2, y2=int(x1/4),int(y1/4),int(x2/4),int(y2/4)
                cx,cy = int((x2+x1)/2),int((y2+y1)/2)
                dx,dy = self.gaussian(kernel=(x2-x1)),self.gaussian(kernel=(y2-y1))
                gaussian_map = np.multiply(dy,np.transpose(dx))

                center_map[y1:y2, x1:x2,0] = np.maximum(center_map[y1:y2, x1:x2,0], gaussian_map)
                center_map[y1:y2, x1:x2,1] = 1
                center_map[cy, cx,2] = 1

                scale_map[cy - self.radius:cy + self.radius + 1, cx - self.radius:cx + self.radius + 1,0] = np.log(
                    (y2 - y1))
                scale_map[cy - self.radius:cy + self.radius + 1, cx - self.radius:cx + self.radius + 1,1] = np.log(
                    (x2 - x1))

                scale_map[cy - self.radius:cy + self.radius + 1, cx - self.radius:cx + self.radius + 1,2] = 1

                offset_map[cy, cx,0] = (y1 + y2) / 2 - cy - 0.5
                offset_map[cy, cx,1] = (x1 + x2) / 2 - cx - 0.5
                offset_map[cy, cx,2] = 1

        return center_map,scale_map,offset_map

    def __getitem__(self,index):
        """Get batch of images and labels
        Params:
            index: Index of a batch
        Return:
            x: Batch of images (Tuple)
            y: Batch of labels (List)
        """
        index = self.index[index * self.Batch_Size:(index + 1) * self.Batch_Size]
        batch = [self.indices[i] for i in index]

        img, center,scale,offset = self.__getdata__(batch)

        return img, center,scale,offset

    def __getdata__(self,batch):

        Images = []
        Center = []
        Scale = []
        Offset = []

        if self.Augmentation:
            augment = image_augmentation(flip='horizontal')
            batch_of_images = [self.Img_list[i] for i in batch]

            for k in range(len(batch)):
                label_arr = batch_of_images[k].split('_')
                label_arr[-1] = 'gtBboxCityPersons.json'
                label = '_'.join(label_arr)
                if label in self.Label_list:
                    Img = cv2.imread(filename=os.path.join(self.Img_Path, batch_of_images[k]))
                    Img = cv2.cvtColor(Img,cv2.COLOR_BGR2RGB)
                    Img = augment.aug_brightness(Img,0.5,2,prob=0.5)
                    Img_arr = np.asarray(Img, dtype=np.float32)

                    label_ind = self.Label_list.index(label)
                    f = open(os.path.join(self.Label_Path, self.Label_list[label_ind]))
                    data = json.load(f)
                    data = CityPersonDataset(data=data)
                    gt_boxes = create_visible_bbox(data=data,subset=self.subset)


                    cropped_image,gt_boxes = augment.aug_random_crop(img=Img_arr,labels=gt_boxes,crop_h=self.Img_Height,crop_w=self.Img_Width)
                    cropped_image,gt_boxes = augment.aug_flip(img=cropped_image,labels=gt_boxes,prob=0.5)
                    Img_arr = tf.keras.applications.resnet50.preprocess_input(cropped_image)
                    Images.append(Img_arr)


                    center_map, scale_map, offset_map = self.get_label(gt_boxes=gt_boxes, img_shape=Img_arr.shape)
                    Center.append(center_map)
                    Scale.append(scale_map)
                    Offset.append(offset_map)
                else:
                    raise Exception('Image and Label does not belong to the same object!')
        else:

            batch_of_images = [self.Img_list[i] for i in batch]
            for k in range(len(batch)):
                label_arr = batch_of_images[k].split('_')
                label_arr[-1] = 'gtBboxCityPersons.json'
                label = '_'.join(label_arr)
                if label in self.Label_list:
                    Img = cv2.imread(filename=os.path.join(self.Img_Path, batch_of_images[k]))
                    Img = cv2.resize(Img, (self.Img_Width, self.Img_Height), interpolation=cv2.INTER_NEAREST)
                    Img = cv2.cvtColor(Img,cv2.COLOR_BGR2RGB)
                    Img_arr = np.asarray(Img, dtype=np.float32)
                    Img_arr = tf.keras.applications.resnet50.preprocess_input(Img_arr)
                    Images.append(Img_arr)

                    label_ind = self.Label_list.index(label)
                    f = open(os.path.join(self.Label_Path, self.Label_list[label_ind]))
                    data = json.load(f)
                    data = CityPersonDataset(data=data)
                    gt_boxes = create_visible_bbox(data=data,subset=self.subset)
                    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                    gt_boxes = resize_boxes(resized_w=self.Img_Width,resized_h=self.Img_Height,gt_boxes=gt_boxes)

                    center_map, scale_map, offset_map = self.get_label(gt_boxes=gt_boxes, img_shape=Img_arr.shape)
                    Center.append(center_map)
                    Scale.append(scale_map)
                    Offset.append(offset_map)
                else:
                    raise Exception('Image and Label does not belong to the same object!')


        return tf.stack(Images, axis=0),tf.cast(tf.stack(Center, axis=0),dtype=tf.float32),tf.cast(tf.stack(Scale, axis=0),dtype=tf.float32),tf.cast(tf.stack(Offset, axis=0),dtype=tf.float32)