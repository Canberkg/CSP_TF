import os
import json
import cv2

import numpy as np
import tensorflow as tf

from Primary.data_generator.class_dict import class_dict
from Primary.data_generator.cityperson import CityPersonDataset

def resize_boxes(image_w,image_h,resized_w,resized_h,x_min,y_min,w_box,h_box):

    ratio_h=resized_h/image_h
    ratio_w=resized_w/image_w
    x_min=x_min*ratio_w
    y_min=y_min*ratio_h
    w_box=w_box*ratio_w
    h_box=h_box*ratio_h

    return x_min,y_min,w_box,h_box

def create_visible_bbox(data,resized_w,resized_h):
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

        occlusion_rate = (w_visible * h_visible) / (w * h)

        x_min,y_min,w,h=resize_boxes(data.get_width(),data.get_height(),resized_w,resized_h,x_min,y_min,w,h)


        if class_label == "pedestrian" and h_unresized >= 50 and occlusion_rate >= 0.65:
            x_max = x_min + w
            y_max = y_min + h

            x_min = max(0.0,x_min)
            y_min = max(0.0, y_min)
            x_max = min(resized_w, x_max)
            y_max = min(resized_h, y_max)

            visbboxes.append([x_min,y_min,x_max,y_max])

    return visbboxes

class data_gen(tf.keras.Sequential):
    """Data Generator Class

    Attributes :
        Img_Path      : Path of Image Directory
        Label_Path    : Path of Label Directory
        Img_Width     : Width of input accepted by the network (Int)
        Img_Height    : Height of input accepted by the network (Int)
        Batch_Size    : Length of a  Batch
        Num_Classes   : Number of Classes
        Shuffle       : Whether Shuffle the data or not (boolean)
        Augmentation  : Augmentation Techniques (Horizontal/Vertical Flip : "FLIP" )

    """
    def __init__(self,Img_Path,Label_Path,Img_Width,Img_Height,Batch_Size,Num_Classes,Augmentation=None,Shuffle=False):
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

        batch_of_images = [self.Img_list[i] for i in batch]

        for k in range(len(batch)):
            label_arr = batch_of_images[k].split('_')
            label_arr[-1] = 'gtBboxCityPersons.json'
            label = '_'.join(label_arr)
            if label in self.Label_list:
                Img = cv2.imread(filename=os.path.join(self.Img_Path, batch_of_images[k]))
                Img = cv2.resize(Img, (self.Img_Width, self.Img_Height), interpolation=cv2.INTER_NEAREST)
                Img_arr = np.asarray(Img, dtype=np.float32)
                Img_arr = tf.keras.applications.resnet50.preprocess_input(Img_arr)
                Images.append(Img_arr)

                label_ind = self.Label_list.index(label)
                f = open(os.path.join(self.Label_Path, self.Label_list[label_ind]))
                data = json.load(f)
                data = CityPersonDataset(data=data)
                gt_boxes = create_visible_bbox(data=data, resized_w=self.Img_Width, resized_h=self.Img_Height)
                center_map, scale_map, offset_map = self.get_label(gt_boxes=gt_boxes, img_shape=Img_arr.shape)
                Center.append(center_map)
                Scale.append(scale_map)
                Offset.append(offset_map)
            else:
                raise Exception('Image and Label does not belong to the same object!')





        return tf.stack(Images, axis=0),tf.cast(tf.stack(Center, axis=0),dtype=tf.float32),tf.cast(tf.stack(Scale, axis=0),dtype=tf.float32),tf.cast(tf.stack(Offset, axis=0),dtype=tf.float32)