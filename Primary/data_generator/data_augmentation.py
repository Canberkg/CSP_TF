import random
import cv2
import numpy as np

class image_augmentation(object):
    """Image Augmentation Class

    Attributes :
        flip : Specify if flip operation will be performed or not. If it will, use horizontal or vertical. Otherwise, None.

    """

    def __init__(self,flip=None):
        super(image_augmentation, self).__init__()
        self.flip=flip

    def aug_random_crop(self,img,labels,crop_h,crop_w,limit=8):
        """Perform random crop
        Params:
            img  : Image
            labels : ground truth boxes as an Array
            crop_h : height of the cropped region
            crop_w : width of the cropped region
        Return:
            img : cropped image
            labels : ground truth boxes in that cropped region
        """

        img_height,img_width=img.shape[0],img.shape[1]

        gt_id = random.randint(0,len(labels)-1)
        crop_cx = int((labels[gt_id, 0] + labels[gt_id, 2]) / 2)
        crop_cy = int((labels[gt_id, 1] + labels[gt_id, 3]) / 2)
        crop_x1 = max(crop_cx - (crop_w / 2), int(0))
        crop_y1 = max(crop_cy - (crop_h / 2), int(0))
        diff_x  = max(crop_x1 + crop_w - img_width, int(0))
        crop_x1 -= diff_x
        crop_x1 = int(crop_x1)
        diff_y = max(crop_y1 + crop_h - img_height, int(0))
        crop_y1 -= diff_y
        crop_y1 = int(crop_y1)
        cropped_image = np.copy (img[crop_y1:crop_y1+crop_h,crop_x1:crop_x1+crop_w,:])

        org_labels = np.copy(labels)
        labels[:, 0] -= crop_x1
        labels[:, 2] -= crop_x1
        labels[:, 1] -= crop_y1
        labels[:, 3] -= crop_y1
        labels[:, 0] = np.clip(labels[:, 0],0,crop_w)
        labels[:, 2] = np.clip(labels[:, 2],0,crop_w)
        labels[:, 1] = np.clip(labels[:, 1],0,crop_h)
        labels[:, 3] = np.clip(labels[:, 3],0,crop_h)

        before_area = (org_labels[:, 2] - org_labels[:, 0]) * (org_labels[:, 3] - org_labels[:, 1])
        after_area = (labels[:, 2] - labels[:, 0]) * (labels[:, 3] - labels[:, 1])

        keep_inds = ((labels[:, 2] - labels[:, 0]) >= limit) & \
                    (after_area >= 0.5 * before_area)
        labels = labels[keep_inds]

        return cropped_image,labels

    def aug_brightness(self,img, min_val, max_val, prob = 0.5):
        """Perform random color distortion
        Params:
            img     : Image
            min_val : minimum scale of distortion
            max_val : maximum scale of distortion
            prob    : probability of whether the augmentation will be performed or not
        Return:
            img: Either the same Image or Augmented Image
        """

        augmentation_prob = random.random()
        if  augmentation_prob > prob:
            img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            distortion_scale = np.random.uniform(min_val,max_val)
            exceeded_pixels = img_hsv[:,:,2] * distortion_scale > 255
            distortion_channel = np.where(exceeded_pixels,255,img_hsv[:,:,2] * distortion_scale)
            img_hsv[:,:,2] = distortion_channel
            img_rgb = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
            return img_rgb
        else:
            return img


    def aug_flip(self, img, labels,prob=0.5):
        """Perform horizontal/vertical flip operations with a random probability
        Params:
            img: Image array (Array)
            labels: Label Tensor (Tensor)
        Return:
            img: Augmented Image
            labels: Augmented Labels
        """

        if self.flip == 'horizontal':
            labels = np.stack([(labels[:, 0] + labels[:, 2]) / 2, (labels[:, 1] + labels[:, 3]) / 2,
                               labels[:, 2] - labels[:, 0],labels[:, 3] - labels[:, 1]], axis=1)
            augmentation_prob = random.random()
            if augmentation_prob > prob :
                img = img[:, ::-1, :]
                labels = np.stack([img.shape[1] - labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]], axis=1)
            labels = np.stack([labels[:, 0] - (labels[:, 2] / 2), labels[:, 1] - (labels[:, 3] / 2),
                               labels[:, 0] + (labels[:, 2] / 2), labels[:, 1] + (labels[:, 3] / 2)], axis=1)
            return img, labels
        elif self.flip == 'vertical':
            labels = np.stack([(labels[:, 0] + labels[:, 2]) / 2, (labels[:, 1] + labels[:, 3]) / 2,
                               labels[:, 2] - labels[:, 0], labels[:, 3] - labels[:, 1]], axis=1)
            augmentation_prob = random.random()
            if augmentation_prob > prob:
                img = img[::-1, :, :]
                labels = np.stack([labels[:, 0], 1 - labels[:, 1], labels[:, 2], labels[:, 3]], axis=1)
            labels = np.stack([labels[:, 0] - (labels[:, 2] / 2), labels[:, 1] - (labels[:, 3] / 2),
                               labels[:, 0] + (labels[:, 2] / 2), labels[:, 1] + (labels[:, 3] / 2)], axis=1)
            return img, labels
        elif self.flip == None:
            return img,labels
        else:
            raise ValueError("Must select an existing flip operation.Use horizontal, vertical or None!")