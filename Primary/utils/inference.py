import os
from cv2 import cv2
import numpy as np
import tensorflow as tf

def test_image(path,img_width,img_height):

    Images=[]
    Img = cv2.imread(filename=path)
    Img = cv2.resize(Img, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    Img_arr = np.asarray(Img, dtype=np.float32)
    Img_arr = tf.keras.applications.resnet50.preprocess_input(Img_arr)
    Images.append(Img_arr)

    return tf.stack(Images,axis=0)

def visualize_bbox(image,bb,scores,color):
    """Visualize the boxes in the image
    Params:
        image: Image
        BB: Boxes to display (Tensor)
    Return:
        image: Image with bounding boxes
    """
    NUM_BB=bb.shape[0]
    for i in range(NUM_BB):
        score="{:.0%}".format(scores[i])
        cv2.rectangle(image,pt1=(int(bb[i,0]),int(bb[i,1])),pt2=(int(bb[i,2]),int(bb[i,3])),color=color,thickness=1)
        cv2.putText(img=image, text=score, org=(int(bb[i,0]), int(bb[i,1]) - 5), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 255), thickness=1)
    return image
def nms(bboxes,thresh):
    """Perform Non-max Suppression
    Params:
        BBoxes: Batch of Bounding Boxes (Tensor)
        BBox_scores: Batch of Bounding Box scores (Tensor)
    Return:
        boxes: Remaining bounding boxes (Tensor)
        scores: Scores of remaining bounding boxes (Tensor)
        classes Classes of remaining bounding boxes (Tensor)
    """
    chosen_boxes =[]
    chosen_scores = []
    bbox = bboxes[:,:4]
    bbox_scr = bboxes[:,4]
    box_indices = tf.image.non_max_suppression(boxes=bbox, scores=bbox_scr, max_output_size=50, iou_threshold=thresh)
    selected_box = tf.gather(bbox, box_indices)
    selected_score = tf.gather(bbox_scr, box_indices)
    chosen_boxes.append(selected_box)
    chosen_scores.append(selected_score)
    boxes = tf.concat(values=chosen_boxes, axis=0)
    scores = tf.concat(values=chosen_scores, axis=0)

    return boxes,scores
def inference(center_prob,height,offset,img_width,img_height,ar,score_thresh=0.50,nms_thresh=0.5,downsample=4):
    detected_boxes=[]
    center_prob = tf.squeeze(center_prob)
    height = tf.squeeze(height)
    offset_x = offset[0, :, :, 1]
    offset_y = offset[0, :, :, 0]
    center_indices = tf.where(center_prob>score_thresh)
    center_y, center_x = center_indices[:,0],center_indices[:,1]
    if center_y.shape[0] > 0:
        for i in range(center_y.shape[0]):
            print(i)
            h = tf.math.exp(height[center_y[i],center_x[i]])*downsample
            w = h*ar
            off_y = offset_y [center_y[i],center_x[i]]
            off_x = offset_x[center_y[i], center_x[i]]
            center_score = center_prob[center_y[i],center_x[i]]
            x1 = max(0.0,(float(center_x[i])+off_x+0.5)*downsample-(w/2))
            y1 = max(0.0, (float(center_y[i]) + off_y + 0.5) * downsample - (h / 2))
            detected_boxes.append([x1,y1,min(img_width,x1+w),min(img_height,y1+h),center_score])
        detected_boxes = tf.stack(detected_boxes,axis=0)
        detected_boxes = tf.cast(detected_boxes,dtype=tf.float32)
        boxes,scores = nms(bboxes=detected_boxes,thresh=nms_thresh)
        return boxes,scores

