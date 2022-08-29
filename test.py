import os
import cv2
import tensorflow as tf

from config import csp_cfg
from Primary.nets.csp import csp
from Primary.utils.inference import test_image,visualize_bbox,inference


def image_eval():
    IMG_WIDTH       = csp_cfg['IMG_WIDTH']
    IMG_HEIGHT      = csp_cfg['IMG_HEIGHT']
    BATCH_SIZE      = csp_cfg['BATCH_SIZE']
    NUM_CLASS       = csp_cfg['NUM_CLASS']
    NMS_THRESHOLD   = csp_cfg['NMS_THRESHOLD']
    SAVE_DIR        = csp_cfg['SAVE_DIR']
    TEST_IMAGE      = csp_cfg['TEST_IMAGE']
    TEST_MODEL_NAME = csp_cfg['TEST_MODEL_NAME']

    #DEFINE AND LOAD MODEL
    csp_model = csp(inp=tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3),batch_size=BATCH_SIZE),num_class=NUM_CLASS,model_name="resnet50")
    csp_model.build(input_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3))
    csp_model.load_weights(os.path.join(SAVE_DIR, "{}.h5".format(TEST_MODEL_NAME)))
    csp_model.trainable = False
    csp_model.summary()

    #LOAD IMAGE
    image = test_image(path=TEST_IMAGE,img_width=IMG_WIDTH,img_height=IMG_HEIGHT)

    #INFERENCE
    center_pred, scale_pred, offset_pred = csp_model(image, training=False)
    boxes,scores = inference(center_prob=center_pred, height=scale_pred, offset=offset_pred, img_width=IMG_WIDTH,
                             img_height=IMG_HEIGHT, ar=0.41, score_thresh=0.55, nms_thresh=NMS_THRESHOLD, downsample=4)
    image_vis = cv2.imread(filename=TEST_IMAGE)
    image_vis = cv2.resize(image_vis, dsize=(IMG_WIDTH, IMG_HEIGHT))
    image_with_boxes = visualize_bbox(image=image_vis, bb=boxes, scores=scores, color=(0,255,0))
    cv2.imshow("image_detection", image_with_boxes)
    cv2.waitKey()


if __name__ == '__main__':
    image_eval()