img_path="D:\\Thesis\\Datasets\\CityScapes\\Realimg_trainvaltest\\leftImg8bit\\train\\train"
label_path="D:\\Thesis\\Datasets\\CityScapes\\gtBbox_cityPersons_trainval\\gtBboxCityPersons\\train\\train"

from Primary.data_generator.datagen import data_gen
import tensorflow as tf
from Primary.nets.csp import csp
from Primary.loss.csp_loss import classification_loss,regression_loss

datagen=data_gen(img_path,label_path,1280,640,1,2)
for i in range(datagen.__len__()):
    img, center,scale,offset=datagen.__getitem__(i)
    csp_model = csp(inp=tf.keras.Input(shape=(640,1280,3),batch_size=1),num_class=2,model_name="resnet50")
    center_pred,scale_pred,offset_pred = csp_model(img, training=False)
    cls_loss = classification_loss().cls_loss(pred_map=center_pred, gt_map=center)
    scale_reg = regression_loss().scale_loss(pred_map=scale_pred, gt_map=scale)
    offset_reg = regression_loss().offset_loss(pred_map=offset_pred, gt_map=offset)
    total_loss = cls_loss + scale_reg + offset_reg
    print(total_loss)
