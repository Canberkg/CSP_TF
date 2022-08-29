import tensorflow as tf

class classification_loss(object):
    def __init__(self):
        super(classification_loss, self).__init__()
        self.bce=tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction="none")

    def cls_loss(self,pred_map,gt_map):

        gaussian_mask = gt_map[:,:,:,0]
        object_mask   = gt_map[:,:,:,1]
        center_map    = gt_map[:,:,:,2]

        pred_map = pred_map[:,:,:,0]

        log_loss=self.bce(tf.expand_dims(center_map,axis=-1),tf.expand_dims(pred_map,axis=-1))

        positive = center_map
        negative = object_mask-center_map

        foreground_weight = positive*((1-pred_map)**2)
        background_weight = negative*((1-gaussian_mask)**4)*(pred_map**2)

        focal_weight = foreground_weight + background_weight
        num_obj = max(1.0,tf.math.reduce_sum(center_map))

        loss = 0.01 * tf.math.reduce_sum(log_loss*focal_weight) / num_obj

        return loss

class regression_loss(object):
    def __init__(self):
        super(regression_loss, self).__init__()
        self.sigma=3

    def _smooth_l1(self,pred_map, gt_map,positive):
        sigma_squared = self.sigma ** 2

        regression = pred_map
        regression_target = gt_map
        positive = positive


        regression_diff = regression - regression_target
        regression_diff = tf.keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        loss= regression_loss*positive
        normalizer = max(1.0,tf.math.reduce_sum(positive))
        return tf.keras.backend.sum(loss) / normalizer

    def scale_loss(self,pred_map,gt_map):
        return self._smooth_l1(pred_map[:,:,:,0],gt_map[:,:,:,0],gt_map[:,:,:,2])
    def offset_loss(self,pred_map,gt_map):
        return 0.1*self._smooth_l1(pred_map[:,:,:,:2],gt_map[:,:,:,:2],tf.expand_dims(gt_map[:,:,:,2],axis=-1))