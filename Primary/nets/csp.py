import numpy as np
import tensorflow as tf

from Primary.utils.backbone import load_mdl

class l2_norm(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(l2_norm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [tf.keras.layers.InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = tf.keras.backend.variable(init_gamma, name='{}_gamma'.format(self.name))

    def call(self, x, mask=None):
        output = tf.keras.backend.l2_normalize(x, self.axis)
        output *= self.gamma
        return output

class conv_block(tf.keras.Model):
    def __init__(self,num_filter,kernel_size,stride,padding,name):
        super(conv_block, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=num_filter, kernel_size=kernel_size, strides=stride,
                                           padding=padding,kernel_initializer='he_normal', name="_conv".format(name))
        self.bn=tf.keras.layers.BatchNormalization(beta_initializer='he_normal',gamma_initializer='he_normal',
                                                   name="_bn".format(name))
        self.relu=tf.keras.layers.ReLU(name="_relu".format(name))

    def call(self, inputs, training=None, mask=None):

        out=self.conv(inputs)
        out=self.bn(out)
        out=self.relu(out)
        return out

class dilated_bottleneck(tf.keras.Model):
    def __init__(self,num_filter,block_name):
        super(dilated_bottleneck, self).__init__()

        self.num_filter=num_filter
        self.CN_1 = tf.keras.layers.Conv2D(filters=self.num_filter/4,kernel_size=(1,1),strides=(1,1),padding='same',
                                           kernel_initializer='he_normal',name='{}_CN_1'.format(block_name),
                                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.CN_2 = tf.keras.layers.Conv2D(filters=self.num_filter/4,kernel_size=(3,3),strides=(1,1),padding='same',
                                           dilation_rate=2,kernel_initializer='he_normal',
                                           name='{}_CN_2'.format(block_name),
                                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.CN_3 = tf.keras.layers.Conv2D(filters=self.num_filter,kernel_size=(1, 1), strides=(1, 1), padding='same',
                                           kernel_initializer='he_normal', name='{}_CN_3'.format(block_name),
                                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.CN_4 = tf.keras.layers.Conv2D(filters=self.num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                           kernel_initializer='he_normal', name='{}_CN_4'.format(block_name),
                                           kernel_regularizer=tf.keras.regularizers.l2(l=0.01))


        self.BN_block_1 = tf.keras.layers.BatchNormalization(beta_initializer='he_normal',
                                                             gamma_initializer='he_normal',name=f"{block_name}_bn_1")
        self.BN_block_2 = tf.keras.layers.BatchNormalization(beta_initializer='he_normal',
                                                             gamma_initializer='he_normal',name=f"{block_name}_bn_2")
        self.BN_block_3 = tf.keras.layers.BatchNormalization(beta_initializer='he_normal',
                                                             gamma_initializer='he_normal',name=f"{block_name}_bn_3")
        self.BN_block_4 = tf.keras.layers.BatchNormalization(beta_initializer='he_normal',
                                                             gamma_initializer='he_normal', name=f"{block_name}_bn_4")

        self.relu=tf.keras.layers.ReLU()


    def call(self, inputs, training=None, mask=None):

        out=self.CN_1(inputs)
        out=self.BN_block_1(out)
        out=self.relu(out)

        out = self.CN_2(out)
        out = self.BN_block_2(out)
        out = self.relu(out)

        out = self.CN_3(out)
        out = self.BN_block_3(out)

        if inputs.shape[-1] != self.num_filter:
            res=self.CN_4(inputs)
            res=self.BN_block_4(res)
            out=tf.math.add(res,out)
        else:
            out = tf.math.add(inputs, out)

        out = self.relu(out)
        return out

class csp(tf.keras.Model):
    def __init__(self,inp,num_class,model_name):
        super(csp, self).__init__()
        assert model_name is not None,"backbone must be specified!"
        self.resnet=load_mdl(inputs=inp,num_class=num_class,model_type=model_name)

        self.dilated_blk_1 = dilated_bottleneck(num_filter=2048,block_name="dilated_block_1")
        self.dilated_blk_2 = dilated_bottleneck(num_filter=2048, block_name="dilated_block_2")
        self.dilated_blk_3 = dilated_bottleneck(num_filter=2048, block_name="dilated_block_3")

        self.dilated_stg=tf.keras.Sequential()
        self.dilated_stg.add(self.dilated_blk_1)
        self.dilated_stg.add(self.dilated_blk_2)
        self.dilated_stg.add(self.dilated_blk_3)

        self.deconv_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=2,
                                                        padding="same", name='deconv_1')
        self.deconv_2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=4,
                                                        padding="same", name='deconv_2')
        self.deconv_3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=4,
                                                        padding="same", name='deconv_3')

        self.l2_norm_1 = l2_norm(scale=10)
        self.l2_norm_2 = l2_norm(scale=10)
        self.l2_norm_3 = l2_norm(scale=10)

        self.downsample_dethead=conv_block(num_filter=256,kernel_size=(1,1),stride=1,
                                           padding="same",name="downsample_dethead")

        self.center=tf.keras.layers.Conv2D(filters=1,kernel_size=(1,1),strides=1,
                                           padding="same",name="center_kernel")
        self.scale = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1,
                                             padding="same",name="scale_kernel")
        self.offset = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=1,
                                             padding="same",name="offset_kernel")

    def call(self, inputs, training=None, mask=None):

        bbone_out = self.resnet(inputs)

        stg_3 = self.deconv_1(bbone_out[1])
        stg_3 = self.l2_norm_1(stg_3)

        stg_4 = self.deconv_2(bbone_out[2])
        stg_4 = self.l2_norm_2(stg_4)

        stg_5 = self.dilated_stg(bbone_out[2])
        stg_5 = self.deconv_3(stg_5)
        stg_5 = self.l2_norm_3(stg_5)

        concat_out= tf.concat([stg_3,stg_4,stg_5],axis=-1)

        out=self.downsample_dethead(concat_out)

        center_heatmap = self.center(out)
        center_heatmap = tf.nn.sigmoid(center_heatmap)
        scale_map = self.scale(out)
        offset_map = self.offset(out)

        return center_heatmap,scale_map,offset_map

    def model(self, inputs):
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

