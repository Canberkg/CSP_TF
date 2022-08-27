import tensorflow as tf

def load_mdl(inputs,num_class,model_type):
    """Load a pre-trained backbone model
    Params:
        inputs: Input tensor (Tensor)
        num_class: Number of Classes (Int)
        model_type: Name of the model (String)
    Return:
        model
    """
    if model_type == 'resnet50':

        Resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            input_shape=None,
            pooling=None,
            classes=num_class
        )
        for layer in Resnet.layers:
            if layer.name == 'input_1':
                layer.trainable=False
            elif layer.name.split('_')[0][-1] == '1':
                layer.trainable = False
            else:
                layer.trainable=False

        for layer in Resnet.layers:
            print("{0}:\t{1}".format(layer.trainable, layer.name))

        out_1 = Resnet.get_layer(name="conv2_block3_out").output
        out_2 = Resnet.get_layer(name="conv3_block4_out").output
        out_3 = Resnet.get_layer(name="conv4_block6_out").output
        out_4 = Resnet.get_layer(name="conv5_block3_out").output
        model=tf.keras.Model(inputs=Resnet.input,outputs=[out_1,out_2,out_3,out_4])
        return model
    elif model_type == 'resnet101':
        Resnet = tf.keras.applications.ResNet101(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            input_shape=None,
            pooling=None,
            classes=num_class
        )

        out_1 = Resnet.get_layer(name="conv2_block3_out").output
        out_2 = Resnet.get_layer(name="conv3_block4_out").output
        out_3 = Resnet.get_layer(name="conv4_block23_out").output
        out_4 = Resnet.get_layer(name="conv5_block3_out").output
        model = tf.keras.Model(inputs=Resnet.input, outputs=[out_1, out_2, out_3,out_4])
        return model
    else:
        raise Exception('Need to specify a possessed model name')