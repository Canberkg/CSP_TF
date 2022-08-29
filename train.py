import os
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from config import csp_cfg
from Primary.nets.csp import csp
from Primary.loss.csp_loss import classification_loss,regression_loss
from Primary.data_generator.datagen import data_gen


def csp_train(root_dir_train,root_dir_valid,_root_dir_train_jsons,root_dir_valid_jsons):


    BATCH_SIZE    = csp_cfg['BATCH_SIZE']
    EPOCH         = csp_cfg['EPOCH']
    IMG_WIDTH     = csp_cfg['IMG_WIDTH']
    IMG_HEIGHT    = csp_cfg['IMG_HEIGHT']
    NUM_CLASS     = csp_cfg['NUM_CLASS']
    LR            = csp_cfg['LR']
    LR_SCHEDULER  = csp_cfg['LR_SCHEDULER']
    DECAY_STEPS   = csp_cfg['DECAY_STEPS']
    DECAY_RATE    = csp_cfg['DECAY_RATE']
    OPTIMIZER     = csp_cfg['OPTIMIZER']
    SHUFFLE       = csp_cfg['SHUFFLE']
    SAVE_DIR      = csp_cfg['SAVE_DIR']
    MODEL_NAME    = csp_cfg['MODEL_NAME']
    CKPT_DIR      = csp_cfg['CKPT_DIR']


    train_generator = data_gen(Img_Path=root_dir_train,Label_Path=root_dir_train_jsons,
                               Img_Width=IMG_WIDTH,Img_Height=IMG_HEIGHT,Batch_Size=BATCH_SIZE,
                               Num_Classes=NUM_CLASS,Augmentation=None,Shuffle=SHUFFLE)
    valid_generator = data_gen(Img_Path=root_dir_valid, Label_Path=root_dir_valid_jsons,
                               Img_Width=IMG_WIDTH, Img_Height=IMG_HEIGHT, Batch_Size=BATCH_SIZE,
                               Num_Classes=NUM_CLASS, Augmentation=None,Shuffle=False)

    csp_model = csp(inp=tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3),batch_size=BATCH_SIZE),num_class=NUM_CLASS,model_name="resnet50")

    if  LR_SCHEDULER == True:
        lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LR,decay_steps=DECAY_STEPS,decay_rate=DECAY_RATE)
    else:
        lr=LR

    if OPTIMIZER.upper() == 'ADAM' :
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif OPTIMIZER.upper() == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
    else:
        raise Exception("Need to specify an existing optimizer")



    training_loss = tf.metrics.Mean()
    validation_loss = tf.metrics.Mean()
    metrics_names = ["training loss","validation loss"]

    best = 9999
    patience = 30
    wait = 0

    checkpoint=tf.train.Checkpoint(optimizer=optimizer,model=csp_model)
    ckpt_manager=tf.train.CheckpointManager(checkpoint=checkpoint,directory=os.path.join(SAVE_DIR,CKPT_DIR),max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("checkpoint named as {} is restored ".format(ckpt_manager.latest_checkpoint))
    else:
        print("Model will be initialized from scratch as there are no checkpoints !! ")

    train_loss = []
    valid_loss = []
    epochs = []


    def train_step(Image_Batch, center_map,scale_map,offset_map):
        with tf.GradientTape() as tape:
            center_pred,scale_pred,offset_pred = csp_model(Image_Batch, training=True)
            cls_loss = classification_loss.cls_loss(pred_map=center_pred,gt_map=center_map)
            scale_reg = regression_loss.scale_loss(pred_map=scale_pred,gt_map=scale_map)
            offset_reg = regression_loss.offset_loss(pred_map=offset_pred,gt_map=offset_map)
            total_loss=cls_loss+scale_reg+offset_reg
        grads = tape.gradient(total_loss, csp_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, csp_model.trainable_weights))
        return total_loss


    def test_step(Image_Batch, center_map,scale_map,offset_map):
        center_pred,scale_pred,offset_pred = csp_model(Image_Batch, training=False)
        cls_loss = classification_loss.cls_loss(pred_map=center_pred, gt_map=center_map)
        scale_reg = regression_loss.scale_loss(pred_map=scale_pred, gt_map=scale_map)
        offset_reg = regression_loss.offset_loss(pred_map=offset_pred, gt_map=offset_map)
        total_loss = cls_loss + scale_reg + offset_reg
        return total_loss


    for epoch in range(EPOCH):
        pb_i = tf.keras.utils.Progbar(train_generator.num_images(), stateful_metrics=metrics_names)
        batch_training_loss=[]
        batch_validation_loss = []
        print("\n Epoch : {}/{} -".format(epoch, EPOCH))
        for batch_index in range(train_generator.__len__()):
            Images,center_map,scale_map,offset_map = train_generator.__getitem__(index=batch_index)
            batch_loss=train_step(Image_Batch=Images, center_map=center_map,scale_map=scale_map,offset_map=offset_map)
            batch_training_loss.append(batch_loss)
            pb_i.update((batch_index+1) * BATCH_SIZE, values=[('training loss', batch_loss)])
        training_loss.update_state(values=batch_training_loss)
        temp_tl = training_loss.result()
        pb_i.update(current=train_generator.num_images(),values=[('training loss', training_loss.result())], finalize=True)

        for batch_index in range(valid_generator.__len__()):
            Images,center_map,scale_map,offset_map = valid_generator.__getitem__(index=batch_index)
            batch_valid_loss=test_step(Image_Batch=Images, center_map=center_map,scale_map=scale_map,offset_map=offset_map)
            batch_validation_loss.append(batch_valid_loss)
        validation_loss.update_state(values=batch_validation_loss)
        mean_val=validation_loss.result()
        pb_i.update(current=train_generator.num_images(),values=[('training loss', training_loss.result()),('validation loss',mean_val)],finalize=True)

        if (epoch + 1) % 10 == 0:
            csp_model.save_weights(filepath=os.path.join(SAVE_DIR,"{}.h5".format(MODEL_NAME)),overwrite=True, save_format='h5', options=None)

        training_loss.reset_states()
        validation_loss.reset_states()

        train_loss.append(temp_tl)
        valid_loss.append(mean_val)
        epochs.append(epoch)

        train_generator.on_epoch_end()
        valid_generator.on_epoch_end()
        ckpt_manager.save()
        wait += 1
        if temp_tl < best:
            wait = 0
            best = temp_tl
        if wait >= patience:
            break

    csp_model.save_weights(filepath=os.path.join(SAVE_DIR,"{}.h5".format(MODEL_NAME)),overwrite=True, save_format='h5', options=None)
    training_loss_graph = np.stack((epochs, train_loss), axis=-1)
    validation_loss_graph=np.stack((epochs,valid_loss),axis=-1)
    plt.plot(training_loss_graph[..., 0], training_loss_graph[..., 1], 'r', label='Training loss')
    plt.plot(validation_loss_graph[..., 0], validation_loss_graph[..., 1], 'g', label='Validation loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training loss','Validation Loss'])
    plt.show()


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    root_dir_train       = csp_cfg['TRAIN_IMG']
    root_dir_valid       = csp_cfg['VALID_IMG']
    root_dir_train_jsons = csp_cfg['TRAIN_LABEL']
    root_dir_valid_jsons = csp_cfg['VALID_LABEL']

    csp_train(root_dir_train,root_dir_valid,root_dir_train_jsons,root_dir_valid_jsons)