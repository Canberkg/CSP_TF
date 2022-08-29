csp_cfg = {
    ## Training Settings
    'BATCH_SIZE' :  4,
    'EPOCH'      :  20,
    'IMG_WIDTH'  :  1280,
    'IMG_HEIGHT' :  640,
    'OPTIMIZER'  :  'ADAM',
    'LR'         :  0.001,
    'LR_SCHEDULER' : False,
        'DECAY_STEPS' : 1000,
        'DECAY_RATE'  : 0.96,
    'NUM_CLASS'  :  2,

    ## Test Settings
    "TEST_MODEL_NAME" : "Model_Test_3",
    "TEST_IMAGE"      : "D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\Img\\val\\2008_002835.jpg",
    "NMS_THRESHOLD"   : 0.5,

    ## Model Save and Checkpoint Settings
    'SAVE_DIR'   : "D:\\PersonalResearch\\Projects\\Models\\SSD",
    'MODEL_NAME' : 'Name of the Model',
    'CKPT_DIR'   : 'Name of the Checkpoint Directory',

    ## Dataset Settings
    'TRAIN_IMG'   : "Dataset/images/train",
    'VALID_IMG'   : "Dataset/images/valid",
    "TRAIN_LABEL" : "Dataset/annotations_json/train",
    "VALID_LABEL" : "Dataset/annotations_json/valid",
    'SHUFFLE' : True,

    ## Dataset Prepearation
    'IMG_PATH'       : "Directory Path of Images",
    'XML_ANNOTATION' : "Directory Path of XML Annotations",
    'JSON_ANNOTATION': "Directory Path of JSON Annotations",
    'TRAIN_SPLIT'    : "Directory Path of training set txt file which contains the name of the all the image names belong to training",
    'VALID_SPLIT'    : "Directory Path of training set txt file which contains the name of the all the image names belong to validation",

}