csp_cfg = {
    ## Training Settings
    'BATCH_SIZE' :  1,
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
    "TEST_MODEL_NAME" : "Name of the Model to be Tested",
    "TEST_IMAGE"      : "Path of the Test Image",
    "NMS_THRESHOLD"   : 0.5,

    ## Model Save and Checkpoint Settings
    'SAVE_DIR'   : "Saved_Models",
    'MODEL_NAME' : 'Name of the Model',
    'CKPT_DIR'   : 'Name of the Checkpoint Directory',

    ## Dataset Settings
    'TRAIN_IMG'   : "Dataset/images/train",
    'VALID_IMG'   : "Dataset/images/valid",
    "TRAIN_LABEL" : "Dataset/annotations_json/train",
    "VALID_LABEL" : "Dataset/annotations_json/valid",
    "SUBSET"  : "reasonable", ## (Reasonable : "reasonable", Small : "small", Highly occluded : "highly_occluded", No Subset : "None")
    "AUGMENTATION" : True,
    'SHUFFLE' : True,

    ## Dataset Prepearation
    'IMG_PATH'       : "Path of leftImg8bit directory",
    'JSON_ANNOTATION': "Path of gtBboxCityPersons",
    'SET_IMG_PATH'  : "Dataset/images/train",
    'SET_JSON_PATH' : "Dataset/images/valid"

}