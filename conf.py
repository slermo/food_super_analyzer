class Config:
    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "resnet50"
    
    # Какие слои размораживаем - совпадают с неймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"  
    IMAGE_MODEL_UNFREEZE = "layer.3|layer.4"  
    
    # Гиперпараметры
    BATCH_SIZE = 32
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    REGRESSOR_LR = 5e-4
    EPOCHS = 10
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 5
    
    # Пути
    TRAIN_DF_PATH = "path/train.csv"
    VAL_DF_PATH = "path/val.csv"
    SAVE_PATH = "best_model.pth"