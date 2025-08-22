"""
Configurações do projeto de detecção de bananas
"""
import os

class Config:
    # Diretórios de dados
    TRAIN_IMAGES_DIR = 'data/bananas_train/images'
    TRAIN_LABELS_FILE = 'data/bananas_train/label.csv'
    VAL_IMAGES_DIR = 'data/bananas_val/images'
    VAL_LABELS_FILE = 'data/bananas_val/label.csv'
    
    # Parâmetros da imagem
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    CHANNELS = 3
    INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    
    # Parâmetros de treinamento
    BATCH_SIZE = 100
    NUM_CLASSES = 4  # xmin, ymin, xmax, ymax
    EPOCHS = 200
    LEARNING_RATE = 0.001
    
    # Parâmetros de augmentação
    NUM_AUGMENTATIONS_PER_IMAGE = 6
    MAX_TRANSLATION_X = 0.2
    MAX_TRANSLATION_Y = 0.2
    
    # Parâmetros do modelo
    KERNEL_REGULARIZER = 5e-4
    BIAS_REGULARIZER = 5e-4
    DROPOUT_RATE = 0.3
    
    # Callbacks
    REDUCE_LR_FACTOR = 0.9
    REDUCE_LR_PATIENCE = 2
    MIN_LR = 0.00001
    
    # Caminhos para salvamento
    MODEL_SAVE_DIR = 'saved_models'
    MODEL_NAME = 'inception_banana_detector'
    
    # Configurações do ambiente TensorFlow
    TF_CPP_MIN_LOG_LEVEL = '3'
    TF_FORCE_GPU_ALLOW_GROWTH = 'true'
    
    @classmethod
    def setup_tf_environment(cls):
        """Configura o ambiente TensorFlow"""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = cls.TF_CPP_MIN_LOG_LEVEL
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = cls.TF_FORCE_GPU_ALLOW_GROWTH