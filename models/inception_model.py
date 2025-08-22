"""
Módulo com a arquitetura Inception para detecção de bananas
"""
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv2D, BatchNormalization, Activation, Dropout,
    AveragePooling2D, MaxPooling2D, Input, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from config.config import Config

class InceptionBananaDetector:
    def __init__(self, config=None):
        self.config = config or Config()
        
    def inception_module(self, n_filters, x):
        """
        Módulo Inception customizado
        
        Args:
            n_filters: Número base de filtros
            x: Input tensor
            
        Returns:
            tf.Tensor: Output do módulo Inception
        """
        k_weight = self.config.KERNEL_REGULARIZER
        b_weight = self.config.BIAS_REGULARIZER
        
        # Torre 1: Conv 1x1
        tower_0 = Conv2D(
            n_filters, (1, 1), 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(k_weight),
            bias_regularizer=l2(b_weight)
        )(x)
        
        # Torre 2: Conv 1x1 -> Conv 3x3
        tower_1 = Conv2D(
            2 * n_filters, (1, 1), 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(k_weight),
            bias_regularizer=l2(b_weight)
        )(x)
        tower_1 = Conv2D(
            2 * n_filters, (3, 3), 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(k_weight),
            bias_regularizer=l2(b_weight)
        )(tower_1)
        
        # Torre 3: Conv 1x1 -> Conv 5x5
        tower_2 = Conv2D(
            n_filters // 2, (1, 1), 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(k_weight),
            bias_regularizer=l2(b_weight)
        )(x)
        tower_2 = Conv2D(
            n_filters // 2, (5, 5), 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(k_weight),
            bias_regularizer=l2(b_weight)
        )(tower_2)
        
        # Torre 4: MaxPool -> Conv 1x1
        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        tower_3 = Conv2D(
            n_filters // 2, (1, 1), 
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(k_weight),
            bias_regularizer=l2(b_weight)
        )(tower_3)
        
        # Concatenar todas as torres
        x = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)
        x = BatchNormalization()(x)
        
        return x
    
    def build_model(self):
        """
        Constrói o modelo Inception completo
        
        Returns:
            tf.keras.Model: Modelo compilado
        """
        inputs = Input(shape=self.config.INPUT_SHAPE)
        
        # Primeira sequência de módulos Inception
        x = self.inception_module(64, inputs)
        x = self.inception_module(64, x)
        x = MaxPooling2D(2)(x)
        
        # Segunda sequência de módulos Inception
        x = self.inception_module(64, x)
        x = self.inception_module(64, x)
        x = MaxPooling2D(2)(x)
        
        # Terceira sequência de módulos Inception
        x = self.inception_module(64, x)
        x = self.inception_module(64, x)
        
        # Camadas finais
        output = AveragePooling2D(8)(x)
        output = Flatten()(output)
        outputs = Dense(4, activation='linear')(output)
        
        # Criar modelo
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def compile_model(self, model):
        """
        Compila o modelo com otimizador e métricas
        
        Args:
            model: Modelo a ser compilado
            
        Returns:
            tf.keras.Model: Modelo compilado
        """
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        return model
    
    def create_model(self):
        """
        Cria e compila o modelo completo
        
        Returns:
            tf.keras.Model: Modelo pronto para treinamento
        """
        model = self.build_model()
        model = self.compile_model(model)
        return model