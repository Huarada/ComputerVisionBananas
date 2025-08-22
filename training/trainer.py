"""
Módulo para treinamento do modelo
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from config.config import Config

class ModelTrainer:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or Config()
        
    def get_callbacks(self):
        """
        Retorna lista de callbacks para treinamento
        
        Returns:
            list: Lista de callbacks
        """
        reduce_lr = ReduceLROnPlateau(
            monitor="val_mean_absolute_error",
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.MIN_LR,
            verbose=True
        )
        
        return [reduce_lr]
    
    def train(self, train_dataset, val_data):
        """
        Treina o modelo
        
        Args:
            train_dataset: Dataset de treinamento
            val_data: Dados de validação (X_val, y_val)
            
        Returns:
            tf.keras.callbacks.History: Histórico do treinamento
        """
        X_val, y_val = val_data
        
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            verbose=2,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, filepath=None):
        """
        Salva o modelo treinado
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if filepath is None:
            os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
            filepath = os.path.join(
                self.config.MODEL_SAVE_DIR,
                f"{self.config.MODEL_NAME}.keras"
            )
        
        self.model.save(filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def plot_model_architecture(self, filepath=None):
        """
        Plota a arquitetura do modelo
        
        Args:
            filepath: Caminho para salvar o diagrama
        """
        if filepath is None:
            filepath = f"{self.config.MODEL_NAME}_architecture.png"
            
        plot_model(
            self.model,
            to_file=filepath,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )
        print(f"Arquitetura do modelo salva em: {filepath}")
    
    def plot_training_history(self, history):
        """
        Plota o histórico de treinamento
        
        Args:
            history: Histórico retornado pelo fit()
        """
        # Plot MAE
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mean_absolute_error'], label='Train MAE')
        plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        plt.title('Model Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Salvar gráficos
        plt.savefig(f"{self.config.MODEL_NAME}_training_history.png", dpi=300, bbox_inches='tight')
        print(f"Histórico de treinamento salvo como {self.config.MODEL_NAME}_training_history.png")
    
    def print_model_summary(self):
        """Imprime o resumo do modelo"""
        self.model.summary()