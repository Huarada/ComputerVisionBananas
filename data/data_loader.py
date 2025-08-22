"""
Módulo para carregamento de dados do dataset de bananas
"""
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config.config import Config

class DataLoader:
    def __init__(self, config=None):
        self.config = config or Config()
        
    def load_training_data(self):
        """Carrega dados de treinamento"""
        return self._load_data(
            self.config.TRAIN_IMAGES_DIR,
            self.config.TRAIN_LABELS_FILE
        )
    
    def load_validation_data(self):
        """Carrega dados de validação"""
        return self._load_data(
            self.config.VAL_IMAGES_DIR,
            self.config.VAL_LABELS_FILE
        )
    
    def _load_data(self, images_dir, labels_file):
        """
        Carrega imagens e labels de um diretório específico
        
        Args:
            images_dir: Diretório com as imagens
            labels_file: Arquivo CSV com os labels
            
        Returns:
            tuple: (imagens, coordenadas_normalizadas)
        """
        # Carrega os rótulos do arquivo CSV
        labels_df = pd.read_csv(labels_file)
        
        # Listas para armazenar dados
        images = []
        coordinates = []
        
        # Obtém lista de arquivos ordenada
        image_files = os.listdir(images_dir)
        image_files_sorted = sorted(image_files, key=lambda x: int(x.split('.')[0]))
        
        # Processa cada imagem
        for i, image_file in enumerate(image_files_sorted):
            # Carrega e redimensiona imagem
            img_path = os.path.join(images_dir, image_file)
            img = load_img(img_path, target_size=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH))
            img_array = img_to_array(img)
            images.append(img_array)
            
            # Normaliza coordenadas
            normalized_coords = [
                labels_df.iloc[i]['xmin'] / self.config.IMAGE_WIDTH,
                labels_df.iloc[i]['ymin'] / self.config.IMAGE_HEIGHT,
                labels_df.iloc[i]['xmax'] / self.config.IMAGE_WIDTH,
                labels_df.iloc[i]['ymax'] / self.config.IMAGE_HEIGHT
            ]
            coordinates.append(normalized_coords)
        
        return np.array(images), np.array(coordinates)
    
    def preprocess_images(self, images):
        """
        Pré-processa as imagens
        
        Args:
            images: Array de imagens
            
        Returns:
            np.array: Imagens pré-processadas
        """
        images = images.astype('float32')
        images /= 255.0
        images = 2 * (images - 0.5)  # Normalização para [-1, 1]
        return images
    
    def get_image_files_sorted(self, images_dir):
        """Retorna lista ordenada de arquivos de imagem"""
        image_files = os.listdir(images_dir)
        return sorted(image_files, key=lambda x: int(x.split('.')[0]))