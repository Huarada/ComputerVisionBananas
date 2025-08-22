"""
Módulo para data augmentation com ajuste de bounding boxes
"""
import numpy as np
import tensorflow as tf
from config.config import Config

class DataAugmentation:
    def __init__(self, config=None):
        self.config = config or Config()
    
    def adjust_bounding_boxes(self, bboxes, dx, dy):
        """
        Ajusta as coordenadas dos bounding boxes conforme a translação aplicada
        
        Args:
            bboxes: Lista com [xmin, ymin, xmax, ymax] normalizadas
            dx, dy: Deslocamentos normalizados
            
        Returns:
            list: Coordenadas ajustadas
        """
        bboxes_new = [0, 0, 0, 0]
        bboxes_new[0] = bboxes[0] + dx / self.config.IMAGE_WIDTH
        bboxes_new[1] = bboxes[1] + dy / self.config.IMAGE_HEIGHT  
        bboxes_new[2] = bboxes[2] + dx / self.config.IMAGE_WIDTH
        bboxes_new[3] = bboxes[3] + dy / self.config.IMAGE_HEIGHT
        return bboxes_new
    
    def manual_translation(self, image, bboxes, max_dx, max_dy, img_width, img_height):
        """
        Aplica translação manual e ajusta os bounding boxes
        
        Args:
            image: Imagem a ser transladada
            bboxes: Coordenadas do bounding box
            max_dx, max_dy: Máximo deslocamento permitido
            img_width, img_height: Dimensões da imagem
            
        Returns:
            tuple: (imagem_transladada, bboxes_ajustados)
        """
        # Gerar deslocamento aleatório
        dx = tf.random.uniform([], -max_dx, max_dx)
        dy = tf.random.uniform([], -max_dy, max_dy)
        
        # Converter para pixels
        dx_pixels = dx * img_width
        dy_pixels = dy * img_height
        
        # Matriz de translação
        translation_matrix = [1, 0, dx_pixels, 0, 1, dy_pixels, 0, 0]
        
        # Aplicar translação
        translated_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, axis=0),
            transforms=tf.reshape(translation_matrix, (1, 8)),
            output_shape=tf.shape(image)[:2],
            fill_value=0.0,
            interpolation="BILINEAR"
        )[0]
        
        # Ajustar bounding boxes
        adjusted_bboxes = self.adjust_bounding_boxes(bboxes, dx, dy)
        
        return translated_image, adjusted_bboxes
    
    def augment_dataset(self, images, coordinates):
        """
        Aplica data augmentation ao dataset completo
        
        Args:
            images: Array de imagens originais
            coordinates: Array de coordenadas originais
            
        Returns:
            tuple: (imagens_augmentadas, coordenadas_augmentadas)
        """
        augmented_images = []
        augmented_coords = []
        
        # Adiciona dados originais
        for i in range(len(images)):
            augmented_images.append(images[i])
            augmented_coords.append(coordinates[i])
        
        # Aplica augmentation
        for i in range(len(images)):
            for _ in range(self.config.NUM_AUGMENTATIONS_PER_IMAGE):
                aug_image, aug_coords = self.manual_translation(
                    images[i], 
                    coordinates[i],
                    self.config.MAX_TRANSLATION_X,
                    self.config.MAX_TRANSLATION_Y,
                    self.config.IMAGE_WIDTH,
                    self.config.IMAGE_HEIGHT
                )
                augmented_images.append(aug_image)
                augmented_coords.append(aug_coords)
        
        return np.array(augmented_images), np.array(augmented_coords)
    
    def augment_image_brightness_contrast(self, image):
        """
        Aplica augmentação de brilho e contraste
        
        Args:
            image: Imagem a ser augmentada
            
        Returns:
            tf.Tensor: Imagem augmentada
        """
        image = tf.image.random_brightness(image, max_delta=0.002)
        image = tf.image.random_contrast(image, lower=0, upper=255)
        return image