"""
Script principal para detecção de bananas usando Inception Network
"""
import os
import tensorflow as tf

# Configurar ambiente TensorFlow
from config.config import Config
Config.setup_tf_environment()

from data.data_loader import DataLoader
from data.augmentation import DataAugmentation
from models.inception_model import InceptionBananaDetector
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from evaluation.visualization import ResultVisualizer

def main():
    """Função principal do programa"""
    print("=" * 50)
    print("DETECÇÃO DE BANANAS - INCEPTION NETWORK")
    print("=" * 50)
    
    # Configuração
    config = Config()
    
    # 1. CARREGAMENTO DE DADOS
    print("\n1. Carregando dados...")
    data_loader = DataLoader(config)
    
    # Dados de treinamento
    train_images, train_coords = data_loader.load_training_data()
    print(f"Dados de treinamento carregados: {len(train_images)} imagens")
    
    # Dados de validação
    val_images, val_coords = data_loader.load_validation_data()
    val_images_processed = data_loader.preprocess_images(val_images)
    print(f"Dados de validação carregados: {len(val_images)} imagens")
    
    # 2. DATA AUGMENTATION
    print("\n2. Aplicando data augmentation...")
    augmentation = DataAugmentation(config)
    augmented_images, augmented_coords = augmentation.augment_dataset(train_images, train_coords)
    
    # Preprocessar dados augmentados
    augmented_images_processed = data_loader.preprocess_images(augmented_images)
    print(f"Dataset augmentado: {len(augmented_images)} imagens")
    
    # Criar dataset do TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (augmented_images_processed, augmented_coords)
    ).batch(config.BATCH_SIZE)
    
    # 3. CONSTRUÇÃO DO MODELO
    print("\n3. Construindo modelo Inception...")
    model_builder = InceptionBananaDetector(config)
    model = model_builder.create_model()
    
    print("Arquitetura do modelo:")
    model.summary()
    
    # 4. TREINAMENTO
    print(f"\n4. Iniciando treinamento por {config.EPOCHS} épocas...")
    trainer = ModelTrainer(model, config)
    
    # Plotar arquitetura
    trainer.plot_model_architecture()
    
    # Treinar modelo
    history = trainer.train(train_dataset, (val_images_processed, val_coords))
    
    # Plotar histórico
    trainer.plot_training_history(history)
    
    # Salvar modelo
    trainer.save_model()
    
    # 5. AVALIAÇÃO
    print("\n5. Avaliando modelo...")
    evaluator = ModelEvaluator(model, config)
    
    # Avaliar no conjunto de validação
    results = evaluator.evaluate_model(val_images_processed, val_coords)
    print(f"Loss: {results['loss']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"Acurácia: {results['accuracy_percentage']:.2f}%")
    
    # Fazer predições
    predictions = evaluator.predict(val_images_processed)
    
    # Calcular MAE geral
    overall_mae = evaluator.calculate_overall_mae(predictions, val_coords)
    print(f"MAE geral: {overall_mae:.4f}")
    
    # 6. VISUALIZAÇÃO
    print("\n6. Gerando visualizações...")
    visualizer = ResultVisualizer(config)
    
    # Plotar algumas predições
    print("Plotando primeiras 12 imagens com predições...")
    visualizer.plot_multiple_predictions(
        val_images, predictions, val_coords, num_images=12
    )
    
    # Exemplos individuais
    print("\nExemplos de predições individuais:")
    for i in [0, 5, 10]:
        visualizer.plot_prediction_vs_ground_truth(
            val_images[i], predictions[i], val_coords[i], image_idx=i
        )
        visualizer.print_coordinates(predictions[i], f"Predição {i}")
        visualizer.print_coordinates(val_coords[i], f"Ground Truth {i}")
    
    print("\n" + "=" * 50)
    print("TREINAMENTO CONCLUÍDO!")
    print(f"Modelo salvo em: {config.MODEL_SAVE_DIR}/{config.MODEL_NAME}.keras")
    print("=" * 50)

if __name__ == "__main__":
    main()