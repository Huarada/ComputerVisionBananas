"""
Script específico para treinamento do modelo
"""
import argparse
import os
import tensorflow as tf

from config.config import Config
Config.setup_tf_environment()

from data.data_loader import DataLoader
from data.augmentation import DataAugmentation
from models.inception_model import InceptionBananaDetector
from training.trainer import ModelTrainer

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Treinar modelo de detecção de bananas')
    
    parser.add_argument('--epochs', type=int, default=200,
                        help='Número de épocas para treinamento (default: 200)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Tamanho do batch (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Taxa de aprendizado (default: 0.001)')
    parser.add_argument('--model_name', type=str, default='inception_banana_detector',
                        help='Nome do modelo para salvamento')
    parser.add_argument('--resume_training', type=str, default=None,
                        help='Caminho para modelo existente para continuar treinamento')
    parser.add_argument('--augmentations', type=int, default=6,
                        help='Número de augmentações por imagem (default: 6)')
    
    return parser.parse_args()

def train_model(args):
    """
    Função principal de treinamento
    
    Args:
        args: Argumentos parseados
    """
    print("=" * 60)
    print("TREINAMENTO - DETECÇÃO DE BANANAS")
    print("=" * 60)
    
    # Configuração com argumentos personalizados
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.MODEL_NAME = args.model_name
    config.NUM_AUGMENTATIONS_PER_IMAGE = args.augmentations
    
    print(f"Configurações de treinamento:")
    print(f"  Épocas: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Augmentações por imagem: {config.NUM_AUGMENTATIONS_PER_IMAGE}")
    print(f"  Nome do modelo: {config.MODEL_NAME}")
    
    # Carregar dados
    print(f"\n1. Carregando dados...")
    data_loader = DataLoader(config)
    
    train_images, train_coords = data_loader.load_training_data()
    val_images, val_coords = data_loader.load_validation_data()
    val_images_processed = data_loader.preprocess_images(val_images)
    
    print(f"   Imagens de treinamento: {len(train_images)}")
    print(f"   Imagens de validação: {len(val_images)}")
    
    # Data augmentation
    print(f"\n2. Aplicando data augmentation...")
    augmentation = DataAugmentation(config)
    augmented_images, augmented_coords = augmentation.augment_dataset(train_images, train_coords)
    augmented_images_processed = data_loader.preprocess_images(augmented_images)
    
    print(f"   Total após augmentation: {len(augmented_images)} imagens")
    
    # Criar dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (augmented_images_processed, augmented_coords)
    ).batch(config.BATCH_SIZE)
    
    # Construir ou carregar modelo
    print(f"\n3. Preparando modelo...")
    if args.resume_training:
        print(f"   Carregando modelo existente: {args.resume_training}")
        from tensorflow.keras.models import load_model
        model = load_model(args.resume_training)
    else:
        print("   Construindo novo modelo Inception...")
        model_builder = InceptionBananaDetector(config)
        model = model_builder.create_model()
    
    # Treinar
    print(f"\n4. Iniciando treinamento...")
    trainer = ModelTrainer(model, config)
    
    # Salvar arquitetura
    trainer.plot_model_architecture()
    trainer.print_model_summary()
    
    # Executar treinamento
    history = trainer.train(train_dataset, (val_images_processed, val_coords))
    
    # Plotar resultados
    trainer.plot_training_history(history)
    
    # Salvar modelo
    trainer.save_model()
    
    # Avaliação final
    print(f"\n5. Avaliação final...")
    final_loss, final_mae = model.evaluate(val_images_processed, val_coords, verbose=0)
    print(f"   Loss final: {final_loss:.4f}")
    print(f"   MAE final: {final_mae:.4f}")
    print(f"   Acurácia estimada: {100 * (1 - final_mae):.2f}%")
    
    print(f"\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO!")
    print(f"Modelo salvo em: {config.MODEL_SAVE_DIR}/{config.MODEL_NAME}.keras")
    print("=" * 60)

if __name__ == "__main__":
    args = parse_arguments()
    train_model(args)