"""
Script específico para avaliação do modelo
"""
import argparse
import os

from config.config import Config
Config.setup_tf_environment()

from data.data_loader import DataLoader
from evaluation.evaluator import ModelEvaluator
from evaluation.visualization import ResultVisualizer

def parse_arguments():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='Avaliar modelo de detecção de bananas')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Caminho para o modelo salvo (.keras)')
    parser.add_argument('--visualize', action='store_true',
                        help='Mostrar visualizações dos resultados')
    parser.add_argument('--num_examples', type=int, default=12,
                        help='Número de exemplos para visualizar (default: 12)')
    parser.add_argument('--save_results', action='store_true',
                        help='Salvar visualizações em arquivos')
    parser.add_argument('--detailed_analysis', action='store_true',
                        help='Análise detalhada com estatísticas adicionais')
    
    return parser.parse_args()

def evaluate_model(args):
    """
    Função principal de avaliação
    
    Args:
        args: Argumentos parseados
    """
    print("=" * 60)
    print("AVALIAÇÃO - DETECÇÃO DE BANANAS")
    print("=" * 60)
    
    # Verificar se modelo existe
    if not os.path.exists(args.model_path):
        print(f"ERRO: Modelo não encontrado em {args.model_path}")
        return
    
    # Configuração
    config = Config()
    
    # Carregar dados de validação
    print("1. Carregando dados de validação...")
    data_loader = DataLoader(config)
    val_images, val_coords = data_loader.load_validation_data()
    val_images_processed = data_loader.preprocess_images(val_images)
    
    print(f"   Imagens de validação: {len(val_images)}")
    
    # Carregar modelo
    print(f"2. Carregando modelo...")
    evaluator = ModelEvaluator(config=config)
    evaluator.load_model(args.model_path)
    
    # Avaliação básica
    print(f"3. Executando avaliação...")
    results = evaluator.evaluate_model(val_images_processed, val_coords)
    
    print(f"\n📊 RESULTADOS DA AVALIAÇÃO:")
    print(f"   Loss: {results['loss']:.4f}")
    print(f"   MAE: {results['mae']:.4f}")
    print(f"   Acurácia estimada: {results['accuracy_percentage']:.2f}%")
    print(f"   Erro estimado: {results['error_percentage']:.2f}%")
    
    # Predições
    print(f"\n4. Gerando predições...")
    predictions = evaluator.predict(val_images_processed)
    
    # MAE geral
    overall_mae = evaluator.calculate_overall_mae(predictions, val_coords)
    print(f"   MAE geral (coordenadas): {overall_mae:.4f} pixels")
    
    # Análise detalhada
    if args.detailed_analysis:
        print(f"\n5. Análise detalhada...")
        analyze_predictions_detailed(predictions, val_coords, config)
    
    # Visualizações
    if args.visualize:
        print(f"\n6. Gerando visualizações...")
        visualizer = ResultVisualizer(config)
        
        # Grade de múltiplas predições
        print(f"   Plotando {args.num_examples} exemplos...")
        visualizer.plot_multiple_predictions(
            val_images, predictions, val_coords, 
            num_images=min(args.num_examples, len(val_images))
        )
        
        if args.save_results:
            visualizer.save_visualization("predictions_grid.png")
        
        # Exemplos individuais detalhados
        print("   Mostrando exemplos individuais...")
        examples_to_show = min(5, len(val_images))
        
        for i in range(examples_to_show):
            print(f"\n   Exemplo {i+1}:")
            visualizer.plot_prediction_vs_ground_truth(
                val_images[i], predictions[i], val_coords[i], image_idx=i
            )
            
            if args.save_results:
                visualizer.save_visualization(f"prediction_example_{i}.png")
            
            # Imprimir coordenadas
            visualizer.print_coordinates(predictions[i], "Predição", i)
            visualizer.print_coordinates(val_coords[i], "Ground Truth", i)
            
            # Calcular erro por coordenada
            pred_coords = evaluator.get_prediction_coordinates(predictions, i)
            gt_coords = evaluator.get_ground_truth_coordinates(val_coords, i)
            
            print(f"   Erros absolutos:")
            print(f"     xmin: {abs(pred_coords['xmin'] - gt_coords['xmin']):.2f}")
            print(f"     ymin: {abs(pred_coords['ymin'] - gt_coords['ymin']):.2f}")
            print(f"     xmax: {abs(pred_coords['xmax'] - gt_coords['xmax']):.2f}")
            print(f"     ymax: {abs(pred_coords['ymax'] - gt_coords['ymax']):.2f}")
    
    print(f"\n" + "=" * 60)
    print("AVALIAÇÃO CONCLUÍDA!")
    if args.save_results:
        print("Visualizações salvas na pasta atual.")
    print("=" * 60)

def analyze_predictions_detailed(predictions, ground_truth, config):
    """
    Análise detalhada das predições
    
    Args:
        predictions: Predições do modelo
        ground_truth: Valores verdadeiros
        config: Configurações
    """
    import numpy as np
    
    num_images = len(predictions)
    
    # Calcular erros por coordenada
    errors_xmin = []
    errors_ymin = []
    errors_xmax = []
    errors_ymax = []
    
    for i in range(num_images):
        # Desnormalizar coordenadas
        pred_xmin = predictions[i, 0] * config.IMAGE_WIDTH
        pred_ymin = predictions[i, 1] * config.IMAGE_HEIGHT
        pred_xmax = predictions[i, 2] * config.IMAGE_WIDTH
        pred_ymax = predictions[i, 3] * config.IMAGE_HEIGHT
        
        gt_xmin = ground_truth[i, 0] * config.IMAGE_WIDTH
        gt_ymin = ground_truth[i, 1] * config.IMAGE_HEIGHT
        gt_xmax = ground_truth[i, 2] * config.IMAGE_WIDTH
        gt_ymax = ground_truth[i, 3] * config.IMAGE_HEIGHT
        
        # Calcular erros absolutos
        errors_xmin.append(abs(pred_xmin - gt_xmin))
        errors_ymin.append(abs(pred_ymin - gt_ymin))
        errors_xmax.append(abs(pred_xmax - gt_xmax))
        errors_ymax.append(abs(pred_ymax - gt_ymax))
    
    # Estatísticas
    print(f"\n📈 ESTATÍSTICAS DETALHADAS:")
    print(f"   Erro médio por coordenada:")
    print(f"     xmin: {np.mean(errors_xmin):.2f} ± {np.std(errors_xmin):.2f}")
    print(f"     ymin: {np.mean(errors_ymin):.2f} ± {np.std(errors_ymin):.2f}")
    print(f"     xmax: {np.mean(errors_xmax):.2f} ± {np.std(errors_xmax):.2f}")
    print(f"     ymax: {np.mean(errors_ymax):.2f} ± {np.std(errors_ymax):.2f}")
    
    print(f"\n   Erro máximo por coordenada:")
    print(f"     xmin: {np.max(errors_xmin):.2f}")
    print(f"     ymin: {np.max(errors_ymin):.2f}")
    print(f"     xmax: {np.max(errors_xmax):.2f}")
    print(f"     ymax: {np.max(errors_ymax):.2f}")
    
    print(f"\n   Erro mínimo por coordenada:")
    print(f"     xmin: {np.min(errors_xmin):.2f}")
    print(f"     ymin: {np.min(errors_ymin):.2f}")
    print(f"     xmax: {np.min(errors_xmax):.2f}")
    print(f"     ymax: {np.min(errors_ymax):.2f}")

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_model(args)