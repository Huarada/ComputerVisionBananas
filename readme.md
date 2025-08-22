# 🍌 Banana Detection - Inception Network

Um projeto de **visão computacional** para detecção de bananas usando uma arquitetura **Inception Network customizada**. O modelo prevê as coordenadas do bounding box (xmin, ymin, xmax, ymax) das bananas nas imagens.

## 📋 Características

- **Arquitetura Inception**: Módulos Inception customizados para detecção de objetos
- **Data Augmentation**: Translação com ajuste automático de bounding boxes
- **Treinamento Robusto**: Callbacks para redução de learning rate e regularização L2
- **Visualização Completa**: Ferramentas para visualizar predições vs ground truth
- **Modularidade**: Código organizado em módulos reutilizáveis

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/banana-detection.git
cd banana-detection
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Organize seus dados na seguinte estrutura:
```
data/
├── bananas_train/
│   ├── images/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── label.csv
└── bananas_val/
    ├── images/
    │   ├── 0.png
    │   ├── 1.png
    │   └── ...
    └── label.csv
```

## 🏃‍♂️ Uso Rápido

### Treinamento e Avaliação Completa
```bash
python main.py
```

### Apenas Treinamento
```bash
python train.py --epochs 200 --batch_size 100
```

### Apenas Avaliação
```bash
python evaluate.py --model_path saved_models/inception_banana_detector.keras --visualize
```

## 📁 Estrutura do Projeto

```
banana-detection/
├── config/
│   └── config.py              # Configurações do projeto
├── data/
│   ├── data_loader.py         # Carregamento de dados
│   └── augmentation.py        # Data augmentation
├── models/
│   ├── inception_model.py     # Arquitetura Inception
│   └── model_utils.py         # Utilitários do modelo
├── training/
│   ├── trainer.py             # Lógica de treinamento
│   └── callbacks.py           # Callbacks personalizados
├── evaluation/
│   ├── evaluator.py           # Avaliação do modelo
│   └── visualization.py       # Visualização de resultados
├── utils/
│   └── general_utils.py       # Utilitários gerais
├── main.py                    # Script principal
├── train.py                   # Script de treinamento
├── evaluate.py                # Script de avaliação
└── saved_models/              # Modelos salvos
```

## ⚙️ Configurações

Edite `config/config.py` para personalizar:

```python
class Config:
    # Parâmetros da imagem
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    
    # Treinamento
    BATCH_SIZE = 100
    EPOCHS = 200
    LEARNING_RATE = 0.001
    
    # Data Augmentation
    NUM_AUGMENTATIONS_PER_IMAGE = 6
    MAX_TRANSLATION_X = 0.2
    MAX_TRANSLATION_Y = 0.2
```

## 🧠 Arquitetura do Modelo

### Módulo Inception
Cada módulo Inception contém 4 torres paralelas:
- **Torre 1**: Conv2D 1x1
- **Torre 2**: Conv2D 1x1 → Conv2D 3x3  
- **Torre 3**: Conv2D 1x1 → Conv2D 5x5
- **Torre 4**: MaxPooling2D 3x3 → Conv2D 1x1

### Arquitetura Completa
```
Input (32x32x3)
    ↓
Inception Module (64 filters) → Inception Module (64 filters)
    ↓
MaxPooling2D (2x2)
    ↓
Inception Module (64 filters) → Inception Module (64 filters)
    ↓
MaxPooling2D (2x2)
    ↓
Inception Module (64 filters) → Inception Module (64 filters)
    ↓
AveragePooling2D (8x8) → Flatten → Dense(4, linear)
    ↓
Output: [xmin, ymin, xmax, ymax]
```

## 🔄 Data Augmentation

O sistema aplica **translação manual** com ajuste automático dos bounding boxes:

```python
# Translação com preservação dos bounding boxes
translated_image, adjusted_bbox = manual_translation(
    image, bbox, 
    max_dx=0.2, max_dy=0.2,
    img_width=32, img_height=32
)
```

## 📊 Avaliação

### Métricas Disponíveis
- **Mean Absolute Error (MAE)**: Erro médio absoluto
- **Mean Squared Error (MSE)**: Erro quadrático médio
- **MAE por coordenada**: Análise detalhada de cada coordenada
- **Visualizações**: Comparação predição vs ground truth

### Exemplo de Saída
```
📊 RESULTADOS DA AVALIAÇÃO:
   Loss: 0.0234
   MAE: 0.1456
   Acurácia estimada: 85.44%
   MAE geral (coordenadas): 4.65 pixels
```

## 🎨 Visualização

### Grid de Múltiplas Predições
```python
# Mostrar 12 exemplos com predições (vermelho) e ground truth (verde)
visualizer.plot_multiple_predictions(images, predictions, ground_truth, num_images=12)
```

### Comparação Individual
```python
# Análise detalhada de uma imagem específica
visualizer.plot_prediction_vs_ground_truth(image, prediction, ground_truth, image_idx=0)
```

## 🛠️ Scripts Disponíveis

### 1. Treinamento Personalizado
```bash
python train.py \
    --epochs 300 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --model_name meu_modelo_bananas \
    --augmentations 8
```

### 2. Continuar Treinamento
```bash
python train.py \
    --resume_training saved_models/modelo_existente.keras \
    --epochs 100
```

### 3. Avaliação Detalhada
```bash
python evaluate.py \
    --model_path saved_models/inception_banana_detector.keras \
    --visualize \
    --detailed_analysis \
    --save_results \
    --num_examples 20
```

## 🎯 Resultados Esperados

Com os hiperparâmetros padrão, o modelo tipicamente alcança:
- **MAE**: ~0.14 (coordenadas normalizadas)
- **Acurácia**: ~85%
- **Erro por pixel**: ~4-6 pixels (em imagens 32x32)

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## 📞 Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter) - seuemail@exemplo.com

Link do Projeto: [https://github.com/seu-usuario/banana-detection](https://github.com/seu-usuario/banana-detection)

## 🙏 Agradecimentos

- [TensorFlow](https://tensorflow.org/) - Framework de deep learning
- [Inception Network](https://arxiv.org/abs/1409.4842) - Arquitetura original
- Comunidade de visão computacional