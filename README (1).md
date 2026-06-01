# 🎌 AnimeFace GAN — Geração de Rostos Anime com DCGAN

Este projeto implementa uma **DCGAN (Deep Convolutional Generative Adversarial Network)** treinada no [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) para gerar rostos de personagens anime sinteticamente.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Estrutura do Notebook](#estrutura-do-notebook)
- [Hiperparâmetros](#hiperparâmetros)
- [Como Executar](#como-executar)
- [Resultados](#resultados)
- [Dependências](#dependências)

---

## Visão Geral

Uma GAN é composta por dois modelos que competem entre si:

- **Gerador (G):** aprende a criar imagens falsas convincentes a partir de ruído aleatório.
- **Discriminador (D):** aprende a distinguir imagens reais das falsas geradas por G.

O treinamento é um jogo de soma zero: G tenta enganar D, enquanto D tenta não ser enganado. Ao final, G é capaz de gerar imagens indistinguíveis das reais.

---

## Arquitetura

```
Gerador (nz=100 → 64x64 RGB)
  ConvTranspose2d: 100 → 512 → 256 → 128 → 64 → 3
  Ativações: ReLU + Tanh final

Discriminador (64x64 RGB → escalar)
  Conv2d: 3 → 64 → 128 → 256 → 512 → 1
  Ativações: LeakyReLU(0.2)
```

---

## Estrutura do Notebook

### Célula 1 — Imports e Configuração Inicial

```python
import torch, torchvision, matplotlib, ...
```

Importa todas as bibliotecas necessárias: PyTorch (redes neurais e treinamento), Torchvision (transformações e utilitários de imagem), Matplotlib (visualização), PIL (leitura de imagens) e os módulos padrão do Python (`os`, `random`).

---

### Célula 2 — Hiperparâmetros e Dataset

Define a semente aleatória (`manualSeed = 999`) para reprodutibilidade, e configura os hiperparâmetros principais do modelo:

| Parâmetro | Valor | Descrição |
|---|---|---|
| `batch_size` | 128 | Imagens por lote de treinamento |
| `image_size` | 64 | Resolução das imagens (64×64 px) |
| `nc` | 3 | Canais de cor (RGB) |
| `nz` | 100 | Dimensão do vetor de ruído (latent space) |
| `ngf` / `ndf` | 64 | Filtros base do Gerador e Discriminador |
| `num_epochs` | 100 | Épocas de treinamento |
| `beta1` | 0.5 | Parâmetro do otimizador Adam |

Em seguida, define um **pipeline de transformações** que redimensiona cada imagem para 64×64, converte para tensor e normaliza os valores para o intervalo `[-1, 1]`.

Por fim, implementa a classe `AnimeDataset` — um dataset customizado que lê as imagens da pasta especificada e aplica as transformações — e cria o `DataLoader` para iterar em lotes durante o treinamento.

---

### Célula 3 — Inicialização de Pesos

```python
def weights_init(m): ...
```

Função auxiliar aplicada a todas as camadas do Gerador e do Discriminador após a criação. Segue a recomendação do paper original da DCGAN:

- **Camadas convolucionais:** pesos inicializados com distribuição normal de média `0` e desvio padrão `0.02`.
- **Camadas de BatchNorm:** pesos inicializados com média `1.0` e desvio `0.02`; bias zerado.

Uma boa inicialização é essencial para a estabilidade do treinamento de GANs.

---

### Célula 4 — Gerador

Define a arquitetura do **Gerador**, que transforma um vetor de ruído aleatório (de dimensão `nz=100`) em uma imagem colorida de `64×64` pixels.

A rede usa camadas `ConvTranspose2d` (convolução transposta) para aumentar progressivamente a resolução espacial:

```
Entrada: ruído (100, 1, 1)
  → (512, 4, 4)
  → (256, 8, 8)
  → (128, 16, 16)
  → (64, 32, 32)
  → (3, 64, 64)   ← imagem RGB final
```

Cada bloco intermediário usa **BatchNorm + ReLU**. A camada final usa **Tanh**, que mantém os valores no intervalo `[-1, 1]` (compatível com a normalização aplicada no dataset).

Ao final, o modelo é enviado para GPU/CPU e os pesos são inicializados com `weights_init`.

---

### Célula 5 — Discriminador

Define a arquitetura do **Discriminador**, que recebe uma imagem `64×64` e retorna um único valor indicando a probabilidade de ela ser real.

A rede usa camadas `Conv2d` para reduzir progressivamente a resolução espacial:

```
Entrada: imagem (3, 64, 64)
  → (64, 32, 32)
  → (128, 16, 16)
  → (256, 8, 8)
  → (512, 4, 4)
  → escalar (1, 1, 1)   ← score real/falso
```

Cada bloco usa **BatchNorm + LeakyReLU(0.2)**, que permite gradientes pequenos para valores negativos — importante para estabilizar o treinamento do Discriminador. Não há BatchNorm na primeira camada (recomendação do paper DCGAN).

---

### Célula 6 — Otimizadores e Função de Perda

Configura os componentes de treinamento:

- **Função de perda:** `BCEWithLogitsLoss` — combina Sigmoid + Binary Cross-Entropy em uma operação numericamente estável. Adequada para classificação binária (real vs. falso).
- **`fixed_noise`:** vetor de ruído fixo usado para monitorar visualmente a evolução do Gerador ao longo do treinamento.
- **Label smoothing:** em vez de usar `1.0` e `0.0` como rótulos, usa `0.9` (real) e `0.1` (falso) para suavizar o treinamento e evitar que o Discriminador fique muito confiante.
- **Otimizador do Discriminador:** Adam com `lr=0.00005` (mais lento para equilibrar com o Gerador).
- **Otimizador do Gerador:** Adam com `lr=0.0002`.

---

### Célula 7 — Inicialização do Histórico

Cria as listas `G_losses` e `D_losses` para armazenar as perdas a cada iteração, e `img_list` para salvar amostras visuais do progresso. O contador `iters` é inicializado para rastrear o número total de iterações.

---

### Célula 8 — Loop de Treinamento

O núcleo do projeto. Para cada época e cada lote, executa três etapas:

**Etapa 1 — Treinar o Discriminador:**
1. Passa imagens **reais** pelo Discriminador e calcula a perda (D deve classificá-las como reais → label `0.9`).
2. Gera imagens **falsas** com o Gerador e passa pelo Discriminador (D deve classificá-las como falsas → label `0.1`).
3. Soma as duas perdas e atualiza os pesos do Discriminador.

**Etapa 2 — Treinar o Gerador (1ª vez):**
1. Passa as mesmas imagens falsas pelo Discriminador novamente.
2. Calcula a perda do Gerador (G quer que D classifique as imagens falsas como reais → label `0.9`).
3. Atualiza os pesos do Gerador.

**Etapa 3 — Segundo Treino do Gerador:**
- Gera um novo lote de ruído e repete o processo de treinamento do Gerador. Essa etapa extra foi adicionada para dar mais "força" ao Gerador em relação ao Discriminador, evitando o colapso de modo (*mode collapse*).

**Monitoramento:**
- A cada 15 iterações, imprime as métricas: `Loss_D`, `Loss_G`, `D(x)` (confiança do D em imagens reais) e `D(G(z))` (confiança do D nas imagens falsas antes e depois de atualizar G).
- A cada 5 épocas, exibe uma grade com as imagens geradas a partir do `fixed_noise`.

---

### Célula 9 — Seleção e Salvamento das Melhores Imagens

Após o treinamento, avalia a qualidade das imagens geradas:

1. Gera **200 imagens** a partir de ruído aleatório.
2. Passa todas pelo Discriminador para obter um score de realismo.
3. Seleciona as **Top 10** imagens com maior score (`torch.topk`).
4. Salva cada uma em `best_images/best_N.png`.
5. Exibe as 10 melhores em uma grade com seus scores.

---

### Célula 10 — Teste Cego do Discriminador

Avalia a qualidade do Discriminador em distinguir imagens reais de falsas em um cenário controlado:

1. Coleta **10 imagens reais** do DataLoader e gera **10 imagens falsas** com o Gerador.
2. Embaralha aleatoriamente as 20 imagens.
3. Para cada imagem, o Discriminador emite um score: ≥ 0.5 → predição REAL, < 0.5 → predição FAKE.
4. Compara a predição com o rótulo verdadeiro e calcula a **acurácia final**.
5. Exibe todas as 20 imagens com predição, verdade e score.

> Quanto mais próxima de 50%, mais equilibrada está a competição entre Gerador e Discriminador — o ideal em uma GAN bem treinada.

---

## Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Épocas | 100 |
| Batch size | 128 |
| Tamanho da imagem | 64×64 |
| Dimensão do ruído (nz) | 100 |
| LR Gerador | 0.0002 |
| LR Discriminador | 0.00005 |
| Beta1 (Adam) | 0.5 |
| Label smoothing (real) | 0.9 |
| Label smoothing (fake) | 0.1 |

---

## Como Executar

Este projeto foi desenvolvido para rodar no **Kaggle** com GPU habilitada.

1. Faça o download do dataset [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) e adicione ao seu notebook Kaggle.
2. Certifique-se de que o caminho do dataset está correto:
   ```python
   dataroot = "/kaggle/input/datasets/splcher/animefacedataset/images"
   ```
3. Habilite o acelerador **GPU** nas configurações do notebook Kaggle.
4. Execute todas as células em ordem.

---

## Resultados

Ao final do treinamento, o modelo gera rostos anime sintéticos com resolução de 64×64 pixels. As melhores imagens são salvas na pasta `best_images/`.

---

## Dependências

```
torch
torchvision
numpy
matplotlib
Pillow
```

Todas as dependências já estão disponíveis no ambiente padrão do Kaggle.

---

## Referências

- [Radford et al., 2015 — Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Anime Face Dataset — Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset)
