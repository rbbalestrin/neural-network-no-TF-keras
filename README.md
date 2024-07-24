Claro! Aqui está o `README.md` com as expressões matemáticas formatadas usando LaTeX:

---

# Rede Neural MNIST do Zero

Este projeto implementa uma rede neural simples do zero para classificar o conjunto de dados MNIST usando NumPy. A rede é treinada usando gradiente descendente e avaliada em um conjunto de testes. Este README explica o pré-processamento de dados, a arquitetura da rede neural, a propagação para frente e para trás, e a atualização dos parâmetros.

## Índice

- [Rede Neural MNIST do Zero](#rede-neural-mnist-do-zero)
  - [Índice](#índice)
  - [Conjunto de Dados](#conjunto-de-dados)
  - [Pré-processamento de Dados](#pré-processamento-de-dados)
  - [Arquitetura da Rede Neural](#arquitetura-da-rede-neural)
  - [Propagação para Frente](#propagação-para-frente)
    - [Equações:](#equações)
    - [Funções de Ativação:](#funções-de-ativação)
  - [Propagação para Trás](#propagação-para-trás)
    - [Equações:](#equações-1)
    - [Funções de Gradiente:](#funções-de-gradiente)
  - [Atualização de Parâmetros](#atualização-de-parâmetros)
    - [Gradiente Descendente:](#gradiente-descendente)
  - [Treinamento e Avaliação](#treinamento-e-avaliação)
    - [Treinamento:](#treinamento)
    - [Avaliação:](#avaliação)
  - [Executando o Código](#executando-o-código)

## Conjunto de Dados

O conjunto de dados MNIST consiste em imagens em tons de cinza de 28x28 pixels de dígitos manuscritos (0-9). O conjunto de dados é dividido em conjuntos de treinamento e teste:

- `train-images.idx3-ubyte`: Imagens de treinamento
- `train-labels.idx1-ubyte`: Rótulos de treinamento
- `t10k-images.idx3-ubyte`: Imagens de teste
- `t10k-labels.idx1-ubyte`: Rótulos de teste

## Pré-processamento de Dados

1. **Carregar os Dados**:

   - Use `load_mnist_images` para carregar os dados das imagens e remodelá-los para (número de imagens, 28, 28).
   - Use `load_mnist_labels` para carregar os dados dos rótulos.

2. **Achatar e Normalizar**:

   - Achate cada imagem de 28x28 em um vetor de 784 dimensões.
   - Normalize os valores dos pixels para ficarem entre 0 e 1.

3. **Embaralhar e Dividir**:
   - Embaralhe os dados de treinamento.
   - Divida os dados em conjuntos de treinamento e desenvolvimento.

## Arquitetura da Rede Neural

A rede neural consiste em:

- **Camada de Entrada**: 784 neurônios (um para cada pixel da imagem 28x28).
- **Camada Oculta**: 128 neurônios com ativação ReLU.
- **Camada de Saída**: 10 neurônios com ativação softmax (um para cada classe de dígitos).

## Propagação para Frente

### Equações:

1. **Camada Oculta**:

   - Quando $W1 \ne 0$, a equação é:
     $$
     Z1 = W1 \cdot X + b1
     $$
   - A ativação é:
     $$
     A1 = \text{ReLU}(Z1)
     $$

2. **Camada de Saída**:
   - Quando $W2 \ne 0$, a equação é:
     $$
     Z2 = W2 \cdot A1 + b2
     $$
   - A ativação é:
     $$
     A2 = \text{softmax}(Z2)
     $$

### Funções de Ativação:

- **ReLU (Rectified Linear Unit)**:
  - $$
    \text{ReLU}(z) = \max(0, z)
    $$
- **Softmax**:
  - $$
    \text{softmax}(z) = \frac{\exp(z)}{\sum \exp(z)}
    $$

## Propagação para Trás

### Equações:

1. **Camada de Saída**:

   - O gradiente é:
     $$
     dZ2 = A2 - Y_{\text{one-hot}}
     $$
   - A derivada do peso é:
     $$
     dW2 = \frac{1}{m} dZ2 \cdot A1^T
     $$
   - A derivada do viés é:
     $$
     db2 = \frac{1}{m} \sum dZ2
     $$

2. **Camada Oculta**:
   - O gradiente é:
     $$
     dZ1 = W2^T \cdot dZ2 \cdot \text{ReLU}'(Z1)
     $$
   - A derivada do peso é:
     $$
     dW1 = \frac{1}{m} dZ1 \cdot X^T
     $$
   - A derivada do viés é:
     $$
     db1 = \frac{1}{m} \sum dZ1
     $$

### Funções de Gradiente:

- **Derivada da ReLU**:
  - $$
    \text{ReLU}'(z) = 1 \text{ se } z > 0 \text{ senão } 0
    $$

## Atualização de Parâmetros

### Gradiente Descendente:

- Atualize os parâmetros usando os gradientes calculados durante a propagação para trás:
  - W1:
    $$
    W1 = W1 - \alpha \cdot dW1
    $$
  - b1
    $$
    b1 = b1 - \alpha \cdot db1
    $$
  - W2
    $$
    W2 = W2 - \alpha \cdot dW2
    $$
  - db1
    $$
    b2 = b2 - \alpha \cdot db2
    $$

## Treinamento e Avaliação

### Treinamento:

- Treine a rede neural usando gradiente descendente por um número especificado de iterações.

### Avaliação:

- Avalie a precisão do modelo no conjunto de testes.

## Executando o Código

Para executar o código, siga estes passos:

1. **Certifique-se de que você tem os arquivos do conjunto de dados MNIST** (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`).
2. **Execute o script** substituindo `filename` pelo caminho dos seus arquivos.

---

Este README fornece uma visão geral abrangente do código, incluindo explicações dos cálculos da rede neural e instruções para executar o código.
