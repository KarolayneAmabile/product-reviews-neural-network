# Previsão da nota de avaliações de produtos com uma Rede Neural MLP feita do zero

Projeto de **redes neurais** que prevê a **nota (1 a 5 estrelas)** de uma avaliação de produto a partir do texto escrito
pelo cliente. A base é difícil: tem 5 classes desbalanceadas, e notas vizinhas, como 4 e 5 estrelas, usam palavras muito
parecidas. O centro do trabalho é uma **MLP com duas camadas ocultas implementada do zero em NumPy**, comparada com a
mesma rede em **Keras**.

## O que está sendo feito

Tudo está no notebook [Projeto3_MLP_B2W.ipynb](Projeto3_MLP_B2W.ipynb):

1. **Estudo introdutório e preparação:** distribuição das notas, tamanho dos textos, remoção de nulos e duplicados,
   amostra estratificada de 20 mil avaliações (80% desenvolvimento, 20% teste) e TF-IDF com 3.000 termos.
2. **MLP do zero:** forward pass, **backpropagation pela regra da cadeia** (derivadas dos pesos e dos bias), softmax
   com entropia cruzada e gradiente descendente por mini-batch. As derivadas são conferidas com a derivada numérica.
3. **Hiperparâmetros com 3-fold cross-validation:** ativação ReLU × Sigmoid, regularização L2 × sem regularização e
   três tamanhos de camada, totalizando 12 configurações.
4. **Curvas de aprendizado:** análise de overfitting, underfitting, viés e variância.
5. **Comparação com Keras:** a mesma rede, com os mesmos parâmetros.
6. **Mapas de calor da variação acumulada dos pesos (ReLU × Sigmoid)** e análise da saturação.
7. **Ponto extra:** embedding **Skip-gram** treinado com a própria implementação (uma camada oculta a menos),
   concatenado ao TF-IDF.

**Principal resultado:** a melhor configuração foi ReLU, sem regularização, com (16, 8) neurônios: acurácia de 0,553 e
F1 macro de 0,479 no teste.

## Como rodar

**No Google Colab:** abra o notebook e execute todas as células. A base é baixada direto do GitHub da B2W, e o
TensorFlow/Keras já vem instalado no Colab.

**Localmente:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib scipy tensorflow jupyter
jupyter notebook Projeto3_MLP_B2W.ipynb
```

A execução completa leva cerca de 7 minutos em CPU. A maior parte do tempo é a validação cruzada das 12 configurações.

## Dataset

**B2W-Reviews01**: 132.373 avaliações de produtos de clientes da Americanas (B2W Digital), em português, com texto,
nota de 1 a 5 e outras informações do produto e do cliente.

- Repositório original: <https://github.com/americanas-tech/b2w-reviews01>
- No Hugging Face: <https://huggingface.co/datasets/fgops05/b2w-reviews01> (licença CC BY 4.0)

Créditos da base à Americanas Tech / B2W Digital.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `Projeto3_MLP_B2W.ipynb` | Notebook do projeto |
| `enunciado.txt` | Enunciado do projeto |

## Créditos

Trabalho desenvolvido para a disciplina de **Inteligência Artificial** do curso de Bacharelado em Ciência da Computação
do Instituto Federal de Goiás (IFG), ministrada pelo
[Prof. Dr. Daniel Xavier de Sousa](https://www.escavador.com/sobre/3446148/daniel-xavier-de-sousa).

Feito pelas alunas **Karolayne Amábile Brito Borges** e **[Ana Laura Machado Pereira](https://github.com/aanalaura)**.

## Licença

O código deste projeto está sob a [licença MIT](LICENSE). A base de dados segue a licença e os termos de uso da sua fonte original.
