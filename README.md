# Análise de Tweets Racistas e Sexistas
Este é um trabalho para a disciplina de Deep Learning - Unicamp 2º semestre de 2019.
Neste trabalho é o treinamento de diferentes redes neurais para o problema de classificação de textos contendo discurso de ódio (racista ou sexista).


Este repositório contém os seguintes arquivos:

    baseline.py: o programa da entrega intermediária para o trabalho
    lstm.py: o programa utilizando a arquitetura LSTM
    cnn.py: o programa utilizando a arquitetura CNN
    test.csv e train.csv: base de dados
    singularity-install.sh: script para instalação do Singularity e criação do container com o Singularity
    docker-recipe.def: receita para a criação do container
    experiments: script para realizar os experimentos
    pasta result: resultados dos experimentos


## Execução dos experimentos
    ./singularity-install.sh
    sudo singularity run imagem.img
    ./experiments-lstm.sh
    ./experiments-lstm-glove.sh
    ./experiments-cnn.sh
    ./experiments-cnn-glove.sh
