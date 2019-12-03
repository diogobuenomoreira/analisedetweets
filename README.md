# Análise de Tweets Racistas e Sexistas
Este é um trabalho para a disciplina de Deep Learning - Unicamp 2º semestre de 2019.
Neste trabalho é o treinamento de diferentes redes neurais para o problema de classificação de textos contendo discurso de ódio (racista ou sexista).


Este repositório contém os seguintes arquivos:

    baseline.py: o programa da entrega intermediária para o trabalho
    lstm.py: o programa utilizando a arquitetura LSTM
    lstm-glove.py: o programa utilizando a arquitetura LSTM utilizando Glove
    cnn.py: o programa utilizando a arquitetura CNN
    cnn-glove.py: o programa utilizando a arquitetura CNN utilizando Glove
    singularity-install.sh: script para instalação do Singularity e criação do container com o Singularity
    docker-recipe.def: receita para a criação do container
    experiments*: scripts para realizar os experimentos
    contagem.py: contagem de tweets nas bases de dados
    criar-database.py: cria base de dados a partir do NAACL_SRW_2016.csv
    script-results*: cria os gráficos e estatísticas
    pasta result*: resultados dos experimentos
    pasta data: bases de dados
    pasta graficos: gráficos dos resultados

## Execução dos experimentos
    sh ./singularity-install.sh
    sudo singularity run imagem.img
    ./experiments-lstm.sh
    ./experiments-lstm-glove.sh
    ./experiments-cnn.sh
    ./experiments-cnn-glove.sh
