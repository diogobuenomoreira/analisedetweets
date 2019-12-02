#!/bin/bash

#INSTALAR SINGULARITY
echo "INSTALANDO DEPENDENCIAS"
DIRETORIO=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qy && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -qy build-essential \
  libssl-dev uuid-dev libgpgme11-dev libseccomp-dev pkg-config squashfs-tools

echo "INSTALANDO GOLANG"
VERSION=1.11.4
OS=linux
ARCH=amd64
export GOPATH=${HOME}/go >> ~/.bashrc
export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin >> ~/.bashrc
source ~/.bashrc
wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz && \
sudo tar -C /usr/local -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz
curl -sfL https://install.goreleaser.com/github.com/golangci/golangci-lint.sh |
  sh -s -- -b $(go env GOPATH)/bin v1.15.0

echo "INSTALANDO SINGULARITY"
mkdir -p ${GOPATH}/src/github.com/sylabs && \
  cd ${GOPATH}/src/github.com/sylabs && \
  git clone https://github.com/sylabs/singularity.git && \
  cd singularity
git checkout v3.2.1
cd ${GOPATH}/src/github.com/sylabs/singularity && \
  ./mconfig && \
  cd ./builddir && \
  make && \
  sudo make install
singularity version
cd ${DIRETORIO}

#CRIAR IMAGEM A PARTIR DA RECIPE
echo "CRIANDO IMAGEM"
sudo singularity build imagem.img docker-recipe.def

#BAIXAR DADOS
echo "BAIXANDO E EXTRAINDO DADOS"
sudo apt-get install -y unzip
wget http://nlp.stanford.edu/data/glove.6B.zip
mv glove.6B.zip data/glove.6B.zip
unzip data/glove.6B.zip -d data/
chmod +x experiments-cnn-glove.sh
chmod +x experiments-cnn.sh
chmod +x experiments-lstm-glove.sh
chmod +x experiments-lstm.sh
