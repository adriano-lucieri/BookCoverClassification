#!/bin/bash
cd /netscratch/lucieri/Github/BookCoverClassification/

bash main.sh test 30cat-NAS 30cat-SEResNet 30cat-SEResNeXt 30cat-IncResV2 30cat-ResNet152 30cat-ResNet50 30cat-VGG16 30cat-DenseNet 28cat-IncResV2 Aug Att-softmax Att-sigmoid Att-tempSM Att-saliency Att-residual Att-resStacked Att-STN Att-combinedSTN MSE GAN07 GAN10 Text-early Text-late Text-dual Text-lateSVM

bash main.sh ensemble 1 2 3