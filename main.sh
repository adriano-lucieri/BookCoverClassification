#!/bin/bash
cd /netscratch/lucieri/Github/BookCoverClassification/
export PATH=/netscratch/lucieri/venv/anaconda3/envs/BookCover27/bin/:$PATH

declare -a params=()
declare -i cnt

cnt=0

while (( "$#" )); do
    #echo $#
    case "$1" in
        --)
            shift
            break
            ;;
        -*|--*=) # unsupported flags
            echo "Error: Unsupported flag $1" >&2
            exit 1
            ;;
        *) # preserve positional arguments
            params[cnt]=$1
            # PARAMS="$PARAMS $1"
            shift
            cnt+=1
            ;;
    esac
done

mode="${params[0]}"
experiments=("${params[@]:1}")

while (( "${#experiments[@]}" )); do

    echo Running experiment: ${experiments[0]} $mode!

    if [ $mode = train ] || [ $mode = test ] ; then

        if [ "${experiments[0]}" = 30cat-NAS ] ; then

            if [ $mode = train ] ; then
                python trainer.py -m NAS -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-NASNet --modelName 30cat-NASNet \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m NAS -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-NASNet --modelName 30cat-NASNet \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi
        elif [ "${experiments[0]}" = 30cat-SEResNet ] ; then

            if [ $mode = train ] ; then
                python trainer.py -m SEResNet -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-SEResNet --modelName 30cat-SEResNet \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m SEResNet -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-SEResNet --modelName 30cat-SEResNet \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi

        elif [ "${experiments[0]}" = 30cat-SEResNeXt ] ; then

            if [ $mode = train ] ; then
                python trainer.py -m SEResNeXt -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-SEResNeXt --modelName 30cat-SEResNeXt \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m SEResNeXt -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-SEResNeXt --modelName 30cat-SEResNeXt \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi

        elif [ "${experiments[0]}" = 30cat-IncResV2 ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-IncResV2 --modelName 30cat-IncResV2 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-IncResV2 --modelName 30cat-IncResV2 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi

        elif [ "${experiments[0]}" = 30cat-ResNet152 ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m ResNet -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-ResNet152 --modelName 30cat-ResNet152 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m ResNet -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-ResNet152 --modelName 30cat-ResNet152 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi

        elif [ "${experiments[0]}" = 30cat-ResNet50 ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m ResNet50 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-ResNet50 --modelName 30cat-ResNet50 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m ResNet50 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-ResNet50 --modelName 30cat-ResNet50 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi
        elif [ "${experiments[0]}" = 30cat-VGG16 ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m VGG16 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-VGG16 --modelName 30cat-VGG16 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m VGG16 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-VGG16 --modelName 30cat-VGG16 \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi

        elif [ "${experiments[0]}" = 30cat-DenseNet ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m DenseNet -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-DenseNet --modelName 30cat-DenseNet \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m DenseNet -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/30cat-DenseNet --modelName 30cat-DenseNet \
                --trainDataFile ./Data/title30cat-labels-train.txt \
                --valDataFile ./Data/title30cat-labels-test.txt \
                --testDataFile ./Data/title30cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title30cat-classes.csv
            fi

        elif [ "${experiments[0]}" = 28cat-IncResV2 ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2 --modelName 28cat-IncResV2 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2 --modelName 28cat-IncResV2 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            fi

        elif [ "${experiments[0]}" = Aug ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Augmentation --modelName 28cat-IncResV2-Augmentation \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --inputAugmentation
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Augmentation --modelName 28cat-IncResV2-Augmentation \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --inputAugmentation
            fi

        elif [ "${experiments[0]}" = Att-softmax ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Softmax --modelName 28cat-IncResV2-Attention-Softmax \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType softmax
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Softmax --modelName 28cat-IncResV2-Attention-Softmax \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType softmax
            fi

        elif [ "${experiments[0]}" = Att-sigmoid ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Sigmoid --modelName 28cat-IncResV2-Attention-Sigmoid \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType sigmoid
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Sigmoid --modelName 28cat-IncResV2-Attention-Sigmoid \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType sigmoid
            fi

        elif [ "${experiments[0]}" = Att-tempSM ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-TempSM --modelName 28cat-IncResV2-Attention-TempSM \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType tempSM
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-TempSM --modelName 28cat-IncResV2-Attention-TempSM \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType tempSM
            fi

        elif [ "${experiments[0]}" = Att-saliency ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Saliency --modelName 28cat-IncResV2-Attention-Saliency \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType saliency
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Saliency --modelName 28cat-IncResV2-Attention-Saliency \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType saliency
            fi

        elif [ "${experiments[0]}" = Att-residual ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Residual --modelName 28cat-IncResV2-Attention-Residual \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType residual
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Residual --modelName 28cat-IncResV2-Attention-Residual \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType residual
            fi

        elif [ "${experiments[0]}" = Att-resStacked ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-ResidualStacked --modelName 28cat-IncResV2-Attention-ResidualStacked \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType resStack
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-ResidualStacked --modelName 28cat-IncResV2-Attention-ResidualStacked \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType resStack
            fi

        elif [ "${experiments[0]}" = Att-STN ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-STN --modelName 28cat-IncResV2-Attention-STN \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType STN
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-STN --modelName 28cat-IncResV2-Attention-STN \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType STN
            fi

        elif [ "${experiments[0]}" = Att-combinedSTN ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-CombinedSTN --modelName 28cat-IncResV2-Attention-CombinedSTN \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType combSTN
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-CombinedSTN --modelName 28cat-IncResV2-Attention-CombinedSTN \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --attentionType combSTN
            fi

        elif [ "${experiments[0]}" = MSE ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction mse --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-MSE --modelName 28cat-IncResV2-MSE \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction mse --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-MSE --modelName 28cat-IncResV2-MSE \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            fi

        elif [ "${experiments[0]}" = GAN07 ] ; then
            if [ $mode = train ] ; then
                echo Models allows only inference
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 7 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-GAN07 --modelName 28cat-IncResV2-GAN07 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            fi

        elif [ "${experiments[0]}" = GAN10 ] ; then
            if [ $mode = train ] ; then
                echo Models allows only inference
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-GAN10 --modelName 28cat-IncResV2-GAN10 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            fi

        elif [ "${experiments[0]}" = Text-early ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-Early --modelName 28cat-IncResV2-Text-Early \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion early \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-Early --modelName 28cat-IncResV2-Text-Early \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion early \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100
            fi

        elif [ "${experiments[0]}" = Text-late ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-Late --modelName 28cat-IncResV2-Text-Late \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion late \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-Late --modelName 28cat-IncResV2-Text-Late \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion late \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100
            fi

        elif [ "${experiments[0]}" = Text-dual ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-Dual --modelName 28cat-IncResV2-Text-Dual \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion dual \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-Dual --modelName 28cat-IncResV2-Text-Dual \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion dual \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100
            fi

        elif [ "${experiments[0]}" = Text-lateSVM ] ; then
            if [ $mode = train ] ; then
                python trainer.py -m IncResV2 -t -s -v --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-LateSVM --modelName 28cat-IncResV2-Text-LateSVM \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion late \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100 \
                --trainSVM
            elif [ $mode = test ] ; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Text-LateSVM --modelName 28cat-IncResV2-Text-LateSVM \
                --trainDataFile ./Data/textEmbed28catPP-labels-train.txt \
                --valDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --testDataFile ./Data/textEmbed28catPP-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/textEmbed28cat-classes.csv \
                --useMultimodal \
                --multimodalFusion late \
                --modelFasttext ./FastText/model.bin \
                --embeddingVector 100 \
                --trainSVM
            fi
        else
            echo ERROR: Unknown experiment flag
            break
        fi

    elif [ $mode = ensemble ] ; then

        FILE=./Predictions/28cat-IncResV2-MSE.csv
        if ! test -f "$FILE"; then
            python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
            --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction mse --batchSize 20 --saveStep 1000 \
            --displayStep 1000 --modelDir ./Models/28cat-IncResV2-MSE --modelName 28cat-IncResV2-MSE \
            --trainDataFile ./Data/title28cat-labels-train.txt \
            --valDataFile ./Data/title28cat-labels-test.txt \
            --testDataFile ./Data/title28cat-labels-test.txt \
            --imageBaseDir ./Data/images/ \
            --classesFile ./Data/title28cat-classes.csv 
        fi

        FILE=./Predictions/28cat-IncResV2-Attention-TempSM.csv
        if ! test -f "$FILE"; then
            python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
            --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
            --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-TempSM --modelName 28cat-IncResV2-Attention-TempSM \
            --trainDataFile ./Data/title28cat-labels-train.txt \
            --valDataFile ./Data/title28cat-labels-test.txt \
            --testDataFile ./Data/title28cat-labels-test.txt \
            --imageBaseDir ./Data/images/ \
            --classesFile ./Data/title28cat-classes.csv \
            --attentionType tempSM 
        fi

        FILE=./Predictions/28cat-IncResV2-Attention-Saliency.csv
        if ! test -f "$FILE"; then
            python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
            --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
            --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Saliency --modelName 28cat-IncResV2-Attention-Saliency \
            --trainDataFile ./Data/title28cat-labels-train.txt \
            --valDataFile ./Data/title28cat-labels-test.txt \
            --testDataFile ./Data/title28cat-labels-test.txt \
            --imageBaseDir ./Data/images/ \
            --classesFile ./Data/title28cat-classes.csv \
            --attentionType saliency 
        fi

        FILE=./Predictions/28cat-IncResV2-Attention-Residual.csv
        if ! test -f "$FILE"; then
            python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
            --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
            --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Attention-Residual --modelName 28cat-IncResV2-Attention-Residual \
            --trainDataFile ./Data/title28cat-labels-train.txt \
            --valDataFile ./Data/title28cat-labels-test.txt \
            --testDataFile ./Data/title28cat-labels-test.txt \
            --imageBaseDir ./Data/images/ \
            --classesFile ./Data/title28cat-classes.csv \
            --attentionType residual 
        fi

        if [ "${experiments[0]}" = 1 ] ; then

            python ./Utils/ensemble_score.py --ensemble 1

        elif [ "${experiments[0]}" = 2 ] ; then

            FILE=./Predictions/28cat-IncResV2-GAN07.csv
            if ! test -f "$FILE"; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 7 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-GAN07 --modelName 28cat-IncResV2-GAN07 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv 
            fi

            FILE=./Predictions/28cat-IncResV2-GAN10.csv
            if ! test -f "$FILE"; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 7 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-GAN10 --modelName 28cat-IncResV2-GAN10 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv  
            fi

            python ./Utils/ensemble_score.py --ensemble 2

        elif [ "${experiments[0]}" = 3 ] ; then

            FILE=./Predictions/28cat-IncResV2-GAN07.csv
            if ! test -f "$FILE"; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 7 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-GAN07 --modelName 28cat-IncResV2-GAN07 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv 
            fi

            FILE=./Predictions/28cat-IncResV2-GAN10.csv
            if ! test -f "$FILE"; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 7 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-GAN10 --modelName 28cat-IncResV2-GAN10 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv  
            fi

            FILE=./Predictions/28cat-IncResV2-Augmentation.csv
            if ! test -f "$FILE"; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-Augmentation --modelName 28cat-IncResV2-Augmentation \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv \
                --inputAugmentation 
            fi

            FILE=./Predictions/28cat-IncResV2-random1.csv
            if ! test -f "$FILE"; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-random1 --modelName 28cat-IncResV2-random1 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            fi

            FILE=./Predictions/28cat-IncResV2-random2.csv
            if ! test -f "$FILE"; then
                python trainer.py -m IncResV2 -c --imageWidth 224 --imageHeight 224 --imageChannels 3 \
                --resizeRatio 1.15 --learningRate 1e-4 --trainingEpochs 10 --lossFunction ce --batchSize 20 --saveStep 1000 \
                --displayStep 1000 --modelDir ./Models/28cat-IncResV2-random2 --modelName 28cat-IncResV2-random2 \
                --trainDataFile ./Data/title28cat-labels-train.txt \
                --valDataFile ./Data/title28cat-labels-test.txt \
                --testDataFile ./Data/title28cat-labels-test.txt \
                --imageBaseDir ./Data/images/ \
                --classesFile ./Data/title28cat-classes.csv
            fi

            python ./Utils/ensemble_score.py --ensemble 3
        else
            echo ERROR: Unknown ensemble [1, 2, 3]
            break
        fi

    else
        echo ERROR: Unknown mode flag [train, test, ensemble]
        break
    fi

    echo Finished experiment: ${experiments[0]}!

    # Drop first flag
    experiments=("${experiments[@]:1}")
done
