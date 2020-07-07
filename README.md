# BookCoverClassification

This repository provides source code and models for the experiments in "[Benchmarking Deep Learning Models for Classification of Book Covers](https://link.springer.com/content/pdf/10.1007/s42979-020-00132-z.pdf)".

If you use this repository, please consider citing the associated paper as follows:
```
    @article{lucieri2020benchmarking,
  title={Benchmarking Deep Learning Models for Classification of Book Covers},
  author={Lucieri, Adriano and Sabir, Huzaifa and Siddiqui, Shoaib Ahmed and Rizvi, Syed Tahseen Raza and Iwana, Brian Kenji and Uchida, Seiichi and Dengel, Andreas and Ahmed, Sheraz},
  journal={SN Computer Science},
  volume={1},
  pages={1--16},
  year={2020},
  publisher={Springer}
}
```

## Models 

All trained models can be accessed [here](https://cloud.dfki.de/owncloud/index.php/s/4MTX8AbZ9ijDHPB). The FastText models need to be placed in the *FastTest* folder. All remaining models belong in the *Models* folder. 

## Dependencies

Python dependendcies are specified in the requirements file. Further dependencies include the Repositories [TFSENet](https://github.com/HiKapok/TF-SENet), [tensorflow_densenet](https://github.com/pudae/tensorflow-densenet), [download_google_drive](https://github.com/chentinghao/download_google_drive) and [tensorflow/models/research/slim/nets](https://github.com/tensorflow/models/tree/master/research/slim/nets) which must be placed in the folder *Repos*.

## Usage

To reproduce the results simply run:

```
bash main.sh $MODE $EXPERIMENT1 $ EXPERIMENT2 ...
```

An example is given in the *test.sh* file. For training and testing use the **train** and **test** mode flags. The experiment variable must be chosen as one of the following:

| Variable | Experiment Description |
| --- | --- |
| 30cat-NAS | NASNet trained on 30 classes |
| 30cat-SEResNet | SEResNet trained on 30 classes |
| 30cat-SEResNeXt | SEResNeXt trained on 30 classes |
| 30cat-IncResV2 | IncResV2 trained on 30 classes |
| 30cat-ResNet152 | ResNet152 trained on 30 classes |
| 30cat-ResNet50 | ResNet50 trained on 30 classes |
| 30cat-VGG16 | VGG16 trained on 30 classes |
| 30cat-DenseNet | DenseNet trained on 30 classes |
| 28cat-IncResV2 | IncResV2 trained on 28 classes |
| Aug | IncResV2 with enhanced augmentation |
| Att-softmax | IncResV2 with softmax attention |
| Att-sigmoid | IncResV2 with sigmoid attention |
| Att-tempSM | IncResV2 with tempered softmax attention |
| Att-saliency | IncResV2 with saliency-based attention |
| Att-residual | IncResV2 with residual attention |
| Att-resStacked | IncResV2 with alternative residual attention |
| Att-STN | IncResV2 with STN attention |
| Att-combinedSTN | IncResV2 with STN attention with IncResV2 as localization network |
| MSE | IncResV2 with MSE loss |
| GAN07 | IncResV2 pre-trained on GAN images (Only **test**) |
| GAN10 | IncResV2 pre-trained on GAN images (Only **test**) |
| Text-early | Early text and image fusion |
| Text-late | Late text and image fusion |
| Text-dual | Dual text and image fusion |
| Text-lateSVM | Late text and image fusion wiht SVM on head |

The **ensemble** mode allows to reproduce the ensemble results given in the paper. As experiment variable use one of the follwoing:

| Variable | Experiment Description |
| --- | --- |
| 1 | Ensemble of 4 Models |
| 2 | Ensemble of 6 Models |
| 3 | Ensemble of 9 Modles |

## Acknowledgement

The tensorflow-based training script is based on [shoaibahmed](https://github.com/shoaibahmed)'s [ClassificationCNN-TF](https://github.com/shoaibahmed/ClassificationCNN-TF).