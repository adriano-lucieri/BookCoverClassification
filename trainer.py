from __future__ import division
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import tensorflow as tf

slim = tf.contrib.slim

import numpy as np
import seaborn as sns
import pandas as pd
import tarfile
import wget
import time
import sys
import cv2
import csv
import os

from optparse import OptionParser
from pyfasttext import FastText
from sklearn import svm

# Import base networks
import Repos.nets.inception_resnet_v2 as inception_resnet_v2
import custom_nets.inception_resnet_v2 as inception_resnet_v2_sas
import Repos.nets.vgg as vgg
import Repos.nets.resnet_v1 as resnet_v1
import Repos.nets.nasnet.nasnet as nasnet
import Repos.TFSENet.se_resnet as se_resnet
import Repos.TFSENet.se_resnext as se_resnext
import Repos.tensorflow_densenet.nets.densenet as densenet
import Repos.download_google_drive.download_gdrive as driveGet

TRAIN = 0
VAL = 1
TEST = 2

if sys.version_info[0] == 3:
    print("Using Python 3")
    import pickle as cPickle
else:
    print("Using Python 2")
    import cPickle

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-m", "--model", action="store", type="string", dest="model", default="NAS",
                  help="Model to be used for Cross-Layer Pooling")
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False,
                  help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False,
                  help="Test model")
# TODO: Leave eval option?
parser.add_option("-e", "--evalClasses", action="store_true", dest="evalClasses", default=False,
                  help="Evaluate precision and recall per class")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch",
                  default=False, help="Start training from scratch")
parser.add_option("-v", "--tensorboardVisualization", action="store_true", dest="tensorboardVisualization",
                  default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=56,
                  help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=56,
                  help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3,
                  help="Number of channels in the image")
# TODO: Set to fixed or leave changeable?
parser.add_option("--resizeRatio", action="store", type="float", dest="resizeRatio", default=1.15,
                  help="Resizing image ratio")
# TODO: Same?
parser.add_option("--useImageMean", action="store_true", dest="useImageMean", default=False,
                  help="Use image mean for normalization")
parser.add_option("--attentionType", action="store", dest="attentionType", default="None",
                  help="Select which type of attention mechanism to use")
parser.add_option("--useMultimodal", action="store_true", dest="useMultimodal", default=False,
                  help="Use text + image ensemble classifier")
parser.add_option("--multimodalFusion", action="store", type="string", dest="multimodalFusion", default="early",
                  help="Select type of fusion for multimodal text + image ensemble")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4,
                  help="Learning rate")
parser.add_option("--labelSmoothing", action="store", type="float", dest="labelSmoothing", default=0.1,
                  help="Label smoothing parameter")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=1,
                  help="Training epochs")
parser.add_option("--lossFunction", action="store", type="string", dest="lossFunction", default="ce",
                  help="Loss function to be used for training")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5,
                  help="Progress display step")
parser.add_option("--inputAugmentation", action="store_true", dest="inputAugmentation", default=False,
                  help="Activate extended input image augmentation")
parser.add_option("--trainSVM", action="store_true", dest="trainSVM", default=False,
                  help="Train SVM on top of the features extracted from the trained model")

# Directories
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="./Models/mymodel/",
                  help="Directory for saving the model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="mymodel",
                  help="Name to be used for saving the model")

parser.add_option("--modelFasttext", action="store", type="string", dest="modelFasttext",
                  default="./Repos/fastText/models/model-v2.bin",
                  help="Path to fasttext model to be used")
# TODO: Set fixed
parser.add_option("--embeddingVector", action="store", type="int", dest="embeddingVector", default=100,
                  help="Length of text embedding vector")

parser.add_option("--trainDataFile", action="store", type="string", dest="trainDataFile",
                  default="/home/adri/Data/title30cat/title30cat-labels-train-smaller.txt",
                  help="Training data file")
# TODO: remove val or always set to testset?
parser.add_option("--valDataFile", action="store", type="string", dest="valDataFile",
                  default="/home/adri/Data/title30cat/title30cat-labels-val-smaller.txt",
                  help="Validation data file")
parser.add_option("--testDataFile", action="store", type="string", dest="testDataFile",
                  default="/home/adri/Data/title30cat/title30cat-labels-test-smaller.txt",
                  help="Test data file")
parser.add_option("--imageBaseDir", action="store", type="string", dest="imageBaseDir",
                  default="/home/adri/Data/title30cat/56x56/", help="Image base directory")
parser.add_option("--classesFile", action="store", type="string", dest="classesFile",
                  default="/home/adri/PycharmProjects/title30cat/Data/title28cat-classes_test.csv",
                  help="Path to classes file")
parser.add_option("--experimentName", action="store", type="string", dest="experimentName",
                  help="Name of experiment")

# Parse command line options
(options, args) = parser.parse_args()
print(options)

baseDir = os.getcwd()
targetModelDir = os.path.join(options.modelDir, options.modelName)
preTrainedModelDir = os.path.join(baseDir, 'Pre-trained')
predictionsDir = os.path.join(baseDir, 'Predictions')
if options.trainModel:
    if os.path.isfile(targetModelDir + '.data-00000-of-00001'):
        print('Model already trained.\n')
        exit()
if not os.path.exists(preTrainedModelDir):
    os.makedirs(preTrainedModelDir)
if not os.path.exists(predictionsDir):
    os.makedirs(predictionsDir)
usingGlobalPooledFeatures = False
imagesBaseDir = options.imageBaseDir

# Override import if attention is used
if options.attentionType != "None":
    if options.attentionType == "softmax":
        import custom_nets.inception_resnet_v2_attent_softmax as inception_resnet_v2
    elif options.attentionType == "sigmoid":
        import custom_nets.inception_resnet_v2_attent_sigmoid as inception_resnet_v2
    elif options.attentionType == "tempSM":
        import custom_nets.inception_resnet_v2_attent_tempSM as inception_resnet_v2
    elif options.attentionType == "saliency":
        import custom_nets.inception_resnet_v2_attent_saliency as inception_resnet_v2
    elif options.attentionType == "residual":
        import custom_nets.inception_resnet_v2_resAttent as inception_resnet_v2
    elif options.attentionType == "resStack":
        import custom_nets.inception_resnet_v2_resAttent_stacked as inception_resnet_v2
    elif options.attentionType in ["STN", "combSTN"]:
        from stn import spatial_transformer_network as transformer
    else:
        print("Error: Unknown attention mechanism selected")
        exit(-1)

if options.multimodalFusion in ["late", "dual"] and options.useMultimodal:
    import custom_nets.inception_resnet_v2_lateFusion_FC as inception_resnet_v2
elif options.multimodalFusion == "early":
    pass
else:
    print("Error: Unknown multimodal fusing scheme selected")
    exit(-1)

# Load Fasttext model if multimodal is selected
if options.useMultimodal:
    modelFT = FastText(options.modelFasttext)

# Download the model
if options.model == "ResNet":
    checkpointFileName = os.path.join(preTrainedModelDir, 'resnet152/resnet_v1_152.ckpt')
    if not os.path.isfile(checkpointFileName):
        # Download file from the link
        url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz'
        filename = wget.download(url, './Pre-trained')

        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/resnet152/')
        tar.close()

    options.imageHeight = options.imageWidth = 224

elif options.model == "ResNet50":
    checkpointFileName = os.path.join(preTrainedModelDir, 'resnet50/resnet_v1_50.ckpt')
    if not os.path.isfile(checkpointFileName):
        # Download file from the link
        url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'
        filename = wget.download(url, './Pre-trained')
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/resnet50/')
        tar.close()
        
        os.remove(filename)
    
    options.imageHeight = options.imageWidth = 224

elif options.model == "VGG16":
    checkpointFileName = os.path.join(preTrainedModelDir, 'vgg16/vgg_16.ckpt')
    if not os.path.isfile(checkpointFileName):
        # Download file from the link
        url = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'
        filename = wget.download(url, './Pre-trained')
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/vgg16/')
        tar.close()
        
        os.remove(filename)
    
    options.imageHeight = options.imageWidth = 224

elif options.model == "IncResV2":
    checkpointFileName = os.path.join(preTrainedModelDir, 'IncResV2/inception_resnet_v2_2016_08_30.ckpt')
    if not os.path.isfile(checkpointFileName):
        # Download file from the link
        url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
        filename = wget.download(url, './Pre-trained')

        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/IncResV2/')
        tar.close()

    options.imageHeight = options.imageWidth = 299

elif options.model == "NAS":
    checkpointFileName = os.path.join(preTrainedModelDir, 'nas/model.ckpt')
    if not os.path.isfile(checkpointFileName + '.index'):
        # Download file from the link
        url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
        filename = wget.download(url, './Pre-trained')

        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/nas/')
        tar.close()

    # Update image sizes
    options.imageHeight = options.imageWidth = 331

elif options.model == "DenseNet":
    checkpointFileName = os.path.join(preTrainedModelDir, 'tf-densenet161/tf-densenet161.ckpt')
    if not os.path.isfile(checkpointFileName + '.index'):
        # Download file from the link
        id = '0B_fUSpodN0t0NmZvTnZZa2plaHc'
        filename = './Pre-trained/tf-densenet161.tar.gz'
        driveGet.download_file_from_google_drive(id, filename)
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/tf-densenet161/')
        tar.close()
        
        os.remove(filename)
    
    # Update image sizes
    options.imageHeight = options.imageWidth = 224

elif options.model == "SEResNet":
    checkpointFileName = os.path.join(preTrainedModelDir, 'seresnet101/se_resnet101.ckpt')
    if not os.path.isfile(checkpointFileName + '.index'):
        # Download file from the link
        id = '19QsGHNZC0BVsaDf4Sx79J2Hl2BV9wpRm'
        filename = './Pre-trained/seresnet101.tar.gz'
        driveGet.download_file_from_google_drive(id, filename)
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/seresnet101/')
        tar.close()
        
        os.remove(filename)
    
    # Update image sizes
    options.imageHeight = options.imageWidth = 224

elif options.model == "SEResNeXt":
    checkpointFileName = os.path.join(preTrainedModelDir, 'seresnext101/se_resnext101.ckpt')
    if not os.path.isfile(checkpointFileName + '.index'):
        # Download file from the link
        id = '1AEYDWTWEGh6xGN-fSFB_f94FujdTJyKS'
        filename = './Pre-trained/seresnext101.tar.gz'
        driveGet.download_file_from_google_drive(id, filename)
        
        # Extract the tar file
        tar = tarfile.open(filename)
        tar.extractall('./Pre-trained/seresnext101/')
        tar.close()
        
        os.remove(filename)
    
    # Update image sizes
    options.imageHeight = options.imageWidth = 224

else:
    print("Error: Unknown model selected")
    exit(-1)

# Define params
IMAGENET_MEAN = [123.68, 116.779, 103.939]  # RGB

# Decide the resizing dimensions
RESIZED_IMAGE_DIMS = [int(options.imageHeight * options.resizeRatio), int(options.imageWidth * options.resizeRatio)]
print("Resized image dimensions: %s" % str(RESIZED_IMAGE_DIMS))


def rand_rot(img, angle_range):

    rows, cols, ch = img.shape
    angle = np.random.uniform(low=-angle_range, high=angle_range)
    m = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, m, (cols, rows))

    return img


def rand_blur(img, max_blur):

    f = int(round(np.random.uniform(low=1, high=max_blur)))
    img = cv2.blur(img, (f, f))

    return img


def rand_trans(img, max_x, max_y):

    rows, cols, ch = img.shape
    rand_x = round(np.random.uniform(low=0, high=max_x))
    rand_y = round(np.random.uniform(low=0, high=max_y))
    M = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
    img = cv2.warpAffine(img, M, (cols, rows))

    return img


def getMask(img):
    # Calc saliency map
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, att_mask = saliency.computeSaliency(img)
    # Binarizing image with threshold of 0.039 (corresponds to 10 for [0, 255])
    _, att_mask = cv2.threshold(att_mask, 0.039, 1, cv2.THRESH_BINARY)
    # Convert to tensor
    att_mask = cv2.resize(att_mask, (8, 8))

    return att_mask


def getSentenceVecs(title):

    vec = modelFT.get_numpy_sentence_vector(str(title))

    return vec


# Reads an image from a file, decodes it into a dense tensor
def _parse_function(filename, label, title=None):
    image_string = tf.read_file(filename)
    img = tf.image.decode_jpeg(image_string)

    if options.trainModel:
        if options.inputAugmentation:
            tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_images(img, RESIZED_IMAGE_DIMS)
            img = tf.random_crop(img, [options.imageHeight, options.imageWidth, options.imageChannels])
            # Random flipping
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)

            # Random contrast
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

            # Random HUE
            img = tf.image.random_hue(img, max_delta=0.1)

            # Random saturation
            img = tf.image.random_saturation(img, lower=0.9, upper=1.1)

            # Random blur
            img = tf.py_func(rand_blur, [img, 4], tf.float32)

            # Random rotation
            img = tf.py_func(rand_rot, [img, 7], tf.float32)

            # Random translation
            img = tf.py_func(rand_trans, [img, 10, 10], tf.float32)

        else:
            tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_images(img, RESIZED_IMAGE_DIMS)
            img = tf.random_crop(img, [options.imageHeight, options.imageWidth, options.imageChannels])

            # Random flipping
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)

    else:
        img = tf.image.resize_images(img, [options.imageHeight, options.imageWidth])

    img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
    img = tf.cast(img, tf.float32)  # Convert to float tensor

    if options.useMultimodal:
        ### Compute sentence vector
        sent_vec = tf.py_func(getSentenceVecs, [title], tf.float32)
        sent_vec = tf.reshape(sent_vec, [options.embeddingVector])

        return img, filename, label, sent_vec

    if options.attentionType in ["saliency", "residual", "resStack"]:
        ### Saliency map
        att_map = tf.py_func(getMask, [img], tf.float32)
        att_map = tf.expand_dims(att_map, axis=2)
        att_map = tf.broadcast_to(att_map, shape=(8, 8, 1536))

        return img, filename, label, att_map

    return img, filename, label


def loadDataset(currentDataFile):
    global imagesBaseDir
    print("Loading data from file: %s" % (currentDataFile))
    dataClasses = {}
    with open(currentDataFile) as f:
        imageFileNames = f.readlines()
        imNames = []
        imLabels = []
        imTitles = []
        for imName in imageFileNames:
            # Image name
            imName = imName.strip().split('|')
            imNames.append(os.path.join(imagesBaseDir, imName[0]))
            # Image label
            currentLabel = int(imName[1])
            imLabels.append(currentLabel)
            # Image title
            if options.useMultimodal:
                imTitles.append(imName[2])

            if currentLabel not in dataClasses:
                dataClasses[currentLabel] = 1
            else:
                dataClasses[currentLabel] += 1

        imNames = tf.constant(imNames)
        imLabels = tf.constant(imLabels)
        if options.useMultimodal:
            imTitles = tf.constant(imTitles)

    numClasses = len(dataClasses)
    numFiles = len(imageFileNames)
    print("Dataset loaded")
    print("Files: %d | Classes: %d" % (numFiles, numClasses))
    print(dataClasses)
    classWeights = [float(numFiles - dataClasses[x]) / float(numFiles) for x in dataClasses]
    print("Class weights: %s" % str(classWeights))

    if options.useMultimodal:
        dataset = tf.data.Dataset.from_tensor_slices((imNames, imLabels, imTitles))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((imNames, imLabels))

    dataset = dataset.map(_parse_function)
    if options.trainModel:
        dataset = dataset.shuffle(buffer_size=numFiles)
    dataset = dataset.batch(options.batchSize)
    dataset = dataset.prefetch(buffer_size=numFiles)

    return dataset, numClasses, classWeights


# A vector of filenames
trainDataset, numClasses, classWeights = loadDataset(options.trainDataFile)
valDataset, _, _ = loadDataset(options.valDataFile)
testDataset, _, _ = loadDataset(options.testDataFile)

trainIterator = trainDataset.make_initializable_iterator()
valIterator = valDataset.make_initializable_iterator()
testIterator = testDataset.make_initializable_iterator()

global_step = tf.train.get_or_create_global_step()


def spatial_transformer(input_placeholder):
    with tf.variable_scope('Attention'):
        # Create identity transform initialization
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32').flatten()

        # Localization network to output transform parameters
        locNet = tf.layers.max_pooling2d(inputs=input_placeholder, pool_size=2, strides=2, padding='valid')
        # 112x112x32
        locNet = tf.layers.conv2d(inputs=locNet, filters=32, kernel_size=[7, 7], strides=(1, 1), padding='same',
                                  activation=tf.nn.relu)
        locNet = tf.layers.batch_normalization(locNet, training=options.trainModel)
        # 56x56x32
        locNet = tf.layers.max_pooling2d(inputs=locNet, pool_size=2, strides=2, padding='valid')
        # 56x56x64
        locNet = tf.layers.conv2d(inputs=locNet, filters=64, kernel_size=[5, 5], strides=(1, 1), padding='same',
                                  activation=tf.nn.relu)
        locNet = tf.layers.batch_normalization(locNet, training=options.trainModel)
        # 28x28x64
        locNet = tf.layers.max_pooling2d(inputs=locNet, pool_size=2, strides=2, padding='valid')
        # 28x28x100
        locNet = tf.layers.conv2d(inputs=locNet, filters=100, kernel_size=[3, 3], strides=(1, 1), padding='same',
                                  activation=tf.nn.relu)
        locNet = tf.layers.batch_normalization(locNet, training=options.trainModel)
        # 14x14x100
        locNet_m1 = tf.layers.max_pooling2d(inputs=locNet, pool_size=2, strides=2, padding='valid')
        # 14x14x200
        locNet = tf.layers.conv2d(inputs=locNet_m1, filters=200, kernel_size=[3, 3], strides=(1, 1), padding='same',
                                  activation=tf.nn.relu)
        locNet = tf.layers.batch_normalization(locNet, training=options.trainModel)
        # 4x4x200
        locNet_m2 = tf.layers.max_pooling2d(inputs=locNet, pool_size=5, strides=3, padding='valid')

        # 4x4x100
        locNet_m1 = tf.layers.max_pooling2d(inputs=locNet_m1, pool_size=5, strides=3, padding='valid')
        # Concatenate
        locNet = tf.concat([locNet_m1, locNet_m2], axis=3)

        # Flatten
        locNet_flat = tf.layers.flatten(inputs=locNet)
        # locNet_flat = tf.layers.dense(inputs=locNet_flat, units=512, activation=tf.nn.relu)
        locNet_flat = tf.layers.dense(inputs=locNet_flat, units=512, activation=tf.nn.tanh)
        locNet = tf.layers.batch_normalization(locNet, training=options.trainModel)
        # LocNet output
        W_fc1 = tf.Variable(tf.zeros([512, 6]), name='W_fc1')
        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(locNet_flat, W_fc1) + b_fc1

        # Apply transformer layer
        output_placeholder = transformer(input_placeholder, h_fc1, (options.imageHeight, options.imageWidth))

        return output_placeholder


def spatial_transformer_combined(input_placeholder, transformer_output):
    with tf.variable_scope('Transformer'):
        # Create identity transform initialization
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32').flatten()

        # ?x35x35x20 to ?x24500
        locNet = tf.reshape(transformer_output, [-1, 133120])

        # Pass through a final localization layer
        W_fc1 = tf.Variable(tf.zeros([133120, 6]), name='W_fc1')
        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(locNet, W_fc1) + b_fc1

        # Apply transformer layer
        output_placeholder = transformer(input_placeholder, h_fc1,
                                         (options.imageHeight, options.imageWidth))

    arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2.inception_resnet_v2(output_placeholder,
                                                                     is_training=options.trainModel,
                                                                     num_classes=numClasses, reuse=True)

    return logits, end_points


if options.modelName in ['30cat-DenseNet', '30cat-SEResNeXt']:
    scope_name = ''
else:
    scope_name = 'Model'

with tf.name_scope(scope_name):
    # Data placeholders
    datasetSelectionPlaceholder = tf.placeholder(dtype=tf.int32, shape=(), name='DatasetSelectionPlaceholder')

    if options.useMultimodal:
        inputBatchImages, inputBatchImageNames, inputBatchLabels, inputBatchSentVec = tf.cond(tf.equal(
            datasetSelectionPlaceholder, TRAIN), lambda: trainIterator.get_next(),
            lambda: tf.cond(tf.equal(datasetSelectionPlaceholder, VAL), lambda: valIterator.get_next(),
                            lambda: testIterator.get_next()))
    elif options.attentionType in ["saliency", "residual", "resStack"]:
        inputBatchImages, inputBatchImageNames, inputBatchLabels, inputBatchMasks = tf.cond(tf.equal(
            datasetSelectionPlaceholder, TRAIN), lambda: trainIterator.get_next(),
            lambda: tf.cond(tf.equal(datasetSelectionPlaceholder, VAL), lambda: valIterator.get_next(),
                            lambda: testIterator.get_next()))
    else:
        inputBatchImages, inputBatchImageNames, inputBatchLabels = tf.cond(tf.equal(datasetSelectionPlaceholder, TRAIN),
                                                                           lambda: trainIterator.get_next(),
                                                                           lambda: tf.cond(
                                                                               tf.equal(datasetSelectionPlaceholder,
                                                                                        VAL),
                                                                               lambda: valIterator.get_next(),
                                                                               lambda: testIterator.get_next()))

    inputBatchImageLabels = tf.one_hot(inputBatchLabels, depth=numClasses)

    print("Data shape: %s" % str(inputBatchImages.get_shape()))
    print("Labels shape: %s" % str(inputBatchImageLabels.get_shape()))

    if options.model == "IncResV2":
        scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImages)
        scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
        scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

        if options.multimodalFusion in ["early", "dual"] and options.useMultimodal:
            # Fusion of network input
            inputBroadcastEmbedding = tf.expand_dims(inputBatchSentVec, 1)
            inputBroadcastEmbedding = tf.expand_dims(inputBroadcastEmbedding, 1)
            inputBroadcastEmbedding = tf.tile(inputBroadcastEmbedding, multiples=[1, 299, 299, 1])

            scaledInputBatchImages = tf.concat([inputBatchImages, inputBroadcastEmbedding], axis=3)

        if options.attentionType == "STN":
            scaledInputBatchImages = spatial_transformer(scaledInputBatchImages)

        if options.multimodalFusion in ["late", "dual"] and options.useMultimodal:
            arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
                logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages,
                                                                             inputBatchSentVec,
                                                                             is_training=options.trainModel,
                                                                             num_classes=numClasses)
        elif options.attentionType in ["saliency", "residual", "resStack"]:
            arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
                logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages,
                                                                             inputBatchMasks,
                                                                             is_training=options.trainModel,
                                                                             num_classes=numClasses)
        elif options.attentionType == "tempSM":
            arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
                logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages,
                                                                             is_training=options.trainModel,
                                                                             num_classes=numClasses,
                                                                             T=50)
        elif options.modelName == '30cat-IncResV2':
            arg_scope = inception_resnet_v2_sas.inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
                logits, _, end_points = inception_resnet_v2_sas.inception_resnet_v2(scaledInputBatchImages,
                                                                             dropout_keep_prob=0.8,
                                                                             is_training=options.trainModel,
                                                                             num_classes=numClasses)
        else:
            arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
            with slim.arg_scope(arg_scope):
                logits, end_points = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages,
                                                                             is_training=options.trainModel,
                                                                             num_classes=numClasses)
        # Apply combined STN architecture if selected
        if options.attentionType == "combSTN":
            locNet_input = end_points['Mixed_7a']
            logits, end_points = spatial_transformer_combined(scaledInputBatchImages, locNet_input)

        tf.summary.image('Input Images', scaledInputBatchImages, max_outputs=3)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        # Depending on attention configuration
        if options.useMultimodal:
            variables_to_restore = slim.get_variables_to_restore(
                exclude=["InceptionResnetV2/Conv2d_1a_3x3", "InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"])
        elif options.attentionType == "None":
            if options.modelName == '30cat-IncResV2':
                variables_to_restore = slim.get_variables_to_restore(include=["InceptionResnetV2"], exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"])
            else:
                variables_to_restore = slim.get_variables_to_restore(exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"])
        elif options.attentionType == "STN":
            variables_to_restore = slim.get_variables_to_restore(
                exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits", "Model/Attention", "Attention"])
        elif options.attentionType == "combSTN":
            variables_to_restore = slim.get_variables_to_restore(
                exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits", "Model/Attention", "Attention",
                         "Model/Transformer", "Transformer"])
        else:
            variables_to_restore = slim.get_variables_to_restore(
                exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits", "InceptionResnetV2/Attention"])

        # Last layer for extraction of features before global pool
        featureVector = end_points['Mixed_7a']
        embeddingLayer = end_points['PreLogitsFlatten']

    elif options.model == "ResNet":
        if options.useImageMean:
            imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
            print("Image mean shape: %s" % str(imageMean.shape))
            processedInputBatchImages = inputBatchImages - imageMean
        else:
            print(inputBatchImages.shape)
            channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
            for i in range(options.imageChannels):
                channels[i] -= IMAGENET_MEAN[i]
            processedInputBatchImages = tf.concat(axis=3, values=channels)
            print(processedInputBatchImages.shape)

        # Create model
        arg_scope = resnet_v1.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=options.trainModel,
                                                         num_classes=numClasses)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(
            exclude=["resnet_v1_152/logits", "resnet_v1_152/AuxLogits"])

        # Last layer for extraction of features before global pool
        featureVector = end_points['resnet_v1_152/block4']

    elif options.model == "ResNet50":
        if options.useImageMean:
            imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
            print("Image mean shape: %s" % str(imageMean.shape))
            processedInputBatchImages = inputBatchImages - imageMean
        else:
            print(inputBatchImages.shape)
            channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
            for i in range(options.imageChannels):
                channels[i] -= IMAGENET_MEAN[i]
            processedInputBatchImages = tf.concat(axis=3, values=channels)
            print(processedInputBatchImages.shape)

        # Create model
        arg_scope = resnet_v1.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            # logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
            logits, end_points = resnet_v1.resnet_v1_50(processedInputBatchImages, is_training=options.trainModel,
                                                         num_classes=numClasses)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(
            exclude=["resnet_v1_50/logits", "resnet_v1_50/AuxLogits"])

        # Last layer for extraction of features before global pool
        featureVector = end_points['resnet_v1_50/block4']

    elif options.model == "VGG16":
        if options.useImageMean:
            imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
            print("Image mean shape: %s" % str(imageMean.shape))
            processedInputBatchImages = inputBatchImages - imageMean
        else:
            print(inputBatchImages.shape)
            channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
            for i in range(options.imageChannels):
                channels[i] -= IMAGENET_MEAN[i]
            processedInputBatchImages = tf.concat(axis=3, values=channels)
            print(processedInputBatchImages.shape)

        # Create model
        arg_scope = vgg.vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            # logits, end_points = resnet_v1.resnet_v1_152(processedInputBatchImages, is_training=options.trainModel, num_classes=numClasses)
            logits, end_points = vgg.vgg_16(processedInputBatchImages, is_training=options.trainModel,
                                                         num_classes=numClasses)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(
            exclude=["vgg_16/fc8"])

        # Last layer for extraction of features before global pool
        featureVector = logits

    elif options.model == "NAS":
        scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImages)
        scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
        scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

        # Create model
        arg_scope = nasnet.nasnet_large_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=options.trainModel,
                                                           num_classes=numClasses)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(exclude=["aux_11/aux_logits/FC", "final_layer/FC"])

        # Last layer for extraction of features before global pool
        featureVector = end_points['Cell_17']

    elif options.model == "DenseNet":
        if options.useImageMean:
            imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
            print("Image mean shape: %s" % str(imageMean.shape))
            processedInputBatchImages = inputBatchImages - imageMean
        else:
            print(inputBatchImages.shape)
            channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
            for i in range(options.imageChannels):
                channels[i] -= IMAGENET_MEAN[i]
            processedInputBatchImages = tf.concat(axis=3, values=channels)
            print(processedInputBatchImages.shape)

        arg_scope = densenet.densenet_arg_scope()
        with slim.arg_scope(arg_scope):
            net, end_points = densenet.densenet161(processedInputBatchImages, num_classes=numClasses, is_training=options.trainModel)
            logits = tf.layers.flatten(net)

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(exclude=["densenet161/logits"])

        # Last layer for extraction of features before global pool
        featureVector = tf.get_default_graph().get_tensor_by_name("densenet161/dense_block4/conv_block24/concat:0")

    elif options.model == "SEResNet":
        if options.useImageMean:
            imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
            print("Image mean shape: %s" % str(imageMean.shape))
            processedInputBatchImages = inputBatchImages - imageMean
        else:
            print(inputBatchImages.shape)
            channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
            for i in range(options.imageChannels):
                channels[i] -= IMAGENET_MEAN[i]
            processedInputBatchImages = tf.concat(axis=3, values=channels)
            print(processedInputBatchImages.shape)

        # Convert RGB to BGR for SEResNet input
        processedInputBatchImages = tf.reverse(processedInputBatchImages, [-1])

        # Create model
        logits, _ = se_resnet.SE_ResNet(processedInputBatchImages, numClasses, is_training=options.trainModel,
                                        data_format='channels_last')

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(exclude=["dense", "global_step"])

        # Last layer for extraction of features
        featureVector = tf.get_default_graph().get_tensor_by_name("Model/conv5_3/relu:0")
        
    elif options.model == "SEResNeXt":
        if options.useImageMean:
            imageMean = tf.reduce_mean(inputBatchImages, axis=[1, 2], keepdims=True)
            print("Image mean shape: %s" % str(imageMean.shape))
            processedInputBatchImages = inputBatchImages - imageMean
        else:
            print(inputBatchImages.shape)
            channels = tf.split(axis=3, num_or_size_splits=options.imageChannels, value=inputBatchImages)
            for i in range(options.imageChannels):
                channels[i] -= IMAGENET_MEAN[i]
            processedInputBatchImages = tf.concat(axis=3, values=channels)
            print(processedInputBatchImages.shape)

        # Convert RGB to BGR for SEResNext input
        processedInputBatchImages = tf.reverse(processedInputBatchImages, [-1])

        # Create model
        logits, _ = se_resnext.SE_ResNeXt(processedInputBatchImages, numClasses, is_training=options.trainModel,
                                        data_format='channels_last')

        # Create list of vars to restore before train op (exclude the logits due to change in number of classes)
        variables_to_restore = slim.get_variables_to_restore(exclude=["dense", "global_step"])

        # Last layer for extraction of features
        featureVector = tf.get_default_graph().get_tensor_by_name("conv5_3/relu:0")

    else:
        print("Error: Unknown model selected")
        exit(-1)

print("Feature Vector Dimensions: %s" % str(featureVector.get_shape()))

with tf.name_scope('Loss'):
    # Define loss
    if options.lossFunction == "ce":
        cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=inputBatchImageLabels, logits=logits,
                                                             label_smoothing=options.labelSmoothing)
        tf.losses.add_loss(cross_entropy_loss)
    elif options.lossFunction == "mse":
        MSE_losses = tf.losses.mean_squared_error(labels=inputBatchImageLabels, predictions=logits)
        tf.losses.add_loss(MSE_losses)
    else:
        print("Error: Unknown loss function selected")
        exit(-1)

    loss = tf.reduce_sum(tf.losses.get_losses())

with tf.name_scope('Accuracy'):
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(inputBatchImageLabels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

with tf.name_scope('Optimizer'):
    # Define Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

    # Op to calculate every variable gradient
    gradients = tf.gradients(loss, tf.trainable_variables())
    gradients = list(zip(gradients, tf.trainable_variables()))

    # Op to update all variables according to their gradient
    trainOp = optimizer.apply_gradients(grads_and_vars=gradients)
    # Update operation for batchnorm
    updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updateOps):
        # applyGradOp = optimizer.minimize(loss, global_step=globalStep)
        #applyGradOp = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
        trainOp = optimizer.apply_gradients(grads_and_vars=gradients)

# Initializing the variables
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

if options.tensorboardVisualization:
    # Create a summary to monitor cost and accuracy
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    # Merge all summaries into a single op
    mergedSummaryOp = tf.summary.merge_all()

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Train model
if options.trainModel:
    with tf.Session(config=config) as sess:
        # Initialize all vars
        sess.run(init)
        sess.run(init_local)

        # Restore the model params
        if options.startTrainingFromScratch:
            print("Removing previous checkpoints and logs")
            os.system("rm -rf " + options.modelDir + "/logs")
            os.system("rm -rf " + options.modelDir)
            os.system("mkdir " + options.modelDir)

            #checkpointFileName = resnet_checkpoint_file if options.model == "ResNet" else inc_res_v2_checkpoint_file if options.model == "IncResV2" else nas_checkpoint_file
            print("Restoring weights from file: %s" % (checkpointFileName))

            # Load the imagenet pre-trained model
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, checkpointFileName)
        else:
            # Load the user trained model
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, targetModelDir)

        # Saver op to save and restore all the variables
        saver = tf.train.Saver()

        if options.tensorboardVisualization:
            # Write the graph to file
            summaryWriter = tf.summary.FileWriter(options.modelDir + "/logs", graph=tf.get_default_graph())

        globalStep = 0
        numEpochs = options.trainingEpochs + 1 if options.trainSVM else options.trainingEpochs
        if options.trainSVM:
            imageNames = []
            imageLabels = []
            imageFeatures = []

        for epoch in range(numEpochs):
            # Initialize the dataset iterator
            sess.run(trainIterator.initializer)
            isLastEpoch = epoch == options.trainingEpochs
            try:
                step = 0
                while True:
                    start_time = time.time()

                    if isLastEpoch:
                        # Collect features for SVM
                        [imageName, imageLabel, featureVec] = sess.run(
                            [inputBatchImageNames, inputBatchLabels, featureVector],
                            feed_dict={datasetSelectionPlaceholder: TRAIN})
                        imageNames.extend(imageName)
                        imageLabels.extend(imageLabel)
                        imageFeatures.extend(np.reshape(featureVec, [featureVec.shape[0], -1]))

                        duration = time.time() - start_time

                        # Print an overview fairly often.
                        if step % options.displayStep == 0:
                            print('Step: %d | Duration: %f' % (step, duration))
                    else:
                        # Run optimization op (backprop)
                        if options.tensorboardVisualization:
                            [trainLoss, currentAcc, _, summary] = sess.run([loss, accuracy, trainOp, mergedSummaryOp],
                                                                           feed_dict={
                                                                               datasetSelectionPlaceholder: TRAIN})
                            summaryWriter.add_summary(summary, globalStep)
                        else:
                            [trainLoss, currentAcc, _] = sess.run([loss, accuracy, trainOp],
                                                                  feed_dict={datasetSelectionPlaceholder: TRAIN})

                        duration = time.time() - start_time

                        # Print an overview fairly often.
                        if step % options.displayStep == 0:
                            print('Step: %d | Loss: %f | Accuracy: %f | Duration: %f' % (
                                step, trainLoss, currentAcc, duration))

                    step += 1
                    globalStep += 1

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (epoch, step))

        # Save final model weights to disk
        saver.save(sess, targetModelDir)
        print("Model saved: %s" % targetModelDir)

        if options.trainSVM:
            # Train the SVM
            print("Training SVM")
            imageFeatures = np.array(imageFeatures)
            imageLabels = np.array(imageLabels)
            print("Data shape: %s" % str(imageFeatures.shape))
            print("Labels shape: %s" % str(imageLabels.shape))

            clf = svm.LinearSVC(C=1.0)
            clf.fit(imageFeatures, imageLabels)
            print("Training Complete!")

            with open(os.path.join(options.modelDir, 'svm.pkl'), 'wb') as fid:
                cPickle.dump(clf, fid)

            print("Evaluating performance on training data")
            trainAccuracy = clf.score(imageFeatures, imageLabels)
            print("Train accuracy: %f" % (trainAccuracy))

    print("Optimization Finished!")

# Test model
if options.testModel:
    print("Testing saved model")

    if options.evalClasses:
        conf_matrix = np.array(np.zeros((numClasses, numClasses)))
        precision = np.array(np.zeros((numClasses)))
        recall = np.array(np.zeros((numClasses)))
        total_prec = 0
        total_rec = 0

    # Now we make sure the variable is now a constant, and that the graph still produces the expected result.
    with tf.Session(config=config) as sess:
        # Saver op to save and restore all the variables
        saver = tf.train.Saver()
        saver.restore(sess, targetModelDir)

        # Initialize the dataset iterator
        sess.run(testIterator.initializer)

        svmFound = False
        if os.path.exists(os.path.join(options.modelDir, 'svm.pkl')):
            print("Loading saved SVM instance")
            with open(os.path.join(options.modelDir, 'svm.pkl'), 'rb') as fid:
                clf = cPickle.load(fid)
                if clf is None:
                    print("Error: Unable to load SVM instance.")
                    exit(-1)
                svmFound = True
                print("SVM instance loaded successfully!")

        try:
            step = 0
            correctInstances = 0
            totalInstances = 0
            allPredictions = []
            allGTLabels = []

            if svmFound:
                correctInstancesSVM = 0
                imageLabels = []
                imageFeatures = []

            while True:
                start_time = time.time()

                [batchLabelsTest, predictions, currentAcc, featureVec] = sess.run(
                    [inputBatchImageLabels, logits, accuracy, featureVector],
                    feed_dict={datasetSelectionPlaceholder: TEST})

                predConf = np.max(predictions, axis=1)
                predClass = np.argmax(predictions, axis=1)
                actualClass = np.argmax(batchLabelsTest, axis=1)
                # Save all predictions in list
                allPredictions.extend(predClass)
                allGTLabels.extend(actualClass)

                if options.evalClasses:
                    mask = np.equal(predClass, actualClass)

                    for idx, el in enumerate(mask):
                        if el:
                            conf_matrix[actualClass[idx], actualClass[idx]] += 1
                        else:
                            conf_matrix[predClass[idx], actualClass[idx]] += 1

                correctInstances += np.sum(predClass == actualClass)
                totalInstances += predClass.shape[0]

                if svmFound:
                    imageLabels.extend(actualClass)
                    imageFeatures.extend(np.reshape(featureVec, [featureVec.shape[0], -1]))

                duration = time.time() - start_time
                if step % options.displayStep == 0:
                    print('Step: %d | Accuracy: %f | Duration: %f' % (step, currentAcc, duration))

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done testing for %d epochs, %d steps.' % (1, step))

    # Save predicted classes for each sample to file
    df = pd.Series(allPredictions)
    outFilename = os.path.join(predictionsDir, options.modelName + '.csv')
    df.to_csv(outFilename, sep="|", header=False, index=True)

    with open('./Scores.csv', 'a+') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow([options.modelName, ((float(correctInstances) / float(totalInstances)) * 100)])

    if svmFound:
        print("Evaluating SVM")
        imageFeatures = np.array(imageFeatures)
        imageLabels = np.array(imageLabels)
        print("Data shape: %s" % str(imageFeatures.shape))
        print("Labels shape: %s" % str(imageLabels.shape))

        testAccuracy = clf.score(imageFeatures, imageLabels)
        print("Test accuracy: %f" % (testAccuracy))
