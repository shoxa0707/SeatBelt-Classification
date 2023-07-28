# SeatBelt classification

<div align="center">

**|** 👀[**Demos**](#-demos-videos) **|** ⚡[**Usage**](#-quick-inference) **|** 🔧[**Install**](#-dependencies-and-installation) **|** 💻[**Train**](#-train) **|**

</div>

## Introduction

Many of us forget or don't want to wear a seatbelt when we drive. This, in turn, is very dangerous for life. Because of this, it is necessary to check whether driver is wearing a seat belt or not. Our following project will help with this.<br>

In addition to classifying whether or not driver is wearing a seatbelt, the program also uses [image super resolution](https://github.com/xinntao/Real-ESRGAN). This, in turn, increases the accuracy of determining whether the driver has violated the rule or not.

## 👀 Demos Videos

### Streamlit

[Streamlit](https://github.com/streamlit/streamlit) lets you turn data scripts into shareable web apps in minutes, not weeks. It’s all Python, open-source, and free! And once you’ve created an app you can use our Community Cloud platform to deploy, manage, and share your app.

You can run following command to use streamlit in our case:

```bash
  streamlit run stream.py
```

The result is as follows:

<img src="images/seatbelt.gif">

## ⚡ Quick Inference

### Python script

```console
  Usage: python inference.py -t tensorflow -m models/seatbelt.model -i some.png -d cpu [options]...

  A common command: python inference.py -i some.png

    -t --type                Usage type. Options: tensorflow | pytorch. Default: tensorflow
    -m --model               Model path. Model path must match the type. Default: models/seatbelt.model
    -i --image               Image for classification. Options: auto | jpg | png.
    -d --device              Inference device. If you have GPU, you can use it. Options: cpu | gpu. Default: cpu
```

## 🔧 Dependencies and Installation

- Python >= 3.8
- PyTorch >= 2
- Tensorflow >= 2.8

  ### Installation

  1. Clone repo

     ```bash
     git clone https://github.com/shoxa0707/SeatBelt-Classification.git
     cd SeatBelt-Classification
     ```

  1. Install dependent packages

     ```bash
     # if you have a GPU, install as follows:
     pip install tensorflow-gpu==2.8
     pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118(sample for CUDA 11.8)
     pip install -r requirements.txt
     # else
     pip install tensorflow==2.8
     pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
     pip install -r requirements.txt
     ```

## 💻 Train

There are usually two ways to train classification.

1. [Tensorflow](https://www.tensorflow.org)
2. [PyTorch](https://pytorch.org/)

### Tensorflow

Tensorflow has a special place in training deep learning models and in particular in image classification. In our case, the tensorflow trained model achieved ~80% accuracy.

```console
  Usage: python train_tensorflow.py -d Data -b 16 -e 100 -s models/seatbelt.model --res [options]...

  A common command: python train_tensorflow.py -d Data -s mdoel.model --res

    -d --dataset                  Dataset path. Default: Data
    -b --batch_size               Training batch size. Default: 16
    -e --epochs                   Training epochs. Default: 100
    -s --save                     Save path trained model. Default: models/seatbelt.model
    --res                         Using resolution for dataset images.
    --no-res                      Don't using resolution for dataset images.
```

### Pytorch

PyTorch is a Python package that provides two high-level features:

- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system

In our case, the pytorch trained model achieved ~60% accuracy.

```console
  Usage: python train_pytorch.py -d Data -b 16 -e 100 -s models/seatbelt.model --res [options]...

  A common command: python train_pytorch.py -d Data -s model.pt

    -d --dataset                  Dataset path. Default: Data
    -c --device                   Training device. Options: cpu | cuda. default: cpu
    -b --batch_size               Training batch size. Default: 16
    -e --epochs                   Training epochs. Default: 100
    -s --save                     Save path trained model. Default: models/seatbelt.pt
```

Where dataset path should be:

```bash
dataset path
    -- train
        -- Aniqlanmadi
        -- Taqilgan
        -- Taqilmagan
    -- test
        -- Aniqlanmadi
        -- Taqilgan
        -- Taqilmagan
```

Our trained models are [here](https://drive.google.com/drive/folders/1cS5wglnvL42UrIckWPooUZa719kHsO8O?usp=sharing).<br>
You can use this models to classify seatbelt.

# Requirements

- Linux
- Python 3.8
- NVIDIA GPU for training
