<div align=center><img src="https://user-images.githubusercontent.com/36858976/175043546-a32a0c92-3797-4a4f-a87b-ec8d046dba7f.png" width=300></div>
<p align="center">
<a href="https://github.com/TensorSpeech/TensorFlowASR/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.5.1-orange?logo=tensorflow">
<h2 align="center">
<p>Audio Classification Models in Tensorflow 2.0</p>
</h2>
</p>
<p align="center">
This library utilizes some automatic speech recognition architectures such as ContextNet, Conformer, etc for audio classification.
</p>

## Kaggle Codes/Notebook
This library is used in the following notebook for **Fake Speech Detection**.
* [Fake Speech Detection: Conformer [TF]](https://www.kaggle.com/code/awsaf49/fake-speech-detection-conformer-tf) (Awarded for Google OSS Expert Award 2022)
> **Note**: You can also access the notebook in [`/notebooks`](/notebooks) folder.
  
## Installation
```shell
pip install -U audio_classification_models
```
or
```shell
pip install git+https://github.com/awsaf49/audio_classification_models
```

## Usage
```py
import audio_classification_models as acm
model = acm.Conformer(pretrain=True)
```

## Acknowledgement
* [TensorflowASR](https://github.com/TensorSpeech/TensorFlowASR)
