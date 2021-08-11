## Iterative Error Decimation

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2012.00089)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Kamassury/IED/blob/main/LICENSE)

This repository contains the codes for [*Iterative Error Decimation for Syndrome-Based Neural Network Decoders*](https://arxiv.org/abs/2012.00089) , accepted for publication in Journal of Communication and Information Systems (JCIS).

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{kamassury2021ied,
  title={Iterative Error Decimation for Syndrome-Based Neural Network Decoders},
  author={Kamassury, Jorge K S and Silva, Danilo},
  journal={Journal of Communication and Information Systems},
  year={2021}
}
```

## Files 
``main_code``:
## Training a model

In order to train a model, please use the [main_code.py](main_code.py) code. The default configuration (i.e., just running ```python main_code.py```) will train a model on the cross-entropy loss function. The following are the important parameters of the training:
```
--dataset: dataset to train on [cifar10/cifar100/tiny_imagenet]
--dataset-root: path of the Tiny ImageNet dataset (not necessary for CIFAR-10/100)
--loss: loss function of choice (cross_entropy/focal_loss/focal_loss_adaptive/mmce/mmce_weighted/brier_score)
--gamma: gamma for focal loss
--lamda: lambda value for MMCE
--gamma-schedule: whether to use a scheduled gamma during training
--save-path: path for saving models
--model: model to train (resnet50/resnet110/wide_resnet/densenet121)
```

### Pretrained models

All pre-trained models are in the folder [models](models):

