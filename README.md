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
```

### Pretrained models

All pre-trained models are in the folder [models](models), where:
* ``model_63_45``: trained model for the BCH(63, 45) code;
* `model_relu_63_36`:   


## Result

To plot the ROC curve and compute the AUROC for a model trained on CIFAR-10 (in-distribution dataset) and tested on SVHN (out-of-distribution dataset), please use the [inference_plots.py](inference_plots.py) notebook. 

<p align="center">
	<img src="roc.png" width="500" />
</p>
