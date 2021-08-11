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
--dataset: 
```

### Pretrained models

All pre-trained models are in the folder [models](models), where:
* ``model_63_45``: trained model for the BCH(63, 45) code;
* `model_relu_63_36`: trained model for BCH(63, 36) code using __ReLU__ as activation function;
* `model_sigmoid_63_36`: trained model for BCH(63, 36) code using __Sigmoid__ as activation function;
*  `model_BN_sigmoid_63_36`: trained model for BH(63, 36) code using __Sigmoid__ as activation function and __batch normalization layers__.


## Result

Performances for BCH codes using the IED decoder are in the folder [results](results).

<p align="center">
	<img src="images/figure_FER_45.png" width="350"/>
	<img src="images/figure_FER_36.png" width="350"/>
</p>
