## Iterative Error Decimation

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2012.00089)
[![Tensorflow 2.4.0](https://img.shields.io/badge/tensorflow-2.4.0-orange.svg)](https://www.tensorflow.org/)
[![Keras 2.4.3](https://img.shields.io/badge/keras-2.4.3-red.svg)](https://keras.io/)
[![Python 3.6](https://img.shields.io/badge/python-3.6.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Kamassury/IED/blob/main/LICENSE)

This repository contains the codes for the paper [*Iterative Error Decimation for Syndrome-Based Neural Network Decoders*](https://arxiv.org/abs/2012.00089), accepted for publication in the Journal of Communication and Information Systems (JCIS).

**In this project, we introduce a new syndrome-based decoder where a deep neural network (DNN) estimates the error pattern from the reliability and syndrome of the received vector. The proposed algorithm works by iteratively selecting the most confident positions to be the error bits of the error pattern, updating the vector received when a new position of the error pattern is selected**.

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{kamassury_ied,
  title={Iterative Error Decimation for Syndrome-Based Neural Network Decoders},
  author={Kamassury, Jorge K S and Silva, Danilo},
  journal={Journal of Communication and Information Systems},
  year={2021}
}
```
---
## Project overview
For an overview of the project, follow the steps from the [main_code](main_code.py) module, namely:
* Get the parity check matrix (H):   ``bch_par``  
* Building the neural network: ``models_nets``
* Model training: `training_nn`
* Model inference using the IED decoder: ``BER_FER``
* Plot of inference results: ``inference``

The default configuration (using the function in [get_training_model](get_training_model.py)) will train a model with the cross entropy as the loss function. The following are the important parameters of the training 

* ``training_nn(model, H, loss, lr, batch_size, spe, epochs, EbN0_dB, tec) ``, where:
	* ``model``: neural network for short length BCH code
	* ``H``: parity check matrix
	* ``loss``: loss function (by default, binary cross entropy)
	* ``lr``:  learning rate
	* ``batch size``: batch size for training
	* ``spe``: steps per epoch
	* ``epochs``: number of epochs for training
	* ``EbN0_dB``: ratio of energy per bit to noise power spectral density
	* ``tec``: technique for changing the learning rate (`ReduceLROnPlateau` or `CyclicalLearningRate`)

---
Important routines can be found in the module [uteis](uteis.py), especially:

* ``training_generator``: simulates the transmission of codewords via the AWGN channel for model training
* ``getfer``: computes the metrics BLER, BER, ... 
* ``biawgn``: simulate codewords for inference
* ``custom_loss``: custom loss function joining binary _cross entropy_ and _loss syndrome_ 

---
### Pretrained models

All pre-trained models are in the folder [models](models), where:
* ``model_63_45``: trained model for the BCH(63, 45) code;
* `model_relu_63_36`: trained model for BCH(63, 36) code using __ReLU__ as activation function;
* `model_sigmoid_63_36`: trained model for BCH(63, 36) code using __Sigmoid__ as activation function;
*  `model_BN_sigmoid_63_36`: trained model for BH(63, 36) code using __Sigmoid__ as activation function and __batch normalization layers__.

---
### Model inference
To perform model inference for the BER and BLER metrics, use the module [ber_fer_result](ber_fer_result.py), where:

* ``max_nfe``: number of block errors
* `` T``: number of iterations using __IED__
* ``p_initial``: ``EbN0_dB`` initial value for inference
* ``p_end``: ``EbN0_dB`` final value for inference

If you just want to load the pre-trained model, perform and plot the inference, use the script [load_infer_plot](load_infer_plot.py).

---
## Result

Performances for BCH codes using the IED decoder are in the folder [results](results).

* BLER and BER for the BCH(63,45) code, respectively:
<p align="center">
	<img src="images/figure_FER_45.png" width="300"/>
	<img src="images/figure_BER_45.png" width="300"/>
</p>


* BLER and BER for the BCH(63, 36) code, respectively:
<p align="center">
	<img src="images/figure_FER_36.png" width="300"/>
	<img src="images/figure_BER_36.png" width="300"/>
</p>
