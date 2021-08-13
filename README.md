## Iterative Error Decimation

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2012.00089)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Kamassury/IED/blob/main/LICENSE)

This repository contains the codes for [*Iterative Error Decimation for Syndrome-Based Neural Network Decoders*](https://arxiv.org/abs/2012.00089) , accepted for publication in Journal of Communication and Information Systems (JCIS).

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{kamassury_ied,
  title={Iterative Error Decimation for Syndrome-Based Neural Network Decoders},
  author={Kamassury, Jorge K S and Silva, Danilo},
  journal={Journal of Communication and Information Systems},
  year={2021}
}
```

## Project overview
For an overview of the project, follow the steps from the [main_code](main_code) code, namely:
* Get the parity check matrix (H):   ``bch_par``  
* Building the neural network: ``models_nets``
* Model training: `training_nn`
* Model inference using the IED decoder: ``BER_FER``
* Plot of inference results: ``inference``

The default configuration (using the function [get_training_model](get_training_model)), will train a model with the cross entropy as the loss function. The following are the important parameters of the training 

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
Important routines can be found in the code [uteis](uteis), especially:
* Functions: 
	* ``training_generator``: simulates the transmission of codewords via the AWGN channel
	* ``getfer``: computes the metrics BLER, BER, ... 
	* ``biawgn``, 
	* ``getniter``, 
	* ``custom_loss``: ``syndrome_loss``, 

* Classes: ``SBND``, ``PrintFER``

### Pretrained models

All pre-trained models are in the folder [models](models), where:
* ``model_63_45``: trained model for the BCH(63, 45) code;
* `model_relu_63_36`: trained model for BCH(63, 36) code using __ReLU__ as activation function;
* `model_sigmoid_63_36`: trained model for BCH(63, 36) code using __Sigmoid__ as activation function;
*  `model_BN_sigmoid_63_36`: trained model for BH(63, 36) code using __Sigmoid__ as activation function and __batch normalization layers__.

### Model inference
To perform model inference for the BER and BLER metrics, use the code [ber_fer_result](ber_fer_result), where:

* ``max_nfe``: number of block errors
* `` T``: number of iterations using __IED__
* ``p_initial``: ``EbN0_dB`` initial value for inference
* ``p_end``: ``EbN0_dB`` final value for inference

If you just want to load the pre-trained model, perform and plot the inference, use the code [load_infer_plot](load_infer_plot).

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
