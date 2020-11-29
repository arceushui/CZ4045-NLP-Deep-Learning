# CZ4045-NLP-Deep-Learning

# Training Codes
## Question 1
The FNN directory has all python scripts for the assignment. 
```
$ cd FNN
```
Contents:
* data.py
* FNN.py
* generate.py
* model.py
* test.py
* FNNresults
* model

## Question 2
The training is done using Jupyter Notebook. The filename is listed below.

`Question_2/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb`

# Python Version
* Python 3.7

# Results directory
The training and validation training results are saved in FNN/FNNresults as default.
The weights of the models are saved in FNN/model as default. It only contains the result of Question 1.

# Installation Guide
Make sure you have the libraries below installed in your machine.
* pytorch
* matplotlib
* numpy

Use below command to install the required library.
```
pip install [library name]
```

For pytorch, follow the instructions in the link below to complete the installion.

https://pytorch.org/get-started/locally/

# Dataset directories
## 1. Word-based model
The FNN directory has the script for data.

`FNN/data.py`

## 2. Named Entity Recognition
There are 2 required directories listed below:
* Question_2/data
  * eng.testa
  * eng.testb
  * eng.train
  * eng.train54019
  * glove.6B.100d.txt
    * Download GloVe vectors and extract glove.6B.100d.txt into "Question_2/data" folder.
    ```
    wget http://nlp.stanford.edu/data/glove.6B.zip
    ```
  * mapping.pkl
* Question_2/models
  * pre-trained-model
  * self-trained-model

# Usage Guide
## 1. Word-based model
Change to the working directory.
```
$ cd FNN
```
Start running using the following command

```
$ python FNN.py
```

You can specify additional hyperparameters of the model by adding --{parameter} after FNN.py.

Example:
```
$ python FNN.py --emsize 200 --nhid 250 --cuda --dropout 0.5
```

To display description of each parameters, type this in the command.
```
$ python FNN.py --help
```

## 2. Named Entity Recognition
We provide the options to change the value of parameters of Word CNN in the `Define constants and parameters` block. Below is the example of parameters in the code.
```python
parameters['dropout'] = 0.1 # Dropout value
parameters['layers'] = 3  # number of CNN layers
parameters['batch_norm'] = True # use of batch normalization. True when it is used.
parameters['max_pooling'] = True # use of max pooling. True when it is used.
```
Run all the code in the notebook.

Important outputs are under the `Training Step` block.

Outputs Explanation (Example):
* Train: new_F: 0.8630311400763935 best_F: 0.8807954224049779 => the best F1 score and new F1 score for training data
* Dev: new_F: 0.7551980833404637 best_F: 0.7975013014055179 => the best F1 score and new F1 score for validation data
* Test: new_F: 0.6472667935817242 best_F: 0.6999357857077333  => the best F1 score and new F1 score for test data
* Graph: This is the graph generated in the final output. It illustrates the loss in different iterations.

We evaluate the best F1 score in our report.

# Tasks outputs
## 1.

## 2. Named Entity Recognition 
The output are described in the Jupyter notebook.
