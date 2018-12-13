# MXNet Workshop in Hong Kong 2018

Areas of focus:

* Natural Language Processing
* Generative Adversarial Networks
* Recommender Systems
* Reinforcement Learning

## Setup

Since this repository contains submodules use
the following command to clone:

`git clone --recurse-submodules https://github.com/Ishitori/MXNetWorkshopHongKong.git`

## Day 1: 10th December 2018

Area of focus: Get introduced to MXNet (Gluon API) on AWS using Amazon SageMaker.
Presenters: Thom, Thomas, Cyrus, Sergey

|Title	|Description	|Duration (Mins)	|Presenter	|
|---	|---	|---	|---	|
|MXNet & Gluon Overview	|An overview of MXNet architecture and components	|30	|Thom	|
|SageMaker Setup	|Get configured with SageMaker for the day ahead	|30	|Cyrus	|
|Gluon Crash Course	|Walkthrough of core Gluon components, and use them to create and train a convolutional neural network.	|180	|Thomas	|
|SageMaker Introduction	|An introduction to SageMaker SDK	|30	|Cyrus	|
|Multi GPU training	|An introduction to training using multiple GPUs in Gluon, with lab.	|60	|Sergey	|
|Multi GPU training with SageMaker	|Same as above on SageMaker	|30	|Sergey	|

## Day 2: 11th December 2018

Area of focus: Natural Language Processing examples
Presenters: Sergey, Thom and Cyrus


|Title	|Description	|Duration (Mins)	|Presenter	|
|---	|---	|---	|---	|
|Deep Learning AMI Setup	|An introduction to DLAMI and get configured for the day ahead	|30	|Thom	|
|PyCharm Setup	|Setup for PyCharm for Remote Debugging	|30	|Thom	|
|Gluon NLP	|Introduction into solving common NLP tasks using GluonNLP library	|30	|Sergey & Cyrus	|
|Stacked Bidirectional LSTM (LAB)	|We work up from plain RNN to Stacked Bi-directional LSTMs using Gluon layers	|60	|Thom	|
|Keyphrase Extraction	|Will show how to implement a model from [Bi-directional LSTM recurrent neural network for kyphrase extraction paper](https://github.com/dmlc/gluon-nlp/blob/keyphrase/scripts/keyphrase/BiLSTM_for_Keyphrase_Extraction.pdf), using Gluon components and cover how to do common NLP tasks like Data pipelining, Tokenization, Embedding, modeling using LSTM.	|60	|Sergey	|
|MXBoard	|Using MXNet with Tensorboard to monitor training	|30	|Thom	|
| Sentiment Analysis in code-switching text	|We will take part in [NLPCC 2018 Emotion Detection in Code-Switching Text](http://tcci.ccf.org.cn/conference/2018/dldoc/taskgline01.pdf) competition, dive deep into working with Embeddings, and learn how to use Convolutional encoder from [this paper](https://arxiv.org/pdf/1508.06615.pdf)	|60	|Sergey	|
|Beyond the Defaults	|Alternatives for initialization, optimization and evaluation metrics.	|30	|Cyrus	|
|Intent detection and slot filling	|We will see how to use advanced embedding like [ELMo](https://arxiv.org/abs/1802.05365) to transfer learning in NLP, do multitasking and use [Highway layer](https://arxiv.org/pdf/1505.00387.pdf) and [Conditional Random Field](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers).	|60	|Sergey	|

## Day 3: 12th December 2018

Area of focus: Recommender Systems
Presenters: Cyrus, Thom and Sergey


|Title	|Description	|Duration (Mins)	|Presenter	|
|---	|---	|---	|---	|
|Survey of Recomender Systems	|	|60	|Cyrus	|
|Profiling MXNet	|Analysis of runtime code to identify performance bottlenecks	|30	|Thom	|
|Implementation of MLP on Movie Lens	|	|30	|Cyrus	|
|MXNet Performance Tricks	|Useful tips for maximizing performance of MXNet	|60	|Sergey	|
|DSSM Theory	|	|30	|Cyrus	|
|LR Schedules	|Using custom learning rate schdules	|45	|Thom	|
|DSSM Implementation	|	|60	|Cyrus	|
|Sparse Matrix Operations	| |45	|Cyrus	|

## Day 4: 13th December 2018

Area of focus: Generative Adversarial Networks, Model Deployment and multi-host training.
Presenters: Thom, Sergey & Steve

|Title	|Description	|Duration (Mins)	|Presenter	|
|---	|---	|---	|---	|
|InfoGAN Theory	|A paper walkthrough of InfoGAN	|30	| Thom	|
|InfoGAN Implementation	|A code walkthrough of Gluon implementation of InfoGAN	|60	|Thom	|
|ECommerce GAN Theory	|A paper walkthrough of e-Commerce GAN	|30	|Sergey	|
|SageMaker Automatic Model Tuning	|A look at SageMaker's hyperparameter optimization features.	|60	|Thom	|
|Multi Host training	|Using multiple instances to speed up training (on SageMaker)	|60	|Sergey	|
|SageMaker Deployment	|Useful patterns for deployment on SageMaker.	|30	|Steve |
|MMS Deployment	|MXNet Model Server deployment on AWS Fargate	|60	|Sergey |


## Day 5: 14th December 2018

Area of focus: Reinforcement Learning
Presenters: Thom, Sergey & Steve

|Title	|Description	|Duration (Mins)	|Presenter	|
|---	|---	|---	|---	|
|re:Invent ML Announcements| Recap of new ML features	|60	|Steve	|
|AWS Neo & TVM	|An introduction to model compiliation with AWS Neo	|30	|Thom	|
|AWS Elastic Inference	|	|30	|Steve	|
|RL Introduction	|An introduction to RL theory, and SageMaker RL (frameworks)	|45	|Thom	|
|RL DQN	|An introduction to DQN with SageMaker RL Coach demo.	|45	|Thom	|
|RL PPO	|An introduction to PPO with SageMaker RL Coach demo.	|45	|Thom	|
