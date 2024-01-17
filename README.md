## SCHEMA: State CHangEs MAtter for Procedure Planning in Instructional Videos

[Yulei Niu](https://yuleiniu.github.io/), [Wenliang Guo](https://wenliangguo.github.io/), [Long Chen](https://zjuchenlong.github.io/), [Xudong Lin](https://xudonglinthu.github.io/), [Shih-fu Chang](https://www.ee.columbia.edu/~sfchang/)

![Network Architecture](assets/structure.png)

**Abstract**: We study the problem of procedure planning in instructional videos, which aims to make a goal-oriented sequence of action steps given partial visual state observations. The motivation of this problem is to learn a structured and plannable state and action space. Recent works succeeded in sequence modeling of steps with only sequence-level annotations accessible during training, which overlooked the roles of states in the procedures. In this work, we point out that State CHangEs MAtter (SCHEMA) for procedure planning in instructional videos. We aim to establish a more structured state space by investigating the causal relations between steps and states in procedures. Specifically, we explicitly represent each step as state changes and track the state changes in procedures. For step representation, we leveraged the commonsense knowledge in large language models (LLMs) to describe the state changes of steps via our designed chain-of-thought prompting. For state changes tracking, we align visual state observations with language state descriptions via cross-modal contrastive learning, and explicitly model the intermediate states of the procedure using LLM-generated state descriptions. Experiments on CrossTask, COIN, and NIV benchmark datasets demonstrate that our proposed SCHEMA model achieves state-of-the-art performance and obtains explainable visualizations.

## Environment Setup
python = 3.10.9  
pytorch = 1.13.1  
cuda = 11.6  
numpy = 1.23.5  
tensorboard = 2.12.2  
timm = 0.6.13  
torchvision = 0.14.1

## Data Preparation
### Dataset Info
|  Dataset  | Actions | Tasks | Observation Dim |
|:---------:|:-------:|:-----:|:---------------:|
| CrossTask |   133   |   18  |       512       |
|    COIN   |   778   |  180  |       512       |
|    NIV    |    48   |   5   |       512       |
### CrossTask
```
wget https://vision.eecs.yorku.ca/WebShare/CrossTask_s3d.zip  
unzip CrossTask_s3d.zip  
```
### COIN
```
wget https://vision.eecs.yorku.ca/WebShare/COIN_s3d.zip
unzip COIN_s3d.zip
``` 
### NIV
```
wget https://vision.eecs.yorku.ca/WebShare/NIV_s3d.zip
unzip NIV_s3d.zip
```

## Train
```
bash script/run_{DATASET}.sh
```
DATASET: crosstask or coin or niv

## Evaluation
```
bash script/eval_{DATASET}.sh
```
DATASET: crosstask or coin or niv

|                 | Success Rate | Accuracy |  MIoU  |
|:---------------:|:------------:|:--------:|:------:|
| CrossTask (T=3) |    31.83%    |  57.31%  | 78.33% |
| CrossTask (T=4) |    20.18%    |  51.86%  | 74.45% |
|    COIN (T=3)   |    32.26%    |  49.98%  | 83.93% |
|    COIN (T=4)   |    21.89%    |  45.25%  | 83.51% |
|    NIV (T=3)    |    27.93%    |  41.64%  | 76.77% |
|    NIV (T=4)    |    23.26%    |  39.93%  | 76.75% |