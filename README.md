# RSCP (Randomly Smoothed Conformal Prediction)
This repository contains the code and models necessary to replicate the results of our recent paper:
**Adversarially Robust Conformal Prediction** <br>

## Contents
The major content of our repo are:
 - `RSCP/` The main folder containing the python scripts for running the experiments.
 - `third_party/` Third-party python scripts imported. Specifically we make use of the SMOOTHADV attack by [Salman et al (2019)](https://github.com/Hadisalman/smoothing-adversarial)
 - `Create_Figures/` Python scripts for creating all the figures in the paper. The /Create_Figures/Figures subfolder contains the figures themselves.
 - `Arcitectures/` Architectures for our trained models.
 - `Pretrained models/` Cohen pretrained models. [Cohen et al (2019)](https://github.com/locuslab/smoothing)
 - `checkpoints/` Our pre trained models.
 - `datasets/` A folder that contains the datasets used in our experiments CIFAR10, CIFAR100, Imagenet.
 - `Results/` A folder that contains different csv files from different experiments, used to generate the results in the paper.

RSCP folder contains:

1. `RSCP_exp.py`: the main code for running experiments.
2. `Score_Functions.py`: containing all non-conformity scores used.
3. `utills.py`: calibration and predictions functions, as well as other function used in the main code.

## Prerequisites

Prerequisites for running our code:
 - numpy
 - scipy
 - sklearn
 - torch
 - tqdm
 - seaborn
 - torchvision
 - pandas
 - plotnine
 
## Running instructions
1.  Install dependencies:
```
conda create -n RSCP python=3.8
conda activate RSCP
conda install -c conda-forge numpy
conda install -c conda-forge scipy
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge seaborn
conda install -c conda-forge pandas
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge plotnine
```
2. 
   1. Download our trained models from [here](https://drive.google.com/file/d/1NY25J5lVGyR583J4iUFKrZP3OpfcjDmw/view?usp=sharing) and extract them to Project_RSCP/checkpoints/.
   2. Download cohen models from [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view) and extract them to Project_RSCP/Pretrained_Models/. Change the name of "models" folder to "Cohen".
   3. If you want to run ImageNet experiments, obtain a copy of ImageNet ILSVRC2012 validation set and preprocess the val directory by running [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Put the created folders in Project_RSCP/datasets/imagenet/. 
   4. Optional: download our pre created adversarial examples from [here](https://technionmail-my.sharepoint.com/:f:/g/personal/asafgendler_campus_technion_ac_il/Es1JTaMEdMZEhG480b_qjcYBo6znBVS5FKrOewMjVw0NNw?e=hcbkag) and extract them to Project_RSCP/Adversarial_Examples/.

3. The current working directory when running the scripts should be the top folder RSCP.


To reproduce the results needed to create Figure 5 of the main paper for example run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 64 --batch_size 512 --dataset ImageNet --arc ResNet

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 0.0 --n_s 1  --batch_size 512 --dataset ImageNet --arc ResNet
```


To reproduce the results needed to create Figure 3 of the main paper run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.5 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 1 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 3 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 4 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 6 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 8 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
```


To reproduce the results needed to create Figure 4 of the main paper run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 2 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 4 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 8 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 16 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 32 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 64 --batch_size 1024 --dataset CIFAR10 --arc ResNet 
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 128 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 512 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 1024 --batch_size 1024 --dataset CIFAR10 --arc ResNet
```


To reproduce the results needed to create Figure 1 of the main paper run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc VGG --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc VGG --My_model
```


To reproduce the results needed to create Figure S6 of the Supplementary Material run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 4 --sigma_model 0.0 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc ResNet
```


To reproduce the results needed to create Figure S7 of the Supplementary Material run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.5 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 1 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 4 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 8 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.5 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 1 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 3 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 4 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 6 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 8 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 1 --n_s 64 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 64 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 4 --n_s 64 --batch_size 512 --dataset ImageNet --arc ResNet
```
note: for the first experiment you will need to train a ResNet-110 model for CIFAR10 with Gussian noise of standard deviation 0.0625 using [Cohen et al (2019)](https://github.com/locuslab/smoothing/blob/master/experiments.MD) code.

Then put the model in ./Pretrained_Models/Cohen/cifar10/resnet110/noise_0.0625/

Or you can download the model we allready trained from [here](https://drive.google.com/file/d/1_6nWqsXL-A73jlwBNRYUCaP98QLbg2jD/view?usp=sharing).


To reproduce the results needed to create Figure S8 of the Supplementary Material run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 2 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 4 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 8 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 16 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 32 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 64 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 128 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 512 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 1024 --batch_size 1024 --dataset CIFAR10 --arc ResNet

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 1 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 2 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 4 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 8 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 16 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 32 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 64 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 128 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 512 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 1024 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 1 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 2 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 4 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 8 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 16 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 32 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 64 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 128 --batch_size 512 --dataset ImageNet --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 256 --batch_size 512 --dataset ImageNet --arc ResNet
```


To reproduce the results needed to create Figure S9 of the Supplementary Material run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc VGG --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc VGG --My_model

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 256 --dataset CIFAR10 --arc DenseNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 256 --dataset CIFAR10 --arc DenseNet --My_model

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc ResNet --My_model

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc VGG --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR100 --arc VGG --My_model

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 256 --dataset CIFAR100 --arc DenseNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 256 --dataset CIFAR100 --arc DenseNet --My_model

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
```


To reproduce the results needed to create Figure S10 of the Supplementary Material you simply need the results from the experiments used to create Figure 5 of the main paper.
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 2 --n_s 64 --batch_size 512 --dataset ImageNet --arc ResNet

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR100 --arc ResNet --My_model
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.25 -s 50 -r 0.0 --n_s 1  --batch_size 512 --dataset ImageNet --arc ResNet
```

To reproduce the results needed to create Figure S11 of the Supplementary Material you will need to train a ResNet-110 model for CIFAR10 through adversarial training with Gussian noise of standard deviation 0.25 and eps=32 using [Salman et al (2019)](https://github.com/Hadisalman/smoothing-adversarial) code.

Then put the model in ./Pretrained_Models/Salman/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_32/cifar10/resnet110/noise_0.25/

Or you can download the model we allready trained from [here](https://drive.google.com/file/d/1AspyDLKq11PcYLXFNGae7HeABgf31zmJ/view?usp=sharing).
Then run:
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet --Salman

python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet

```


To reproduce the results needed to create Figure S12 of the Supplementary Material run
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet --coverage_on_label
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.0 --n_s 1 --batch_size 1024 --dataset CIFAR10 --arc ResNet --coverage_on_label
```


To reproduce the results needed to create Figure S13 of the Supplementary Material run
```
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --sigma_model 0.0 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 0.5 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 1 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
python ./RSCP/RSCP_exp.py -a 0.1 -d 0.125 -s 50 -r 2 --n_s 256 --batch_size 1024 --dataset CIFAR10 --arc ResNet
```

The Results will replace the current results in the RSCP/Results folder

If your CPU is out of memory, reduce the value of n_s from 256 and 64 to smaller powers of 2: 128/64/32/16/8/4/2/1.

If your GPU is out of memory, reduce the value of batch_size from 1024 and 512 to smaller powers of 2: 512/256/128/64/32/16/8/4/2/1.

Always make sure that batch_size is greater than n_s though.

To generate all the figures for the papers run:
```
python ./Create_all_figures.py
```
The figures will appear in RSCP/Create_Figures/Figures
