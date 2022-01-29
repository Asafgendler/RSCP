#!/bin/bash
python ./Create_Figures/Motivation.py
python ./Create_Figures/Trivial_Sets.py
python ./Create_Figures/Main_Results.py --no-My_CIFAR10
python ./Create_Figures/Effect_of_sigma_smooth.py --comparison
python ./Create_Figures/Effect_of_sigma_smooth.py --no-comparison --dataset CIFAR100
python ./Create_Figures/Effect_of_n_smooth.py --comparison
python ./Create_Figures/Effect_of_n_smooth.py --no-comparison --dataset CIFAR10
python ./Create_Figures/DenseNet_Results.py --dataset CIFAR100
python ./Create_Figures/DenseNet_Results.py --dataset CIFAR10

