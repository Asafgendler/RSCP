# general imports
import sys
import argparse
import random
import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import torch
import torchvision
import os
import shutil
import pickle
from torch.utils.data.dataset import random_split
import pandas as pd
from sklearn.model_selection import train_test_split

# My imports
sys.path.insert(0, './')
from Third_Party.smoothing_adversarial.architectures import get_architecture
import RSCP.Score_Functions as scores
from RSCP.utils import Smooth_Adv, evaluate_predictions, calculate_accuracy_smooth, \
    smooth_calibration, predict_sets, get_normalize_layer, NormalizeLayer
from Architectures.DenseNet import DenseNet
from Architectures.VGG import vgg19_bn, VGG
from Architectures.ResNet import ResNet

# parameters
parser = argparse.ArgumentParser(description='Experiments on CIFAR10 and CIFAR100')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-d', '--delta', default=0.125, type=float, help='L2 bound on the adversarial noise')
parser.add_argument('-s', '--splits', default=50, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('-r', '--ratio', default=2, type=float,
                    help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('--n_s', default=256, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used: CIFAR100, CIFAR10')
parser.add_argument('--arc', default='ResNet', type=str,
                    help='Architecture of classifier : ResNet, Densenet, VGG. Relevant only of My_model=True')
parser.add_argument('--My_model', dest='My_model', action='store_true', help='True for our trained model, False for Cohens. Relevent only for CIFAR10')
parser.add_argument('--no-My_model', dest='My_model', action='store_false')
parser.set_defaults(My_model=True)
parser.add_argument('--batch_size', default=1024, type=int, help='Number of images to send to gpu at once')

args = parser.parse_args()
alpha = args.alpha
epsilon = args.delta
n_experiments = args.splits
n_test = 10000  # number of test points (if larger then available it takes the entire set)
ratio = args.ratio
sigma_smooth = ratio * epsilon  # sigma used for smoothing
sigma_model = sigma_smooth  # sigma used for training the model
n_smooth = args.n_s
My_model = args.My_model
N_steps = 20  # number of gradiant steps for PGD attack
dataset = args.dataset
calibration_scores = ['HCC', 'SC']  # score function to check 'HCC', 'SC', 'SC_Reg'
model_type = args.arc
Salman = False

# Validate parameters
assert dataset == 'CIFAR10' or args.dataset == 'CIFAR100', 'Dataset can only be CIFAR10 or CIFAR100.'
assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'
assert not(n_smooth & (n_smooth - 1)), 'n_s must be a power of 2.'
assert not(args.batch_size & (args.batch_size - 1)), 'batch size must be a power of 2.'
assert args.batch_size >= n_smooth, 'batch size must be larger than n_s'
assert model_type == 'ResNet' or model_type == 'DenseNet' or model_type == 'VGG', 'Architecture can only be Resnet, ' \
                                                                                  'VGG or DenseNet '

# CIFAR100 has only my models
if dataset == "CIFAR100":
    My_model = True

# All our models are needs to be added a normalization layer
if My_model:
    normalized = True

# Cohen models already have this layer, Plus only ResNet is available for them.
else:
    normalized = False
    model_type = 'ResNet'

# The GPU used for oue experiments can only handle the following quantities of images per batch
GPU_CAPACITY = args.batch_size

# Save results to final results directories only if full data is taken. Otherwise save locally.
if (n_experiments == 50) and (n_test == 10000):
    save_results = True
else:
    save_results = False

# calculate correction based on the Lipschitz constant
if sigma_smooth == 0:
    correction = 10000
else:
    correction = float(epsilon) / float(sigma_smooth)

# set random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load datasets
if dataset == "CIFAR10":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR10(root='./datasets/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())

elif dataset == "CIFAR100":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR100(root='./datasets/',
                                                  train=True,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                                 train=False,
                                                 transform=torchvision.transforms.ToTensor())

else:
    print("No such dataset")
    exit(1)

# cut the size of the test set if necessary
if n_test < len(test_dataset):
    test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

# save the sizes of each one of the sets
n_train = len(train_dataset)
n_test = len(test_dataset)

# Create Data loader for test set
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=n_test,
                                          shuffle=False)

# convert test set into tensor
examples = enumerate(test_loader)
batch_idx, (x_test, y_test) = next(examples)

# get dimension of data
rows = x_test.size()[2]
cols = x_test.size()[3]
channels = x_test.size()[1]
num_of_classes = len(train_dataset.classes)
min_pixel_value = 0.0
max_pixel_value = 1.0

# automatically choose device use gpu 0 if it is available o.w. use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print the chosen device
print("device: ", device)

# loading a pre-trained model

# load my models
if My_model:
    if dataset == "CIFAR10":
        if model_type == 'ResNet':
            model = ResNet(depth=110, num_classes=10)
            state = torch.load('./checkpoints/CIFAR10_ResNet110_Robust_sigma_' + str(sigma_model) + '.pth.tar',
                               map_location=device)
        elif model_type == 'DenseNet':
            model = DenseNet(depth=100, num_classes=10, growthRate=12)
            state = torch.load('./checkpoints/CIFAR10_DenseNet_sigma_' + str(sigma_model) + '.pth.tar',
                               map_location=device)
        elif model_type == 'VGG':
            model = vgg19_bn(num_classes=10)
            state = torch.load('./checkpoints/CIFAR10_VGG_sigma_' + str(sigma_model) + '.pth.tar', map_location=device)
        else:
            print("No such architecture")
            exit(1)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])
    elif dataset == "CIFAR100":
        if model_type == 'ResNet':
            model = ResNet(depth=110, num_classes=100)
            state = torch.load('./checkpoints/ResNet110_Robust_sigma_' + str(sigma_model) + '.pth.tar',
                               map_location=device)
        elif model_type == 'DenseNet':
            model = DenseNet(depth=100, num_classes=100, growthRate=12)
            state = torch.load('./checkpoints/DenseNet_sigma_' + str(sigma_model) + '.pth.tar', map_location=device)
        elif model_type == 'VGG':
            model = vgg19_bn(num_classes=100)
            state = torch.load('./checkpoints/VGG_sigma_' + str(sigma_model) + '.pth.tar', map_location=device)
        else:
            print("No such architecture")
            exit(1)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])
    else:
        print("No such dataset")
        exit(1)
# load cohen and salman models
else:
    # checkpoint = torch.load(
    #   './Pretrained_Models/Salman/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar')
    if dataset == "CIFAR10":
        if Salman:
            checkpoint = torch.load(
                './Pretrained_Models/Salman/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar', map_location=device)
        else:
            checkpoint = torch.load(
                './Pretrained_Models/Cohen/cifar10/resnet110/noise_' + str(sigma_model) + '/checkpoint.pth.tar', map_location=device)
        model = get_architecture(checkpoint["arch"], "cifar10")
    else:
        print("No Cohens model for CIFAR100")
        exit(1)
    model.load_state_dict(checkpoint['state_dict'])

# send model to device
model.to(device)

# put model in evaluation mode
model.eval()

# create indices for the test points
indices = torch.arange(n_test)

# directory to store adversarial examples and noises
directory = "./Adversarial_Examples/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
    sigma_model)

if sigma_smooth != sigma_model:
    directory = directory + "/sigma_smooth_" + str(sigma_smooth)

directory = directory + "/n_smooth_" + str(n_smooth)

if normalized:
    directory = directory + "/Robust"

if model_type != 'ResNet':
    directory = directory + "/" + str(model_type)

if dataset == "CIFAR10" and model_type == 'ResNet':
    if My_model:
        directory = directory + "/My_Model"
    else:
        directory = directory + "/Their_Model"
        if Salman:
            directory = directory + "/Salman"

print("Save results on directories: " + str(save_results))
print("Searching for adversarial examples in: " + str(directory))
if os.path.exists(directory):
    print("Are there saved adversarial examples: Yes")
else:
    print("Are there saved adversarial examples: No")

# If there are no pre created adversarial examples, create new ones
if not os.path.exists(directory) or not n_test == 10000:

    # create noises for each data point
    noises_test = torch.randn((n_test * n_smooth, channels, rows, cols)) * sigma_smooth

    # create noises for the base classifier
    noises_test_base = torch.randn((n_test, channels, rows, cols)) * sigma_model

    # Generate adversarial test examples
    x_test_adv = Smooth_Adv(model, x_test, y_test, noises_test, N_steps, epsilon, device, GPU_CAPACITY=GPU_CAPACITY)

    # Generate adversarial test examples for the base classifier
    x_test_adv_base = Smooth_Adv(model, x_test, y_test, noises_test_base, N_steps, epsilon, device,
                                 GPU_CAPACITY=GPU_CAPACITY)

    # Only store examples for full dataset
    if n_test == 10000:
        os.makedirs(directory)
        with open(directory + "/data.pickle", 'wb') as f:
            pickle.dump([x_test_adv, x_test_adv_base, noises_test, noises_test_base], f)

# If there are pre created adversarial examples, load them
else:
    with open(directory + "/data.pickle", 'rb') as f:
        x_test_adv, x_test_adv_base, noises_test, noises_test_base = pickle.load(f)

# Check noises std is correct
if torch.abs(torch.std(noises_test) - sigma_smooth) > 0.01:
    print("Mismatch in std")
    if sigma_smooth == sigma_model:
        shutil.rmtree(directory)
    exit(1)

# Check noises std is correct
if torch.abs(torch.std(noises_test_base) - sigma_model) > 0.01:
    print("Mismatch in std")
    exit(1)

# translate desired scores to their functions and put in a list
scores_list = []
for score in calibration_scores:
    if score == 'HCC':
        scores_list.append(scores.class_probability_score)
    elif score == 'SC':
        scores_list.append(scores.generalized_inverse_quantile_score)
    elif score == 'SC_Reg':
        scores_list.append(scores.rank_regularized_score)
    else:
        print("Undefined score function")
        exit(1)

# Calculate accuracy of classifier on clean test points
acc, _, _ = calculate_accuracy_smooth(model, x_test, y_test, noises_test_base, num_of_classes, k=1, device=device,
                                      GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy :" + str(acc * 100) + "%")

# Calculate accuracy of classifier on adversarial test points
acc, _, _ = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_test_base, num_of_classes, k=1,
                                      device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy on adversarial examples :" + str(acc * 100) + "%")
exit(1)

# # generate prediction sets on the clean test set
# smoothed_scores_clean = predict_sets(model, x_test, noises_test, num_of_classes, scores_list, "None",
#                                     correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)
#
# # generate prediction sets on the clean test set for base classifier
# scores_clean = predict_sets(model, x_test, noises_test_base, num_of_classes, scores_list,
#                                          "None", correction, base=True, device=device,
#                                          GPU_CAPACITY=GPU_CAPACITY)
#
# # generate prediction sets on the adversarial test set
# smoothed_scores_adv = predict_sets(model, x_test_adv, noises_test, num_of_classes, scores_list, "None",
#                                   correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)
#
# # generate prediction sets on the adversarial test set for base classifier
# scores_adv = predict_sets(model, x_test_adv_base, noises_test_base, num_of_classes,
#                                        scores_list, "None", correction, base=True, device=device,
#                                        GPU_CAPACITY=GPU_CAPACITY)
# one_hot_lables = np.zeros((n_test,num_of_classes))
# flattend_input = np.zeros((n_test,(x_test[0].flatten()).size()[0]))
# flattend_input_adv = np.zeros((n_test,(x_test_adv[0].flatten()).size()[0]))
# for i in range(n_test):
#     one_hot_lables[i, y_test[i]] = 1
#     flattend_input[i, :] = np.array(x_test[i].flatten())
#     flattend_input_adv[i, :] = np.array(x_test_adv[i].flatten())
# dict_to_save_clean = {'labels': one_hot_lables, 'scores_HPS': scores_clean[0],'scores_APS': scores_clean[1],'scores_smoothed_HPS': smoothed_scores_clean[0],'scores_smoothed_APS': smoothed_scores_clean[1], 'features': flattend_input}
# dict_to_save_adv = {'labels': one_hot_lables, 'scores_HPS': scores_adv[0],'scores_APS': scores_adv[1],'scores_smoothed_HPS': smoothed_scores_adv[0],'scores_smoothed_APS': smoothed_scores_adv[1], 'features': flattend_input_adv}
#
#
# np.save("regular_model_clean_test", dict_to_save_clean)
# np.save("regular_model_adv_test", dict_to_save_adv)
# exit(1)

# create dataframe for storing results
results = pd.DataFrame()

# container for storing bounds on "CP+SS"
quantiles = np.zeros((len(scores_list), 2, n_experiments))

# run for n_experiments data splittings
for experiment in tqdm(range(n_experiments)):

    # Split test data into calibration and test
    x_calib, x_test_new, y_calib, y_test_new, idx1, idx2 = train_test_split(x_test, y_test, indices, test_size=0.5)

    # save sizes of calibration and test sets
    n_calib = x_calib.size()[0]
    n_test_new = x_test_new.size()[0]

    # get the relevant noises for the calibration and test sets
    noises_calib = torch.zeros((n_calib * n_smooth, channels, rows, cols))
    noises_calib_base = noises_test_base[idx1]
    noises_test_new = torch.zeros((n_test_new * n_smooth, channels, rows, cols))
    noises_test_new_base = noises_test_base[idx2]

    for j, m in enumerate(idx1):
        noises_calib[(j * n_smooth):((j + 1) * n_smooth), :, :, :] = noises_test[(m * n_smooth):((m + 1) * n_smooth), :,
                                                                     :, :]

    for j, m in enumerate(idx2):
        noises_test_new[(j * n_smooth):((j + 1) * n_smooth), :, :, :] = noises_test[(m * n_smooth):((m + 1) * n_smooth),
                                                                        :, :, :]

    # get the relevant adversarial examples for the new test set
    x_test_adv_new = x_test_adv[idx2]
    x_test_adv_new_base = x_test_adv_base[idx2]

    # calibrate the model with the desired scores and get the thresholds
    thresholds, bounds = smooth_calibration(model, x_calib, y_calib, noises_calib, alpha, num_of_classes, scores_list,
                                            correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # calibrate base model with the desired scores and get the thresholds
    thresholds_base, _ = smooth_calibration(model, x_calib, y_calib, noises_calib_base, alpha, num_of_classes,
                                            scores_list, correction, base=True, device=device,
                                            GPU_CAPACITY=GPU_CAPACITY)

    thresholds = thresholds + thresholds_base

    # put bounds in array of bounds
    for p in range(len(scores_list)):
        quantiles[p, 0, experiment] = bounds[p, 0]
        quantiles[p, 1, experiment] = bounds[p, 1]

    # generate prediction sets on the clean test set
    predicted_clean_sets = predict_sets(model, x_test_new, noises_test_new, num_of_classes, scores_list, thresholds,
                                        correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # generate prediction sets on the clean test set for base classifier
    predicted_clean_sets_base = predict_sets(model, x_test_new, noises_test_new_base, num_of_classes, scores_list,
                                             thresholds, correction, base=True, device=device,
                                             GPU_CAPACITY=GPU_CAPACITY)

    # generate prediction sets on the adversarial test set
    predicted_adv_sets = predict_sets(model, x_test_adv_new, noises_test_new, num_of_classes, scores_list, thresholds,
                                      correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # generate prediction sets on the adversarial test set for base classifier
    predicted_adv_sets_base = predict_sets(model, x_test_adv_new_base, noises_test_new_base, num_of_classes,
                                           scores_list, thresholds, correction, base=True, device=device,
                                           GPU_CAPACITY=GPU_CAPACITY)

    # arrange results on clean test set in dataframe
    for p in range(len(scores_list)):
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
        score_name = calibration_scores[p]
        methods_list = [score_name + '_simple', score_name + '_smoothed_classifier', score_name + '_smoothed_score',
                        score_name + '_smoothed_score_correction']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_clean_sets[p][r], x_test_new.numpy(), y_test_new.numpy(),
                                       conditional=False)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = 0
            res['Black box'] = 'CNN sigma = ' + str(sigma_model)
            # Add results to the list
            results = results.append(res)

    # arrange results on adversarial test set in dataframe
    for p in range(len(scores_list)):
        score_name = calibration_scores[p]
        methods_list = [score_name + '_simple', score_name + '_smoothed_classifier', score_name + '_smoothed_score',
                        score_name + '_smoothed_score_correction']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_adv_sets[p][r], x_test_new.numpy(), y_test_new.numpy(),
                                       conditional=False)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = epsilon
            res['Black box'] = 'CNN sigma = ' + str(sigma_model)
            # Add results to the list
            results = results.append(res)

    # clean memory
    del noises_calib
    del noises_test_new
    gc.collect()

# directory to save results
if save_results:
    directory = "./Results/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
        sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)

    if normalized:
        directory = directory + "/Robust"

    if model_type != 'ResNet':
        directory = directory + "/" + str(model_type)

    if dataset == "CIFAR10" and model_type == 'ResNet':
        if My_model:
            directory = directory + "/My_Model"
        else:
            directory = directory + "/Their_Model"
            if Salman:
                directory = directory + "/Salman"

    for score in calibration_scores:
        if score == 'SC_Reg':
            directory = directory + "/Regularization"
            break

    if alpha != 0.1:
        directory = directory + "/alpha_" + str(alpha)

    print("Saving results in: " + str(directory))

    if not os.path.exists(directory):
        os.makedirs(directory)

    # save results
    results.to_csv(directory + "/results.csv")
    with open(directory + "/quantiles_bounds.pickle", 'wb') as f:
        pickle.dump([quantiles], f)
else:
    # save results
    print("Saving results in main Results folder")
    results.to_csv("./Results/results.csv")

# plot results
# plot marginal coverage results
colors_list = sns.color_palette("husl", len(scores_list) * 4)

ax = sns.catplot(x="Black box", y="Coverage",
                 hue="Method", palette=colors_list, col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)

lower_quantiles_mean = np.zeros(len(scores_list))
upper_quantiles_mean = np.zeros(len(scores_list))
lower_quantiles_std = np.zeros(len(scores_list))
upper_quantiles_std = np.zeros(len(scores_list))

for p in range(len(scores_list)):
    lower_quantiles_mean[p] = np.mean(quantiles[p, 0, :])
    upper_quantiles_mean[p] = np.mean(quantiles[p, 1, :])
    lower_quantiles_std[p] = np.std(quantiles[p, 0, :])
    upper_quantiles_std[p] = np.std(quantiles[p, 1, :])

colors = ['green', 'blue']
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Marginal coverage')
    graph.axhline(1 - alpha, ls='--', color="red")
    for p in range(len(scores_list)):
        graph.axhline(upper_quantiles_mean[p], ls='--', color=colors_list[p * 4 + 2])
        graph.axhline(lower_quantiles_mean[p], ls='--', color=colors_list[p * 4 + 2])

if save_results:
    ax.savefig(directory + "/Marginal.pdf")
else:
    ax.savefig("./Results/Marginal.pdf")

# plot set sizes results
ax = sns.catplot(x="Black box", y="Size",
                 hue="Method", col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Set Size')

if save_results:
    ax.savefig(directory + "/Size.pdf")
else:
    ax.savefig("./Results/Size.pdf")
