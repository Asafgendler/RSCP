# general imports
import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import os
import pickle
import sys
import argparse
from torchvision import transforms, datasets
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import pandas as pd

# My imports
sys.path.insert(0, './')
from Third_Party.smoothing_adversarial.architectures import get_architecture
import RSCP.Score_Functions as scores
from RSCP.utils import evaluate_predictions,calculate_accuracy_smooth, \
    smooth_calibration_ImageNet, predict_sets_ImageNet, Smooth_Adv_ImageNet


# parameters
parser = argparse.ArgumentParser(description='Experiments on ImageNet')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-d', '--delta', default=0.25, type=float, help='L2 bound on the adversarial noise')
parser.add_argument('-s', '--splits', default=20, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('-r', '--ratio', default=2, type=float,
                    help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('--n_s', default=64, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--batch_size', default=64, type=int, help='Number of images to send to gpu at once')

args = parser.parse_args()

# parameters
alpha = args.alpha  # desired nominal marginal coverage
epsilon = args.delta  # L2 bound on the adversarial noise
n_experiments = args.splits  # number of experiments to estimate coverage
n_test = 10000  # number of test points (if larger then available it takes the entire set)
ratio = args.ratio  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon # sigma used fro smoothing
sigma_model = sigma_smooth  # sigma used for training the model
n_smooth = args.n_s  # number of samples used for smoothing
N_steps = 20  # number of gradiant steps for PGD attack
dataset = "ImageNet"  # dataset to be used 'MNIST', 'CIFAR100', 'CIFAR10', 'ImageNet'
calibration_scores = ['HCC', 'SC']  # score function to check 'HCC', 'SC', 'SC_Reg'

# Validate parameters
assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'
assert not(n_smooth & (n_smooth - 1)), 'n_s must be a power of 2.'
assert not(args.batch_size & (args.batch_size - 1)), 'batch size must be a power of 2.'
assert args.batch_size >= n_smooth, 'batch size must be larger than n_s'

# The GPU used for oue experiments can only handle the following quantities of images per batch
GPU_CAPACITY = args.batch_size

# Save results to final results directories only if full data is taken. Otherwise save locally.
if (n_experiments == 20) and (n_test == 10000):
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

# get dir of imagenet validation set
imagenet_dir = "./datasets/imagenet"

# ImageNet images pre-processing
transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])

# load dataset
test_dataset = datasets.ImageFolder(imagenet_dir, transform)

# cut the size of the test set if necessary
if n_test < len(test_dataset):
    torch.manual_seed(0)
    test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

# save the sizes of each one of the sets
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
num_of_classes = 1000
min_pixel_value = 0.0
max_pixel_value = 1.0

# automatically choose device use gpu 0 if it is available o.w. use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print the chosen device
print("device: ", device)

# loading a pre-trained model
# load cohen and salman models
checkpoint = torch.load(
            './Pretrained_Models/Cohen/imagenet/resnet50/noise_' + str(sigma_model) + '/checkpoint.pth.tar', map_location=device)
model = get_architecture(checkpoint["arch"], "imagenet")
model.load_state_dict(checkpoint['state_dict'])

# send model to device
model.to(device)

# put model in evaluation mode
model.eval()

# create indices for the test points
indices = torch.arange(n_test)

# directory to store adversarial examples
directory = "./Adversarial_Examples/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
    sigma_model) + "/n_smooth_" + str(n_smooth)

print("Save results on directories: " + str(save_results))
print("Searching for adversarial examples in: " + str(directory))
if os.path.exists(directory):
    print("Are there saved adversarial examples: Yes")
else:
    print("Are there saved adversarial examples: No")

# If there are no pre created adversarial examples, create new ones
if not os.path.exists(directory):
    # Generate adversarial test examples
    x_test_adv = Smooth_Adv_ImageNet(model, x_test, y_test, indices, n_smooth, sigma_smooth, N_steps, epsilon, device, GPU_CAPACITY=GPU_CAPACITY)

    # Generate adversarial test examples for the base classifier
    x_test_adv_base = Smooth_Adv_ImageNet(model, x_test, y_test, indices, 1, sigma_smooth, N_steps, epsilon, device,
                                GPU_CAPACITY=GPU_CAPACITY)

    # Only store examples for full dataset
    if n_test == 10000:
        os.makedirs(directory)
        with open(directory + "/data.pickle", 'wb') as f:
            pickle.dump([x_test_adv, x_test_adv_base], f)

# If there are pre created adversarial examples, load them
else:
    with open(directory + "/data.pickle", 'rb') as f:
        x_test_adv, x_test_adv_base = pickle.load(f)


# create the noises for the base classifiers only to check its accuracy
noises_base = torch.empty_like(x_test)
for k in range(n_test):
    torch.manual_seed(k)
    noises_base[k:(k + 1)] = torch.randn(
        (1, channels, rows, cols)) * sigma_smooth

# Calculate accuracy of classifier on clean test points
acc, _, _ = calculate_accuracy_smooth(model, x_test, y_test, noises_base, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy :" + str(acc * 100) + "%")

# Calculate accuracy of classifier on adversarial test points
acc, _, _ = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_base, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy on adversarial examples :" + str(acc * 100) + "%")
#exit(1)

del noises_base

# translate desired scores to their functions and put in a list
scores_list = []
for score in calibration_scores:
    if score == 'HCC':
        scores_list.append(scores.class_probability_score)
    if score == 'SC':
        scores_list.append(scores.generalized_inverse_quantile_score)
    if score == 'SC_Reg':
        scores_list.append(scores.rank_regularized_score)

# create dataframe for storing results
results = pd.DataFrame()

# container for storing bounds on "CP+SS"
quantiles = np.zeros((len(scores_list), 2, n_experiments))

# run for n_experiments data splittings
for experiment in tqdm(range(n_experiments)):

    # Split test data into calibration and test
    idx1, idx2 = train_test_split(indices, test_size=0.5)

    # save sizes of calibration and test sets
    n_calib = x_test[idx1].size()[0]
    n_test_new = x_test[idx2].size()[0]

    print("Calibration:")
    # calibrate the model with the desired scores and get the thresholds
    thresholds, bounds = smooth_calibration_ImageNet(model, x_test[idx1], y_test[idx1], n_smooth, sigma_smooth, alpha,
                                                     num_of_classes, scores_list,
                                                     correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # calibrate base model with the desired scores and get the thresholds
    thresholds_base, _ = smooth_calibration_ImageNet(model, x_test[idx1], y_test[idx1], 1, sigma_smooth, alpha,
                                                     num_of_classes,
                                                     scores_list, correction, base=True, device=device,
                                                     GPU_CAPACITY=GPU_CAPACITY)

    thresholds = thresholds + thresholds_base

    print("Prediction:")
    # put bounds in array of bounds
    for p in range(len(scores_list)):
        quantiles[p, 0, experiment] = bounds[p, 0]
        quantiles[p, 1, experiment] = bounds[p, 1]

    # generate prediction sets on the clean test set
    predicted_clean_sets = predict_sets_ImageNet(model, x_test[idx2], idx2, n_smooth, sigma_smooth, num_of_classes, scores_list, thresholds,
                                                correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # generate prediction sets on the clean test set
    predicted_clean_sets_base = predict_sets_ImageNet(model, x_test[idx2], idx2, 1, sigma_smooth, num_of_classes, scores_list,
                                             thresholds, correction, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # generate prediction sets on the adversarial test set
    predicted_adv_sets = predict_sets_ImageNet(model, x_test_adv[idx2], idx2, n_smooth, sigma_smooth, num_of_classes, scores_list, thresholds,
                                      correction, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # generate prediction sets on the adversarial test set
    predicted_adv_sets_base = predict_sets_ImageNet(model, x_test_adv_base[idx2], idx2, 1, sigma_smooth, num_of_classes,
                                           scores_list, thresholds, correction, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)

    # arrange results on clean test set in dataframe
    for p in range(len(scores_list)):
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
        score_name = calibration_scores[p]
        methods_list = [score_name + '_simple', score_name + '_smoothed_classifier', score_name + '_smoothed_score',
                        score_name + '_smoothed_score_correction']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_clean_sets[p][r], x_test[idx2].numpy(), y_test[idx2].numpy(),
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
            res = evaluate_predictions(predicted_adv_sets[p][r], x_test[idx2].numpy(), y_test[idx2].numpy(),
                                       conditional=False)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = epsilon
            res['Black box'] = 'CNN sigma = ' + str(sigma_model)
            # Add results to the list
            results = results.append(res)

    # clean memory
    del idx1, idx2, predicted_clean_sets, predicted_clean_sets_base, predicted_adv_sets, predicted_adv_sets_base, bounds, thresholds, thresholds_base
    gc.collect()

# directory to save results
if save_results:
    directory = "./Results/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
        sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)

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

# plot interval size results
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
