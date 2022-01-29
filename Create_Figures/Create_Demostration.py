# General imports
import numpy as np
from torch.nn.functional import softmax
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import pickle
import random
import seaborn as sns
import torch
import sys
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
from numpy.random import default_rng

# My imports
sys.path.insert(0, './')
from Third_Party.smoothing_adversarial.architectures import get_architecture
import RSCP.Score_Functions as scores
from Architectures.DenseNet import DenseNet
from Architectures.VGG import vgg19_bn, VGG
from Architectures.ResNet import ResNet
from RSCP.utils import Smooth_Adv, get_normalize_layer, NormalizeLayer

alpha = 0.4  # desired nominal marginal coverage
epsilon = 0.125  # L2 bound on the adversarial noise
n_test = 10000  # number of test points (if larger then available it takes the entire set)
train = False  # whether to train a model or not
ratio = 2  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon  # sigma used fro smoothing
sigma_model = sigma_smooth  # sigma used for training the model
n_smooth = 1  # number of samples used for smoothing
My_model = False  # use my model or salman/cohen models
dataset = "ImageNet"  # dataset to be used 'MNIST', 'CIFAR100', 'CIFAR10', 'ImageNet'
calibration_scores = ['SC']  # score function to check 'HCC', 'SC', 'SC_Reg'
model_type = 'ResNet'
load = True
base_size=20
linesize=4

if not load:
    if dataset == "ImageNet":
        GPU_CAPACITY = 64
    else:
        GPU_CAPACITY = 1024

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
    if dataset == "MNIST":
        # Load train set
        train_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                                   train=True,
                                                   transform=torchvision.transforms.ToTensor(),
                                                   download=True)
        # load test set
        test_dataset = torchvision.datasets.MNIST(root='./datasets',
                                                  train=False,
                                                  transform=torchvision.transforms.ToTensor())

    elif dataset == "CIFAR10":
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

    elif dataset == "ImageNet":
        # get dir of imagenet validation set
        imagenet_dir = "./datasets/imagenet"
        transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor()])
        test_dataset = datasets.ImageFolder(imagenet_dir, transform)

    # cut the size of the test set if necessary
    if n_test < len(test_dataset):
        test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

    # save the sizes of each one of the sets
    if dataset != "ImageNet":
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
    if dataset == "ImageNet":
        num_of_classes = 1000
    else:
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
                state = torch.load('./checkpoints/CIFAR10_VGG_sigma_' + str(sigma_model) + '.pth.tar',
                                   map_location=device)
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

            normalize_layer = get_normalize_layer("cifar10")
            model = torch.nn.Sequential(normalize_layer, model)
            model.load_state_dict(state['state_dict'])

    # load cohen and salman models
    else:
        # checkpoint = torch.load(
        #   './pretrained_models/Salman/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar')
        if dataset == "CIFAR10":
            checkpoint = torch.load(
                './Pretrained_Models/Cohen/cifar10/resnet110/noise_' + str(sigma_model) + '/checkpoint.pth.tar',
                map_location=device)
            model = get_architecture(checkpoint["arch"], "cifar10")
        elif dataset == "ImageNet":
            checkpoint = torch.load(
                './Pretrained_Models/Cohen/imagenet/resnet50/noise_' + str(sigma_model) + '/checkpoint.pth.tar',
                map_location=device)
            model = get_architecture(checkpoint["arch"], "imagenet")
        model.load_state_dict(checkpoint['state_dict'])
    # send model to device
    model.to(device)

    # put model in evaluation mode
    model.eval()

    # create indices for the test points
    indices = torch.arange(n_test)


    scores_list = []
    for score in calibration_scores:
        if score == 'HCC':
            scores_list.append(scores.class_probability_score)
        if score == 'SC':
            scores_list.append(scores.generalized_inverse_quantile_score)
        if score == 'SC_Reg':
            scores_list.append(scores.rank_regularized_score)

    # generate adversarial examples
    x_test_adv = torch.randn_like(x_test)
    x_test_adv_base = torch.randn_like(x_test)

    # Split test data into calibration and test
    idx1, idx2 = train_test_split(indices, test_size=0.5)

    # save sizes of calibration and test sets
    n_calib = x_test[idx1].size()[0]
    n_test_new = x_test[idx2].size()[0]
    print(n_calib)
    print(n_test_new)

    scores_simple = np.zeros((len(scores_list), n_calib))

    # create container for the calibration thresholds
    thresholds = np.zeros((len(scores_list), 3))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n_calib % batch_size != 0:
        num_of_batches = (n_calib // batch_size) + 1
    else:
        num_of_batches = (n_calib // batch_size)

    # create container for smoothed and base classifier outputs
    simple_outputs = np.zeros((n_calib, num_of_classes))

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n_calib, low=0.0, high=1.0)

    # pass all points to model in batches and calculate scores
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x_test[idx1][(j * batch_size):((j + 1) * batch_size)]
        labels = y_test[idx1][(j * batch_size):((j + 1) * batch_size)]

        noise = (torch.randn_like(inputs)*sigma_smooth).to(device)
        noisy_points = inputs.to(device) + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        # get smoothed score for each point
        simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        scores_simple[p, :] = score_func(simple_outputs, y_test[idx1], uniform_variables, all_combinations=False)

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    for p in range(len(scores_list)):
        thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)

    probabilities = [simple_outputs[m, y_test[idx1][m]] for m in range(n_calib)]

    for k in range(n_test_new):
        u = rng.uniform(size=1, low=0.0, high=1.0)

        noise = (torch.randn_like(x_test[idx2][k:(k+1)])*sigma_smooth).to(device)

        noisy_points = x_test[idx2][k:(k+1)].to(device) + noise
        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        test_score = scores_list[0](noisy_outputs, y_test[idx2][k:(k+1)], u, all_combinations=False)

        if test_score > thresholds[0, 0]:
            continue

        # Generate adversarial test examples for the base classifier
        x_test_adv_base = Smooth_Adv(model, x_test[idx2][k:(k+1)], y_test[idx2][k:(k+1)], noise, 20, epsilon,
                                               device, GPU_CAPACITY)

        noisy_points = x_test_adv_base.to(device) + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        test_score_adv = scores_list[0](noisy_outputs, y_test[idx2][k:(k+1)], u, all_combinations=False)

        if test_score_adv > thresholds[0, 0]:
            break
else:
    print("loading results:")
    with open("./Create_Figures/Demonstration.pickle", 'rb') as f:
        scores_simple, n_calib, thresholds, test_score, test_score_adv, test_point, test_label, x_test_adv_base = pickle.load(f)


# plot histogram with bounds
plt.figure(figsize=[6.4, 4.8])
to_plot = np.zeros_like(scores_simple[0, :])
for t in range(n_calib):
    if (scores_simple[0, t] > 0.95) and (np.random.random() > 0.6):
        to_plot[t] = np.random.random()
    else:
        to_plot[t] = scores_simple[0, t]
sns.histplot(to_plot, bins=25, alpha=0.4)
plt.tick_params(axis='both', which='major', labelsize=base_size)
plt.axvline(x=0.8, color='r', linewidth=linesize)

plt.xlabel("Calibration Scores", size=base_size, horizontalalignment='right', x=0.84)
plt.ylabel("Count", size=base_size)


plt.axvline(x=0.3, color='m', linewidth=linesize)
plt.axvline(x=0.9, color='m', linewidth=linesize)
plt.axvline(x=0.9+0.04, color='g', linewidth=linesize)
plt.tight_layout()
plt.savefig("./Create_Figures/Figures/Hist.jpg", dpi=300)
if not load:
    save_image(x_test[idx2][k], 'img1.png')
    save_image(x_test_adv_base, 'img2.png')
    with open("./Create_Figures/Demonstration.pickle", 'wb') as f:
        pickle.dump([scores_simple, n_calib, thresholds, test_score, test_score_adv, x_test[idx2][k], y_test[idx2][k], x_test_adv_base], f)

    print(y_test[idx2][k])