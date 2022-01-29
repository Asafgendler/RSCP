import numpy as np
from scipy.stats import rankdata


# The HPS non-conformity score
def class_probability_score(probabilities, labels, u=None, all_combinations=False):

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # calculate scores of each point with all labels
    if all_combinations:
        scores = 1 - probabilities[:, labels]

    # calculate scores of each point with only one label
    else:
        scores = 1 - probabilities[np.arange(num_of_points), labels]

    # return scores
    return scores


# The APS non-conformity score
def generalized_inverse_quantile_score(probabilities, labels, u=None, all_combinations=False):

    # whether to do a randomized score or not
    if u is None:
        randomized = False
    else:
        randomized = True

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # sort probabilities from high to low
    sorted_probabilities = -np.sort(-probabilities)

    # create matrix of cumulative sum of each row
    cumulative_sum = np.cumsum(sorted_probabilities, axis=1)

    # find ranks of each desired label in each row

    # calculate scores of each point with all labels
    if all_combinations:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[:, labels] - 1

    # calculate scores of each point with only one label
    else:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[np.arange(num_of_points), labels] - 1

    # compute the scores of each label in each row
    scores = cumulative_sum[np.arange(num_of_points), label_ranks.T].T

    # compute the probability of the last label that enters
    last_label_prob = sorted_probabilities[np.arange(num_of_points), label_ranks.T].T

    # remove the last label probability or a multiplier of it in the randomized score
    if not randomized:
        scores = scores - last_label_prob
    else:
        scores = scores - np.diag(u) @ last_label_prob

    # return the scores
    return scores


# The RAPS non-conformity score
def rank_regularized_score(probabilities, labels, u=None, all_combinations=False):

    # get the regular scores
    scores = generalized_inverse_quantile_score(probabilities, labels, u, all_combinations)

    # get number of classes
    num_of_classes = np.shape(probabilities)[1]

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # find ranks of each desired label in each row

    # calculate scores of each point with all labels
    if all_combinations:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[:, labels] - 1

    # calculate scores of each point with only one label
    else:
        label_ranks = rankdata(-probabilities, method='ordinal', axis=1)[np.arange(num_of_points), labels] - 1

    # regularize with the ranks
    alpha = 0.99
    k_arg = 25
    lamda = 0.2
    #grid = np.linspace(0, num_of_classes, num_of_classes)
    #scores = alpha * scores + (1-alpha) * ((grid[label_ranks])/num_of_classes)
    tmp = label_ranks+1-k_arg
    tmp[tmp < 0] = 0
    scores = scores + lamda * tmp/(num_of_classes-k_arg)
    scores[scores > 1] = 1

    # return scores
    return scores
