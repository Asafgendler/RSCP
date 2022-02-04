import pandas as pd
import numpy as np
import argparse
from plotnine import ggplot, scale_x_continuous, theme_bw, element_rect, element_line, geom_line, scale_color_brewer, \
    annotate, \
    element_blank, element_text, scale_x_discrete, scale_y_continuous,position_dodge,geom_errorbar, aes, theme, facet_grid, labs, geom_point, \
    facet_wrap, geom_boxplot, geom_hline
import sys
sys.path.insert(0, './')

# parameters
parser = argparse.ArgumentParser(description='Effect_of_n_smooth_figure')
parser.add_argument('--comparison', action='store_true', help='Results on all datasets or on one')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used if results are only on one')
args = parser.parse_args()

# check parameters
assert args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100' or args.dataset == 'ImageNet', 'Dataset can only be CIFAR10, CIFAR100 or ImageNet.'


alpha = 0.1
ratio = 2  # ratio between adversarial noise bound to smoothed noise
dataset = args.dataset  # dataset to be used 'MNIST', 'CIFAR100', 'CIFAR10', 'ImageNet'
Regularization = False
comparison = args.comparison
base_size = 18
line_size = 1.5
error_bar = 0.25

if comparison:
    datasets = ["CIFAR10", "CIFAR100", "ImageNet"]
else:
    datasets = [dataset]

for k, dataset in enumerate(datasets):

    if dataset == "CIFAR100":
        My_model = True
        normalized = True
    else:
        My_model = False
        normalized = False

    if dataset == "ImageNet":
        epsilon = 0.25
        n_smooths = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
        breaks = (1, 2, 4, 8, 16, 64, 128, 256)
    else:
        epsilon = 0.125
        n_smooths = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        breaks = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

    sigma_smooth = ratio * epsilon  # sigma used fro smoothing
    sigma_model = sigma_smooth  # sigma used for training the model

    Coverages_mean = np.zeros((2, np.size(n_smooths)))
    Coverages_std = np.zeros((2, np.size(n_smooths)))
    Sizes_mean = np.zeros((2, np.size(n_smooths)))
    Sizes_std = np.zeros((2, np.size(n_smooths)))

    for j, n_smooth in enumerate(n_smooths):
        directory = "./Results/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
            sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)
        if normalized:
            directory = directory + "/Robust"

        if dataset == "CIFAR10":
            if My_model:
                directory = directory + "/My_Model"
            else:
                directory = directory + "/Their_Model"

        if Regularization:
            directory = directory + "/Regularization"

        path = directory + "/results.csv"

        results = pd.read_csv(path)
        results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
        results = results.drop(columns=['Black box', 'Conditional coverage', 'Size cover'])

        results1 = results[(results["Method"] == "SC_smoothed_score_correction")]
        data1 = results1[results1["noise_L2_norm"] == epsilon].copy()
        Coverages_mean[0, j] = data1['Coverage'].mean()
        Coverages_std[0, j] = data1['Coverage'].sem()
        Sizes_mean[0, j] = data1['Size'].mean()
        Sizes_std[0, j] = data1['Size'].sem()

        results2 = results[(results["Method"] == "HCC_smoothed_score_correction")]
        data2 = results2[results2["noise_L2_norm"] == epsilon].copy()
        Coverages_mean[1, j] = data2['Coverage'].mean()
        Coverages_std[1, j] = data2['Coverage'].sem()
        Sizes_mean[1, j] = data2['Size'].mean()
        Sizes_std[1, j] = data2['Size'].sem()

    df1 = pd.DataFrame(
        {'Dataset': dataset, 'n_s': n_smooths, 'Coverage': Coverages_mean[0, :], 'Coverage_STD': Coverages_std[0, :],
         'Size_STD': Sizes_std[0, :], 'Size': Sizes_mean[0, :], 'Base Score': 'APS'})
    df2 = pd.DataFrame(
        {'Dataset': dataset, 'n_s': n_smooths, 'Coverage': Coverages_mean[1, :], 'Coverage_STD': Coverages_std[1, :],
         'Size_STD': Sizes_std[1, :], 'Size': Sizes_mean[1, :], 'Base Score': 'HPS'})
    df1 = df1.append(df2)
    if k == 0:
        final = df1
    else:
        final = final.append(df1)


def breaks_fun(x):
    if np.max(x) > 260:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]


if comparison:
    p = ggplot(final,
               aes(x="n_s", y="Coverage", color='Base Score')) \
        + geom_line(size=line_size) \
        + facet_wrap('~ Dataset', scales="free", nrow=1) \
        + labs(x=r'$n_s$', y="Marginal Coverage", title="") \
        + theme_bw(base_size=base_size) \
        + theme(panel_grid_minor=element_blank(),
                panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
                plot_title=element_text(face="bold"),
                legend_background=element_rect(fill="white", size=4, colour="white"),
                text=element_text(size=base_size, face="plain"),
                legend_title_align='center',
                legend_direction='horizontal',
                legend_entry_spacing=10,
                axis_text_x=element_text(angle = 90, vjust=1, hjust = 0.5),
                subplots_adjust={'wspace': 0.3},
                legend_position="none") \
        + scale_x_continuous(breaks=breaks_fun, trans='log2') \
        + geom_errorbar(aes(ymin="Coverage-Coverage_STD", ymax="Coverage+Coverage_STD"), width=error_bar) \
        + geom_point(size=2*line_size)

    p.save('./Create_Figures/Figures/Effect_of_n_smooth_coverage_comparison.pdf', width=15, height=4.8)

    p = ggplot(final,
               aes(x="n_s", y="Size", color='Base Score')) \
        + geom_line(size=line_size) \
        + facet_wrap('~ Dataset', scales='free', nrow=1) \
        + labs(x=r'$n_s$', y="Average Set Size", title="") \
        + theme_bw(base_size=base_size) \
        + theme(panel_grid_minor=element_blank(),
                panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
                plot_title=element_text(face="bold"),
                legend_background=element_rect(fill="white", size=4, colour="white"),
                text=element_text(size=base_size, face="plain"),
                legend_title_align='center',
                legend_direction='horizontal',
                legend_entry_spacing=10,
                axis_text_x=element_text(angle = 90, vjust=1, hjust = 0.5),
                subplots_adjust={'wspace': 0.3},
                legend_position=(0.5, -0.16)) \
        + scale_x_continuous(breaks=breaks_fun, trans='log2') \
        + geom_errorbar(aes(ymin="Size-Size_STD", ymax="Size+Size_STD"), width=error_bar) \
        + geom_point(size=2*line_size)

    p.save('./Create_Figures/Figures/Effect_of_n_smooth_size_comparison.pdf', width=15, height=4.8)
else:
    p = ggplot(final,
               aes(x="n_s", y="Coverage", color='Base Score')) \
        + geom_line(size=line_size) \
        + labs(x=r'$n_s$', y="Marginal Coverage", title="") \
        + theme_bw(base_size=base_size) \
        + theme(panel_grid_minor=element_blank(),
                panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
                plot_title=element_text(face="bold"),
                legend_background=element_rect(fill="None", size=4, colour="None"),
                text=element_text(size=base_size, face="plain"),
                legend_title_align='center',
                legend_direction='horizontal',
                legend_entry_spacing=10,
                axis_title_x=element_text(margin={'t': -2}),
                axis_text_x=element_text(angle=90, vjust=1, hjust=0.5),
                legend_position=(0.55, 0.3)) \
        + scale_x_continuous(breaks=breaks, trans='log2') \
        + geom_errorbar(aes(ymin="Coverage-Coverage_STD", ymax="Coverage+Coverage_STD", width=error_bar)) \
        + geom_point(size=2*line_size)

    p.save('./Create_Figures/Figures/Effect_of_n_smooth_coverage_' + str(dataset) + '.pdf')

    p = ggplot(final,
               aes(x="n_s", y="Size", color='Base Score')) \
        + geom_line(size=line_size) \
        + labs(x=r'$n_s$', y="Average Set Size", title="") \
        + theme_bw(base_size=base_size) \
        + theme(panel_grid_minor=element_blank(),
                panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
                plot_title=element_text(face="bold"),
                legend_background=element_rect(fill="None", size=4, colour="None"),
                text=element_text(size=base_size, face="plain"),
                legend_title_align='center',
                legend_direction='horizontal',
                legend_entry_spacing=10,
                axis_title_x=element_text(margin={'t': -2}),
                axis_text_x=element_text(angle=90, vjust=1, hjust=0.5),
                legend_position="none") \
        + scale_x_continuous(breaks=breaks, trans='log2') \
        + geom_errorbar(aes(ymin="Size-Size_STD", ymax="Size+Size_STD"), width=error_bar)\
        + geom_point(size=2*line_size)

    p.save('./Create_Figures/Figures/Effect_of_n_smooth_size_' + str(dataset) + '.pdf')

