import pandas as pd
import pickle
import argparse
import numpy as np
from plotnine import ggplot,guides,guide_legend,scale_size_manual, theme_bw,element_rect,scale_alpha_manual,element_line,scale_color_manual, ggsave,scale_color_brewer,annotate,element_blank, element_text, scale_x_discrete,scale_y_continuous, aes,theme, facet_grid, labs, geom_point, facet_wrap, geom_boxplot, geom_hline
import sys
sys.path.insert(0, './')

# parameters
parser = argparse.ArgumentParser(description='Main Results conditioned on y')
parser.add_argument('--My_CIFAR10', action='store_true', help='Results on my CIFAR10 model or Cohens')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used: CIFAR100, CIFAR10')
args = parser.parse_args()

# check parameters
assert args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100', 'Dataset can only be CIFAR10 or CIFAR100.'

base_size=18
lower_quantiles_mean = np.zeros((3, 2))
lower_quantiles_std = np.zeros((3, 2))
dataset = args.dataset
if dataset == "CIFAR10":
    num_of_classes = 10
else:
    num_of_classes = 100

for k in range(num_of_classes):
    alpha = 0.1
    epsilon = 0.125
    ratio = 2  # ratio between adversarial noise bound to smoothed noise
    sigma_smooth = ratio * epsilon * 0  # sigma used fro smoothing
    sigma_model = sigma_smooth  # sigma used for training the model
    n_smooth = 1  # number of samples used for smoothing
    if (dataset == "CIFAR100") or ((dataset == "CIFAR10") and (args.My_CIFAR10)):
        My_model = True
        normalized = True
    else:
        My_model = False
        normalized = False
    Regularization = False

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

    if alpha != 0.1:
        directory = directory + "/alpha_" + str(alpha)

    path = directory + "/results_given_y.csv"

    results = pd.read_csv(path)
    results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
    results = results.drop(columns=['Black box', 'Conditional coverage', 'Size cover'])

    results1 = results[(results["Method"] == "SC_simple") | (results["Method"] == "HCC_simple")]
    results1["Method"].replace({"SC_simple": "APS", "HCC_simple": "HPS"}, inplace=True)
    results1 = results1.rename(columns={'Method': 'Base Score'}, inplace=False)

    data1 = results1[results1['noise_L2_norm'] == epsilon].copy()
    #data1["Type"] = "CP + BS  \n(baseline)"
    data1["Type"] = " Vanilla CP"
    data1["Dataset"] = dataset
    data1["Position"] = "  "
    data1["To_plot1"] = data1["Coverage given "+str(k)]
    #data1["To_plot2"] = data1["Size given " + str(k)]
    data1["Class"] = str(k)

    sigma_smooth = ratio * epsilon  # sigma used fro smoothing
    sigma_model = sigma_smooth  # sigma used for training the model
    n_smooth = 256

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

    if alpha != 0.1:
        directory = directory + "/alpha_" + str(alpha)

    path = directory + "/results_given_y.csv"

    results = pd.read_csv(path)

    results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
    results = results.drop(columns=['Black box', 'Conditional coverage', 'Size cover'])

    results2 = results[(results["Method"] == "SC_smoothed_score") | (results["Method"] == "HCC_smoothed_score")]
    results2["Method"].replace({"SC_smoothed_score": "APS", "HCC_smoothed_score": "HPS"}, inplace=True)
    results2 = results2.rename(columns={'Method': 'Base Score'}, inplace=False)

    data2 = results2[results2["noise_L2_norm"] == epsilon].copy()
    data2["Type"] = "CP + SS"
    data2["Dataset"] = dataset
    data2["Position"] = " "
    data2["To_plot1"] = data2["Coverage given "+str(k)]
    data2["To_plot2"] = data2["Size given " + str(k)]
    data2["Class"] = str(k)

    results3 = results[
        (results["Method"] == "SC_smoothed_score_correction") | (results["Method"] == "HCC_smoothed_score_correction")]
    results3["Method"].replace({"SC_smoothed_score_correction": "APS", "HCC_smoothed_score_correction": "HPS"},
                               inplace=True)
    results3 = results3.rename(columns={'Method': 'Base Score'}, inplace=False)

    data3 = results3[results3["noise_L2_norm"] == epsilon].copy()
    data3["Type"] = "RSCP"
    data3["Dataset"] = dataset
    data3["Position"] = " "
    data3["To_plot1"] = data3["Coverage given "+str(k)]
    data3["To_plot2"] = data3["Size given " + str(k)]
    data3["Class"] = str(k)

    current = data2.append(data3)
    #current = current.append(data3)

    if k == 0:
        final = current
    else:
        final = final.append(current)


nominal = pd.DataFrame({'name': ['Nominal Level'], 'Coverage': [1-alpha], 'Position': [' ']})

p = ggplot(final,
           aes(x="Class", y="To_plot1", color="Base Score")) \
    + geom_boxplot() \
    + facet_grid('Type ~ .', scales='free', space='free') \
    + geom_hline(nominal, aes(yintercept='Coverage', size='name'), linetype="dashed",  color="black") \
    + labs(x="Class", y="Marginal Coverage", title="") \
    + theme_bw(base_size=base_size) \
    + theme(panel_grid_minor=element_blank(),
            panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
            plot_title=element_text(face="bold"),
            legend_background=element_rect(fill="white", size=4, colour="white"),
            text=element_text(size=base_size, face="plain"),
            legend_title_align='center',
            legend_position=(-0.3, 0.5),
            strip_background_y=element_blank(),
            panel_spacing_y=0.2,
            axis_title_y = element_text(margin={'r': 20}),
            legend_entry_spacing=10,
            legend_direction='horizontal') \
    + scale_size_manual(name=" ",values=(1,1))\
    + guides(color=guide_legend(order=1)) \
    + scale_y_continuous(expand=(0.1, 0, 0.1, 0))

p.save('./Create_Figures/Figures/Coverage_Comparison_given_y.pdf')

p = ggplot(final,
           aes(x="Class", y="To_plot2", color="Base Score")) \
    + geom_boxplot() \
    + facet_grid('Type ~ .', scales='free', space='free') \
    + labs(x="Class", y="Average Set Size", title="") \
    + theme_bw(base_size=base_size) \
    + theme(panel_grid_minor=element_blank(),
            panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
            plot_title=element_text(face="bold"),
            legend_background=element_rect(fill="white", size=4, colour="white"),
            text=element_text(size=base_size, face="plain"),
            legend_title_align='center',
            legend_position=(-0.3, 0.5),
            strip_background_y=element_blank(),
            panel_spacing_y=0.2,
            axis_title_y = element_text(margin={'r': 20}),
            legend_entry_spacing=10,
            legend_direction='horizontal') \
    + scale_size_manual(name=" ",values=(1,1))\
    + guides(color=guide_legend(order=1)) \
    + scale_y_continuous(expand=(0.1, 0, 0.1, 0))

p.save('./Create_Figures/Figures/Size_Comparison_given_y.pdf')