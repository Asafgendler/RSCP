import pandas as pd
import pickle
import argparse
import numpy as np
from plotnine import ggplot,theme_bw,scale_alpha_manual,guides,scale_size_manual,guide_legend,element_rect,element_line, ggsave,scale_color_brewer,annotate,element_blank, element_text, scale_x_discrete,scale_y_continuous, aes,theme, facet_grid, labs, geom_point, facet_wrap, geom_boxplot, geom_hline
import sys
sys.path.insert(0, './')

# parameters
parser = argparse.ArgumentParser(description='Results on DenseNet and VGG')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used: CIFAR100, CIFAR10')
args = parser.parse_args()

# check parameters
assert args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100', 'Dataset can only be CIFAR10 or CIFAR100.'

lower_quantiles_mean = np.zeros((3, 2))
lower_quantiles_std = np.zeros((3, 2))
base_size = 18
for k, model_type in enumerate(['ResNet', 'DenseNet', 'VGG']):
    alpha = 0.1
    epsilon = 0.125
    ratio = 2  # ratio between adversarial noise bound to smoothed noise
    sigma_smooth = ratio * epsilon * 0  # sigma used fro smoothing
    sigma_model = sigma_smooth  # sigma used for training the model
    n_smooth = 1  # number of samples used for smoothing
    My_model = True
    normalized = True
    dataset = args.dataset
    Regularization = False

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

    if Regularization:
        directory = directory + "/Regularization"

    if alpha != 0.1:
        directory = directory + "/alpha_" + str(alpha)

    path = directory + "/results.csv"

    results = pd.read_csv(path)
    results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
    results = results.drop(columns=['Black box', 'Conditional coverage', 'Size cover'])

    results1 = results[(results["Method"] == "SC_simple") | (results["Method"] == "HCC_simple")]
    results1["Method"].replace({"SC_simple": "APS", "HCC_simple": "HPS"}, inplace=True)
    results1 = results1.rename(columns={'Method': 'Base Score'}, inplace=False)

    data1 = results1[results1['noise_L2_norm'] == epsilon].copy()
    data1["Type"] = " Vanilla CP"

    data1["Model"] = model_type
    data1["Position"] = "  "

    sigma_smooth = ratio * epsilon  # sigma used fro smoothing
    sigma_model = sigma_smooth  # sigma used for training the model
    n_smooth = 256

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

    if Regularization:
        directory = directory + "/Regularization"

    if alpha != 0.1:
        directory = directory + "/alpha_" + str(alpha)

    path = directory + "/results.csv"

    results = pd.read_csv(path)

    results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
    results = results.drop(columns=['Black box', 'Conditional coverage', 'Size cover'])

    results2 = results[(results["Method"] == "SC_smoothed_score") | (results["Method"] == "HCC_smoothed_score")]
    results2["Method"].replace({"SC_smoothed_score": "APS", "HCC_smoothed_score": "HPS"}, inplace=True)
    results2 = results2.rename(columns={'Method': 'Base Score'}, inplace=False)

    data2 = results2[results2["noise_L2_norm"] == epsilon].copy()
    data2["Type"] = "CP + SS"

    data2["Model"] = model_type

    data2["Position"] = " "

    results3 = results[
        (results["Method"] == "SC_smoothed_score_correction") | (results["Method"] == "HCC_smoothed_score_correction")]
    results3["Method"].replace({"SC_smoothed_score_correction": "APS", "HCC_smoothed_score_correction": "HPS"},
                               inplace=True)
    results3 = results3.rename(columns={'Method': 'Base Score'}, inplace=False)

    data3 = results3[results3["noise_L2_norm"] == epsilon].copy()
    data3["Type"] = "RSCP"

    data3["Model"] = model_type

    data3["Position"] = " "

    current = data1.append(data2)
    current = current.append(data3)

    #current['Position'] = current['Position'].cat.reorder_categories(['Up', 'Down'])

    if k == 0:
        final = current
    else:
        final = final.append(current)

    with open(directory + "/quantiles_bounds.pickle", 'rb') as f:
        quantiles = np.array(pickle.load(f))[0]

    for p in range(2):
        lower_quantiles_mean[k, p] = np.mean(quantiles[p, 0, :])
        lower_quantiles_std[k, p] = np.std(quantiles[p, 0, :])


nominal = pd.DataFrame({'name': ['Nominal Level'], 'Coverage': [1-alpha], 'Position': [' ']})
lines1 = pd.DataFrame({'name': ['APS', 'APS', 'APS'], 'Coverage': [lower_quantiles_mean[0, 0], lower_quantiles_mean[1, 0], lower_quantiles_mean[2, 0]], 'Position': [' ', ' ', ' '], 'Model': ['DenseNet', 'ResNet', 'VGG']})
lines2 = pd.DataFrame({'name': ['HPS', 'HPS', 'HPS'], 'Coverage': [lower_quantiles_mean[0, 1], lower_quantiles_mean[1, 1], lower_quantiles_mean[2, 1]], 'Position': [' ', ' ', ' '], 'Model': ['DenseNet', 'ResNet', 'VGG']})

p = ggplot(final,
           aes(x="Type", y="Coverage", color="Base Score")) \
    + geom_boxplot() \
    + facet_grid('Position ~ Model', scales='free', space='free') \
    + geom_hline(nominal, aes(yintercept='Coverage', size='name'), linetype="dashed",  color="black") \
    + geom_hline(lines1, aes(yintercept='Coverage', alpha='name'), linetype="dashed", size=1, color='#00BFC4') \
    + geom_hline(lines2, aes(yintercept='Coverage', alpha='name'), linetype="dashed", size=1, color='#F8766D') \
    + labs(x="", y="Marginal Coverage", title="") \
    + theme_bw(base_size=base_size) \
    + theme(panel_grid_minor=element_blank(),
            panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
            plot_title=element_text(face="bold"),
            legend_background=element_rect(fill="white", size=4, colour="white"),
            text=element_text(size=base_size, face="plain"),
            legend_title_align='center',
            legend_position=(-0.3, 0.5),
            strip_background_y=element_blank(),
            axis_text_x=element_text(rotation=45, vjust=1, hjust=1),
            subplots_adjust={'hspace':0.05},
            legend_entry_spacing=10,
            legend_direction='horizontal') \
    + scale_alpha_manual(name="CP+SS Worst Coverage", values=(1, 1),
                         guide=guide_legend(override_aes={"color": ('#F8766D', '#00BFC4')})) \
    + scale_size_manual(name=" ", values=(1, 1)) \
    + guides(color=guide_legend(order=1)) \
    + scale_y_continuous(expand=(0.1, 0, 0.1, 0))


#+ facet_wrap(['Dataset']) \
# + scale_y_continuous(expand=(0, 0)) \
p.save('./Create_Figures/Figures/DenseNet_'+str(dataset)+'_coverage.pdf')

#final = final[final['Type'] != 'CP + BS  \n(baseline)']
p = ggplot(final,
           aes(x="Type", y="Size", color="Base Score")) \
    + facet_wrap('~ Model', nrow = 1) \
    + labs(x="", y="Average Set Size", title="") \
    + theme_bw(base_size=base_size) \
    + theme(legend_title_align='center',
            panel_grid_minor=element_blank(),
            panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
            plot_title=element_text(face="bold"),
            legend_background=element_rect(fill="white", size=4, colour="white"),
            text=element_text(size=base_size, face="plain"),
            legend_position="none",
            axis_text_x=element_text(rotation=45, vjust=1, hjust=1),
            legend_direction='horizontal',
            legend_entry_spacing=10) \
    + scale_y_continuous(expand=(0.1, 0, 0.1 ,0)) \
    + geom_boxplot()
#+ facet_wrap(['Dataset']) \
# + scale_y_continuous(expand=(0, 0)) \
p.save('./Create_Figures/Figures/DenseNet_'+str(dataset)+'_size.pdf')


