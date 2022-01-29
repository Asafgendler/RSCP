import pandas as pd
import pickle
import argparse
import numpy as np
from plotnine import ggplot,guides,guide_legend,scale_size_manual, theme_bw,element_rect,scale_alpha_manual,element_line,scale_color_manual, ggsave,scale_color_brewer,annotate,element_blank, element_text, scale_x_discrete,scale_y_continuous, aes,theme, facet_grid, labs, geom_point, facet_wrap, geom_boxplot, geom_hline
import sys
sys.path.insert(0, './')

# parameters
parser = argparse.ArgumentParser(description='Main Results')
parser.add_argument('--My_CIFAR10', action='store_true', help='Results on my CIFAR10 model or Cohens')
args = parser.parse_args()


base_size=18
lower_quantiles_mean = np.zeros((3, 2))
lower_quantiles_std = np.zeros((3, 2))
for k, dataset in enumerate(["CIFAR10", "CIFAR100", "ImageNet"]):
    alpha = 0.1
    if dataset == "ImageNet":
        epsilon = 0.25
    else:
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

    path = directory + "/results.csv"

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

    sigma_smooth = ratio * epsilon  # sigma used fro smoothing
    sigma_model = sigma_smooth  # sigma used for training the model
    if dataset == "ImageNet":
        n_smooth = 64
    else:
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

    path = directory + "/results.csv"

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

    results3 = results[
        (results["Method"] == "SC_smoothed_score_correction") | (results["Method"] == "HCC_smoothed_score_correction")]
    results3["Method"].replace({"SC_smoothed_score_correction": "APS", "HCC_smoothed_score_correction": "HPS"},
                               inplace=True)
    results3 = results3.rename(columns={'Method': 'Base Score'}, inplace=False)

    data3 = results3[results3["noise_L2_norm"] == epsilon].copy()
    data3["Type"] = "RSCP"
    data3["Dataset"] = dataset
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
lines1 = pd.DataFrame({'name': ['APS', 'APS', 'APS'], 'Coverage': [lower_quantiles_mean[0, 0], lower_quantiles_mean[1, 0], lower_quantiles_mean[2, 0]], 'Position': [' ', ' ', ' '], 'Dataset': ['CIFAR10', 'CIFAR100', 'ImageNet']})
lines2 = pd.DataFrame({'name': ['HPS', 'HPS', 'HPS'], 'Coverage': [lower_quantiles_mean[0, 1], lower_quantiles_mean[1, 1], lower_quantiles_mean[2, 1]], 'Position': [' ', ' ', ' '], 'Dataset': ['CIFAR10', 'CIFAR100', 'ImageNet']})

# lines = pd.DataFrame({'name': ['nominal level', 'nominal level', 'nominal level',
#                                     'CP+SS worst coverage level APS', 'CP+SS worst coverage level APS', 'CP+SS worst coverage level APS',
#                                     'CP+SS worst coverage level HPS', 'CP+SS worst coverage level HPS', 'CP+SS worst coverage level HPS'],
#                         'Coverage' : [1-alpha, 1-alpha, 1-alpha, lower_quantiles_mean[0, 0], lower_quantiles_mean[1, 0], lower_quantiles_mean[2, 0], lower_quantiles_mean[0, 1], lower_quantiles_mean[1, 1], lower_quantiles_mean[2, 1]],
#                         'Position': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
#                         'Dataset': ['CIFAR10', 'CIFAR100', 'ImageNet', 'CIFAR10', 'CIFAR100', 'ImageNet', 'CIFAR10', 'CIFAR100', 'ImageNet'],
#                       })

# a=final[(final["Type"]=="CP + BS  \n(baseline)") & (final["Dataset"]=="CIFAR10") & (final["Base Score"]=="HPS")]['Coverage'].mean()
# b=final[(final["Type"]=="CP + BS  \n(baseline)") & (final["Dataset"]=="CIFAR100") & (final["Base Score"]=="HPS")]['Coverage'].mean()
# c=final[(final["Type"]=="CP + BS  \n(baseline)") & (final["Dataset"]=="ImageNet") & (final["Base Score"]=="HPS")]['Coverage'].mean()
# d=final[(final["Type"]=="CP + BS  \n(baseline)") & (final["Dataset"]=="CIFAR10") & (final["Base Score"]=="HPS")]['Coverage'].mean()
# e=final[(final["Type"]=="CP + BS  \n(baseline)") & (final["Dataset"]=="CIFAR100") & (final["Base Score"]=="HPS")]['Coverage'].mean()
# f=final[(final["Type"]=="CP + BS  \n(baseline)") & (final["Dataset"]=="ImageNet") & (final["Base Score"]=="HPS")]['Coverage'].mean()
#
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)

p = ggplot(final,
           aes(x="Type", y="Coverage", color="Base Score")) \
    + geom_boxplot() \
    + facet_grid('Position ~ Dataset', scales='free', space='free') \
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
    + scale_alpha_manual(name="CP+SS Worst Coverage", values=(1,1),
                        guide=guide_legend(override_aes={"color":('#F8766D', '#00BFC4')})) \
    + scale_size_manual(name=" ",values=(1,1))\
    + guides(color=guide_legend(order=1)) \
    + scale_y_continuous(expand=(0.1, 0, 0.1, 0))

#geom_hline(aes(yintercept='Coverage'), pd.DataFrame(nominal), color='black', linetype="dashed", size=1) \
#geom_hline(aes(yintercept='Coverage'), pd.DataFrame(lines1), color='#00BFC4', linetype="dashed", size=1) \
#geom_hline(aes(yintercept='Coverage'), pd.DataFrame(lines2), color='#F8766D', linetype="dashed", size=1) \
    #+ facet_wrap(['Dataset']) \
# + scale_y_continuous(expand=(0, 0)) \
p.save('./Create_Figures/Figures/Coverage_Comparison.pdf')

#final = final[final['Type'] != 'CP + BS  \n(baseline)']
p = ggplot(final,
           aes(x="Type", y="Size", color="Base Score")) \
    + facet_wrap('~ Dataset', scales='free_y', nrow = 1) \
    + labs(x="", y="Average Set Size", title="") \
    + theme_bw(base_size=base_size) \
    + theme(legend_title_align='center',
            panel_grid_minor=element_blank(),
            panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
            plot_title=element_text(face="bold"),
            legend_background=element_rect(fill="white", size=4, colour="white"),
            text=element_text(size=base_size
                              , face="plain"),
            legend_position="none",
            axis_text_x=element_text(rotation=45, vjust=1, hjust=1),
            legend_direction='horizontal',
            legend_entry_spacing=10,
            subplots_adjust={'wspace':0.4}) \
    + scale_y_continuous(expand=(0.1, 0, 0.1 ,0)) \
    + geom_boxplot()
#+ facet_wrap(['Dataset']) \
# + scale_y_continuous(expand=(0, 0)) \
p.save('./Create_Figures/Figures/Size_Comparison.pdf')


