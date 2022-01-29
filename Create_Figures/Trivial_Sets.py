import pandas as pd
from plotnine import ggplot,theme_bw,element_rect,element_line,guide_legend,guides,scale_size_manual, ggsave,scale_color_brewer,annotate,element_blank, element_text, scale_x_discrete,scale_y_continuous, aes,theme, facet_grid, labs, geom_point, facet_wrap, geom_boxplot, geom_hline
import sys
sys.path.insert(0, './')

dataset = "CIFAR10"
alpha = 0.1
epsilon = 0.125
ratio = 4  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon  # sigma used fro smoothing
sigma_model = sigma_smooth * 0.0  # sigma used for training the model
n_smooth = 256  # number of samples used for smoothing
My_model = False
normalized = False
Regularization = False
base_size = 18

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
data1["Type"] = " Vanilla CP"
data1["Dataset"] = dataset
data1["Position"] = "  "

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

final = current

nominal = pd.DataFrame({'name': ['Nominal Level'], 'Coverage': [1-alpha], 'Position': [' ']})

p = ggplot(final,
           aes(x="Type", y="Coverage", color="Base Score")) \
    + geom_boxplot() \
    + geom_hline(nominal, aes(yintercept='Coverage', size='name'), linetype="dashed",  color="black") \
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
    + scale_size_manual(name=" ", values=(1, 1)) \
    + guides(color=guide_legend(order=1)) \
    + scale_y_continuous(expand=(0.1, 0, 0.1, 0))


p.save('./Create_Figures/Figures/Trivial_Sets_coverage.pdf')

#final = final[final['Type'] != 'CP + BS  \n(baseline)']
p = ggplot(final,
           aes(x="Type", y="Size", color="Base Score")) \
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
            legend_entry_spacing=10,
            subplots_adjust={'wspace':0.4}) \
    + scale_y_continuous(expand=(0.1, 0, 0.1 ,0)) \
    + geom_boxplot()

p.save('./Create_Figures/Figures/Trivial_Sets_size.pdf')


