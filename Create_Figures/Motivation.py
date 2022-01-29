import pandas as pd
from plotnine import ggplot, theme_bw,scale_color_manual, element_text,element_blank,element_rect,element_line,scale_color_brewer, scale_x_discrete,scale_y_continuous, aes,theme, facet_grid, labs, geom_point, facet_wrap, geom_boxplot, geom_hline
import sys
sys.path.insert(0, './')

alpha = 0.1
epsilon = 0.125  # L2 bound on the adversarial noise
ratio = 0.0  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon  # sigma used fro smoothing
sigma_model = sigma_smooth  # sigma used for training the model
n_smooth = 1  # number of samples used for smoothing
My_model = True  # use my model or salman/cohen models
dataset = "CIFAR10"  # dataset to be used 'MNIST', 'CIFAR100', 'CIFAR10', 'ImageNet'
base_size = 18
model_type = 'VGG'
Regularization = False

if My_model:
    normalized = True
else:
    normalized = False

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

results1 = pd.read_csv(path)
results1 = results1.loc[:, ~results1.columns.str.contains('^Unnamed')]
results1 = results1.drop(columns=['Black box', 'Conditional coverage', 'Size cover'])

results1 = results1[results1["Method"] == "HCC_simple"]

data1 = results1[results1['noise_L2_norm'] == 0].copy()
data1["Type"] = " Vanilla CP\n Clean Test"
data1["Position"] = " "

data2 = results1[results1['noise_L2_norm'] == epsilon].copy()
data2["Type"] = " Vanilla CP     \nAdversarial Test"
data2["Position"] = "  "

ratio = 2  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon  # sigma used fro smoothing
sigma_model = sigma_smooth  # sigma used for training the model
n_smooth = 256  # number of samples used for smoothing

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

results2 = pd.read_csv(path)

results2 = results2.loc[:, ~results2.columns.str.contains('^Unnamed')]
results2 = results2.drop(columns=['Black box', 'Conditional coverage', 'Size cover'])

results2 = results2[results2["Method"] == "HCC_smoothed_score_correction"]

data3 = results2[results2["noise_L2_norm"] == epsilon].copy()
data3["Type"] = "Our Method (RSCP)\nAdversarial Test   "
data3["Position"] = " "


final = data1.append(data2)
final = final.append(data3)

nominal = pd.DataFrame({'name': ['Nominal Level'], 'Coverage': [1-alpha], 'Position': [' ']})
p = ggplot(final,
           aes(x="Type", y="Coverage")) \
    + geom_hline(nominal, aes(yintercept='Coverage', color="name"), linetype="dashed", size=1) \
    + labs(x="Method", y="Marginal Coverage", title="") \
    + theme_bw(base_size=base_size)\
    + theme(panel_grid_minor=element_blank(),
            panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
            plot_title=element_text(face="bold"),
            legend_background=element_rect(fill="none", size=4, colour="none"),
            text=element_text(size=base_size, face="plain"),
            legend_title_align='center',
            legend_title=element_blank(),
            legend_position=(0.33, 0.43),
            axis_text_x=element_text(rotation=45, vjust=1, hjust=1),
            axis_title_x=element_blank()) \
    + scale_y_continuous(expand=(0.1, 0, 0.1 ,0)) \
    + scale_color_manual(values = ("black") ) \
    + geom_boxplot(color="#00BFC4")


p.save('./Create_Figures/Figures/Motivation_coverage.pdf')

#final = final[final['Type'] != 'CP + BS  \n(baseline)']
p = ggplot(final,
           aes(x="Type", y="Size")) \
    + labs(x="Method", y="Average Set Size", title="") \
    + theme_bw(base_size=base_size)\
    + theme(panel_grid_minor=element_blank(),
            panel_grid_major=element_line(size=0.2, colour="#d3d3d3"),
            plot_title=element_text(face="bold"),
            legend_background=element_rect(fill="white", size=4, colour="white"),
            text=element_text(size=base_size, face="plain"),
            legend_title_align='center',
            legend_position=(0.5, -0.15),
            axis_text_x=element_text(rotation=45, vjust=1, hjust=1),
            axis_title_x=element_blank()) \
    + scale_y_continuous(expand=(0.1, 0, 0.1 ,0)) \
    + geom_boxplot(color="#00BFC4")

p.save('./Create_Figures/Figures/Motivation_size.pdf')