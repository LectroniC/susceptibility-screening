from __future__ import print_function
import os
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Reference: https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html 
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, aspect='auto', **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)


    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

path_string = 'attack_0to1.pkl'
fn = open(path_string, 'rb')
attack_0to1 = pickle.load(fn)

path_string = 'attack_success.pkl'
fn = open(path_string, 'rb')
attack_success = pickle.load(fn)

broadcasted_attack_0to1 = np.zeros(attack_success.shape)
for i in range(attack_success.shape[0]):
    for j in range(attack_success.shape[1]):
        if attack_0to1[i,0] == 1.0:
            broadcasted_attack_0to1[i,:] = np.ones(attack_success.shape[1])

lambda_level = 20
random_patient_id = 1

# Random patient
# Perturbation 0 to 1

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)
data = noise_advs[random_patient_id][lambda_level][0,:,:]

font = {'size'   : 9}
label_fontsize = 12
matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
def skip_other_two(labels):
    for i in range(1,len(labels),2):
        labels[i] = ''
    return labels
y_labels = [str(i) for i in range(48)]
y_labels = skip_other_two(y_labels)

x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=np.min(data), vmax = np.max(data), vcenter=0)
im, cbar = heatmap(data, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("2a_random_patient_pertubation_0to1.pdf",format="pdf",dpi=800)


# Random patient
# Perturbation 0 to 1
# Magnitude vs Sparsity

path_string = 'adv_metrics_0to1.pkl'
fn = open(path_string, 'rb')
adv_metrics = pickle.load(fn)

maxp = adv_metrics[0][random_patient_id]
ap = adv_metrics[1][random_patient_id]
nonz = adv_metrics[2][random_patient_id]/(19*48)


matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
plt.plot(nonz, maxp, 'o', color='blue')
plt.xlabel('Perturbation Percentage',fontsize=label_fontsize)
plt.ylabel('Perturbation Magnitude',fontsize=label_fontsize)
plt.savefig("2b_random_patient_magnitude_sparsity.pdf",format="pdf",dpi=800)


# Random patient
# Perturbation 1 to 0

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)
data = noise_advs[random_patient_id][lambda_level][0,:,:]

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=np.min(data), vmax = np.max(data), vcenter=0)
im, cbar = heatmap(data, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("2c_random_patient_pertubation_1to0.pdf",format="pdf",dpi=800)


# Random patient
# Perturbation 1 to 0
# Magnitude vs Sparsity

path_string = 'adv_metrics_1to0.pkl'
fn = open(path_string, 'rb')
adv_metrics = pickle.load(fn)

maxp = adv_metrics[0][random_patient_id]
nonz = adv_metrics[2][random_patient_id]/(19*48)

matplotlib.rc('font', **font)

plt.figure(figsize=(5, 5))
plt.plot(nonz, maxp, 'o', color='blue')
plt.xlabel('Perturbation Percentage',fontsize=label_fontsize)
plt.ylabel('Perturbation Magnitude',fontsize=label_fontsize)
plt.savefig("2d_random_patient_magnitude_sparsity.pdf",format="pdf",dpi=800)



# Population Level
# Perturbation 0 to 1
# Maximum Perturbation

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

mp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        mp_population_level[i,j] = np.max(abs(all_noises[:,lambda_level,0,i,j]), axis=0)

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
# norm = mcolors.TwoSlopeNorm(vmin=0, vmax = 0.25, vcenter=0.125)
norm = mcolors.TwoSlopeNorm(vmin=np.min(mp_population_level), vmax = np.max(mp_population_level), 
                            vcenter=(np.min(mp_population_level)+np.max(mp_population_level))/2)
im, cbar = heatmap(mp_population_level, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("3a0_population_level_mp_0to1.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 1 to 0
# Maximum Perturbation

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

mp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        mp_population_level[i,j] = np.max(abs(all_noises[:,lambda_level,0,i,j]), axis=0)

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
# norm = mcolors.TwoSlopeNorm(vmin=0, vmax = 0.25, vcenter=0.125)
norm = mcolors.TwoSlopeNorm(vmin=np.min(mp_population_level), vmax = np.max(mp_population_level), 
                            vcenter=(np.min(mp_population_level)+np.max(mp_population_level))/2)
im, cbar = heatmap(mp_population_level, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("3a1_population_level_mp_1to0.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 0 to 1
# Average Perturbation

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

ap_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        # nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        ap_population_level[i,j] = np.sum(abs(all_noises[:,lambda_level,0,i,j]))/all_noises.shape[0]

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=np.min(ap_population_level), vmax = np.max(ap_population_level), 
                                    vcenter=(np.min(ap_population_level)+np.max(ap_population_level))/2)
im, cbar = heatmap(ap_population_level, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("3b0_population_level_ap_0to1.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 1 to 0
# Average Perturbation

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

ap_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        # nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        ap_population_level[i,j] = np.sum(abs(all_noises[:,lambda_level,0,i,j]))/all_noises.shape[0]

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=np.min(ap_population_level), vmax = np.max(ap_population_level), 
                                    vcenter=(np.min(ap_population_level)+np.max(ap_population_level))/2)
im, cbar = heatmap(ap_population_level, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("3b1_population_level_ap_1to0.pdf",format="pdf",dpi=800)



# Population Level
# Perturbation 0 to 1
# Perturbation Probability

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

pp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        pp_population_level[i,j] = nonz/all_noises.shape[0]

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=0, vmax = 1.0, vcenter=0.5)
im, cbar = heatmap(pp_population_level, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("3c0_population_level_pp_0to1.pdf",format="pdf",dpi=800)

# Population Level
# Perturbation 1 to 0
# Perturbation Probability

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

pp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        pp_population_level[i,j] = nonz/all_noises.shape[0]

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=0, vmax = 1.0, vcenter=0.5)
im, cbar = heatmap(pp_population_level, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("3c1_population_level_pp_1to0.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 0 to 1
# Sensitivity Score 

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

pp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
mp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        mp_population_level[i,j] = np.max(abs(all_noises[:,lambda_level,0,i,j]), axis=0)
        nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        pp_population_level[i,j] = nonz/all_noises.shape[0]
sensitivity_score = pp_population_level*mp_population_level

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
# norm = mcolors.TwoSlopeNorm(vmin=0, vmax = 0.25, vcenter=0.125)
norm = mcolors.TwoSlopeNorm(vmin=np.min(sensitivity_score), vmax = np.max(sensitivity_score), 
                                    vcenter=(np.min(sensitivity_score)+np.max(sensitivity_score))/2)
im, cbar = heatmap(sensitivity_score, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Sensitivity Score',fontsize=label_fontsize)
plt.savefig("4a_population_level_ss_0to1.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 0 to 1
# Cumulative Sensitivity Score 

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

pp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
mp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        mp_population_level[i,j] = np.max(abs(all_noises[:,lambda_level,0,i,j]))
        nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        pp_population_level[i,j] = nonz/all_noises.shape[0]
sensitivity_score = pp_population_level*mp_population_level
cumulative_sensitivity_score = np.sum(sensitivity_score,axis=0)
print(cumulative_sensitivity_score)

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
x_labels = [i for i in range(19)]
plt.plot(x_labels, cumulative_sensitivity_score)
x_ticks = range(19)
plt.xlim(xmin=0)
plt.xlim(xmax=18)
plt.xticks(x_ticks, x_ticks, rotation ='horizontal')
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Sensitivity Score',fontsize=label_fontsize)
plt.savefig("4b_population_level_css_0to1.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 0 to 1
# Success Rate vs Maximum Perturbation

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)

path_string = 'adv_metrics_0to1.pkl'
fn = open(path_string, 'rb')
adv_metrics = pickle.load(fn)

MP = adv_metrics[0]
FL = adv_metrics[3]

max_val = 0.6
sampled_points = 10
x = []
y = []
for i in range(sampled_points):
    curr_threshold = max_val * i/sampled_points
    count_success = FL[MP[:,lambda_level] < curr_threshold, lambda_level].sum()
    x.append(curr_threshold)
    y.append(count_success/FL.shape[0])

matplotlib.rc('font', **font)

plt.figure(figsize=(5, 5))
plt.plot(x, y, 'o', color='blue')
plt.xlabel('Maximum Perturbation',fontsize=label_fontsize)
plt.ylabel('Success Rate',fontsize=label_fontsize)
plt.savefig("5a_population_level_mp_vs_sr_0to1.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 0 to 1
# Success Rate vs Perturbation Percentage

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)

path_string = 'adv_metrics_0to1.pkl'
fn = open(path_string, 'rb')
adv_metrics = pickle.load(fn)

NZ = adv_metrics[2]
FL = adv_metrics[3]

max_val = 0.20
sampled_points = 10
x = []
y = []
for i in range(sampled_points):
    curr_threshold = max_val * i/sampled_points
    count_success = FL[NZ[:, lambda_level] < curr_threshold*noise_advs.shape[-1]*noise_advs.shape[-2], lambda_level].sum()
    x.append(curr_threshold)
    y.append(count_success/FL.shape[0])

matplotlib.rc('font', **font)

plt.figure(figsize=(5, 5))
plt.plot(x, y, 'o', color='blue')
plt.xlabel('Perturbation Percentage',fontsize=label_fontsize)
plt.ylabel('Success Rate',fontsize=label_fontsize)
plt.savefig("5b_population_level_pp_vs_sr_0to1.pdf",format="pdf",dpi=800)



# Population Level
# Perturbation 1 to 0
# Sensitivity Score 

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

pp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
mp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        mp_population_level[i,j] = np.max(abs(all_noises[:,lambda_level,0,i,j]), axis=0)
        nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        pp_population_level[i,j] = nonz/all_noises.shape[0]
sensitivity_score = pp_population_level*mp_population_level

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
y_labels = [i for i in range(48)]
y_labels = skip_other_two(y_labels)
x_labels = [i for i in range(19)]
# norm = mcolors.TwoSlopeNorm(vmin=0, vmax = 0.25, vcenter=0.125)
norm = mcolors.TwoSlopeNorm(vmin=np.min(sensitivity_score), vmax = np.max(sensitivity_score), 
                                    vcenter=(np.min(sensitivity_score)+np.max(sensitivity_score))/2)
im, cbar = heatmap(sensitivity_score, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Sensitivity Score',fontsize=label_fontsize)
plt.savefig("7a_population_level_ss_1to0.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 1 to 0
# Cumulative Sensitivity Score 

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
all_noises = pickle.load(fn)

pp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
mp_population_level = np.zeros((all_noises.shape[3],all_noises.shape[4]))
for i in range(all_noises.shape[3]):
    for j in range(all_noises.shape[4]):
        mp_population_level[i,j] = np.max(abs(all_noises[:,lambda_level,0,i,j]))
        nonz = np.count_nonzero(all_noises[:,lambda_level,0,i,j])
        pp_population_level[i,j] = nonz/all_noises.shape[0]
sensitivity_score = pp_population_level*mp_population_level
cumulative_sensitivity_score = np.sum(sensitivity_score,axis=0)
print(cumulative_sensitivity_score)

matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
x_labels = [i for i in range(19)]
plt.plot(x_labels, cumulative_sensitivity_score)
x_ticks = range(19)
plt.xlim(xmin=0)
plt.xlim(xmax=18)
plt.xticks(x_ticks, x_ticks, rotation ='horizontal')
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Sensitivity Score',fontsize=label_fontsize)
plt.savefig("7b_population_level_css_1to0.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 1 to 0
# Success Rate vs Maximum Perturbation

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)

path_string = 'adv_metrics_1to0.pkl'
fn = open(path_string, 'rb')
adv_metrics = pickle.load(fn)

MP = adv_metrics[0]
FL = adv_metrics[3]

max_val = 0.6
sampled_points = 10
x = []
y = []
for i in range(sampled_points):
    curr_threshold = max_val * i/sampled_points
    count_success = FL[MP[:,lambda_level] < curr_threshold, lambda_level].sum()
    x.append(curr_threshold)
    y.append(count_success/FL.shape[0])

matplotlib.rc('font', **font)

plt.figure(figsize=(5, 5))
plt.plot(x, y, 'o', color='blue')
plt.xlabel('Maximum Perturbation',fontsize=label_fontsize)
plt.ylabel('Success Rate',fontsize=label_fontsize)
plt.savefig("8a_population_level_mp_vs_sr_1to0.pdf",format="pdf",dpi=800)


# Population Level
# Perturbation 1 to 0
# Success Rate vs Perturbation Percentage

path_string = 'noise_advs_1to0.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)

path_string = 'adv_metrics_1to0.pkl'
fn = open(path_string, 'rb')
adv_metrics = pickle.load(fn)

NZ = adv_metrics[2]
FL = adv_metrics[3]

max_val = 0.20
sampled_points = 10
x = []
y = []
for i in range(sampled_points):
    curr_threshold = max_val * i/sampled_points
    count_success = FL[NZ[:, lambda_level] < curr_threshold*noise_advs.shape[-1]*noise_advs.shape[-2], lambda_level].sum()
    x.append(curr_threshold)
    y.append(count_success/FL.shape[0])

matplotlib.rc('font', **font)

plt.figure(figsize=(5, 5))
plt.plot(x, y, 'o', color='blue')
plt.xlabel('Perturbation Percentage',fontsize=label_fontsize)
plt.ylabel('Success Rate',fontsize=label_fontsize)
plt.savefig("8b_population_level_pp_vs_sr_1to0.pdf",format="pdf",dpi=800)


# Random patient
# Perturbation 0 to 1
# Large lambda

lambda_level = 24

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)
data = noise_advs[random_patient_id][lambda_level][0,:,:]

font = {'size'   : 9}
label_fontsize = 12
matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
def skip_other_two(labels):
    for i in range(1,len(labels),2):
        labels[i] = ''
    return labels
y_labels = [str(i) for i in range(48)]
y_labels = skip_other_two(y_labels)

x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=np.min(data), vmax = np.max(data), vcenter=0)
im, cbar = heatmap(data, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("large_lambda_random_patient_pertubation_0to1.pdf",format="pdf",dpi=800)


# Random patient
# Perturbation 0 to 1
# Small lambda

lambda_level = 2

path_string = 'noise_advs_0to1.pkl'
fn = open(path_string, 'rb')
noise_advs = pickle.load(fn)
data = noise_advs[random_patient_id][lambda_level][0,:,:]

font = {'size'   : 9}
label_fontsize = 12
matplotlib.rc('font', **font)
plt.figure(figsize=(5, 5))
def skip_other_two(labels):
    for i in range(1,len(labels),2):
        labels[i] = ''
    return labels
y_labels = [str(i) for i in range(48)]
y_labels = skip_other_two(y_labels)

x_labels = [i for i in range(19)]
norm = mcolors.TwoSlopeNorm(vmin=np.min(data), vmax = np.max(data), vcenter=0)
im, cbar = heatmap(data, y_labels, x_labels, ax=None,
                   cmap=plt.cm.RdBu_r, norm=norm)
plt.xlabel('Measurement',fontsize=label_fontsize)
plt.ylabel('Time Stamp',fontsize=label_fontsize)
plt.savefig("small_lambda_random_patient_pertubation_0to1.pdf",format="pdf",dpi=800)