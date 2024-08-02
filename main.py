import csv
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import statistics
from sklearn import preprocessing, metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import plot_confusion_matrix as plot_cm
# from mlxtend.plotting import plot_confusion_matrix as plot_nn_cm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
import tensorflow as tf



# Set the file paths for the 2 data sets to be read
# Convert the data to a Pandas DataFrame


imu_data_path = os.path.abspath("a3_imu_data.csv")

annotations_path = os.path.abspath("a3_activity_annotations.csv")

imu_data = pd.read_csv(imu_data_path, delimiter=",")

imu_data_frame = pd.DataFrame(data=imu_data)

annotations_data = pd.read_csv(annotations_path, delimiter=",")

annotations_data_frame = pd.DataFrame(data=annotations_data)





## Create empty lists to store the separated data for each of the activity set data points

time_track = []
ax_set = []
ay_set = []
az_set = []

gx_set = []
gy_set = []
gz_set = []

activity_set = []




## Create function to map the activity type and return an integer value for that activity

def map_activity(activity_string):
    if activity_string == 'Standing':
        return 0
    elif activity_string == 'Walking':
        return 1
    elif activity_string == 'Jogging':
        return 2
    elif activity_string == 'Side-Step':
        return 3
    elif activity_string == 'Running':
        return 4






## Read in the data as a CSV, and append the lists depending on the label

with open(imu_data_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        time_track.append(float(row[0]))
        ax_set.append(float(row[1]))
        ay_set.append(float(row[2]))
        az_set.append(float(row[3]))
        gx_set.append(float(row[4]))
        gy_set.append(float(row[5]))
        gz_set.append(float(row[6]))






## Create a color list to visualise the data from the plot

fig, axs = plt.subplots(2,1,figsize=(20,10))
fig.suptitle("Raw Data")
axs[0].plot(ax_set, color='r')
axs[0].plot(ay_set, color='g')
axs[0].plot(az_set, color='b')
axs[0].set_title("Accelerometer")
axs[0].set_xlabel("Sampling Increments")
axs[0].set_ylabel("Value")

axs[1].plot(gx_set, color='r')
axs[1].plot(gy_set, color='g')
axs[1].plot(gz_set, color='b')
axs[1].set_title("Gyroscope")
axs[1].set_xlabel("Sampling Increments")
axs[1].set_ylabel("Value")








## Cut the data to the starting point of the activity, removing prior noise and sync the data to the video

start_idx = 825

fig, axs = plt.subplots(2,1,figsize=(20,10))
fig.suptitle("Raw Data After Clipping Start Point")
axs[0].plot(ax_set[start_idx:370000], color='r')
axs[0].plot(ay_set[start_idx:370000], color='g')
axs[0].plot(az_set[start_idx:370000], color='b')
axs[0].set_title("Accelerometer")
axs[0].set_xlabel("Sampling Increments")
axs[0].set_ylabel("Value")

axs[1].plot(gx_set[start_idx:370000], color='r')
axs[1].plot(gy_set[start_idx:370000], color='g')
axs[1].plot(gz_set[start_idx:370000], color='b')
axs[1].set_title("Gyroscope")
axs[1].set_xlabel("Sampling Increments")
axs[1].set_ylabel("Value")








## Find the end point of the video and cut the unnecessary end noise

#Now we need to find the end point
start_ts = time_track[start_idx]
print(start_ts)

# The video has been cut to 443 seconds in length - need to find
# Start_ts + 443 seconds. This should land us at the end point
# We can sanity check this by inspecting the plot

end_idx = time_track.index(start_ts+711)

# Sanity Check
fig, axs = plt.subplots(2,1,figsize=(20,10))
fig.suptitle("Raw Data After Clipping")
axs[0].plot(ax_set[start_idx:end_idx], color='r')
axs[0].plot(ay_set[start_idx:end_idx], color='g')
axs[0].plot(az_set[start_idx:end_idx], color='b')
axs[0].set_title("Accelerometer")
axs[0].set_xlabel("Sampling Increments")
axs[0].set_ylabel("Value")

axs[1].plot(gx_set[start_idx:end_idx], color='r')
axs[1].plot(gy_set[start_idx:end_idx], color='g')
axs[1].plot(gz_set[start_idx:end_idx], color='b')
axs[1].set_title("Gyroscope")
axs[1].set_xlabel("Sampling Increments")
axs[1].set_ylabel("Value")









## Read in the annotations file data as CSV, iterate through the data and separate to get the time stamp

with open(annotations_path, newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for row in reader:
        time_Stamp = (row[0].split('='))[1]
        activity_set.append([float(time_Stamp),row[-1], map_activity(row[-1])])








## Cut the data from the activity set, and find the vertical line of best fit to separate the points

# Now we can create an activity time track for each data point within
# the imu timeseries. We will have a list to store the numeric code and the string.
activity_timeseries = []
activity_string_timeseries = []
activity_idx = 0

# We need the time stamp for the start point
start_time = time_track[start_idx]

# The time track segment we are interested in
time_track_segment = time_track[start_idx:end_idx]

#Add an 'end' activity - this book-ends the data
activity_set.append([time_track_segment[-1]-start_time, 'Standing'])

# Zero out the time track segment to make it match the video time
time_track_segment = np.array(time_track_segment)-start_time

fig, ax = plt.subplots(figsize=(20,10))
fig.suptitle("Raw Data - Clipped and Segmented for Each Activity")
ax.plot(time_track_segment,ax_set[start_idx:end_idx], color='r')
ax.plot(time_track_segment,ay_set[start_idx:end_idx], color='g')
ax.plot(time_track_segment,az_set[start_idx:end_idx], color='b')
ax.set_xlabel("Time Steps")
ax.set_ylabel("Value")

for act in np.array(list(zip(*activity_set)))[0,:]:
    ax.axvline(float(act))








## Plot the end time steps to check the data

for imu_time_track_item in time_track_segment:
    current_time = imu_time_track_item
    next_activity_ts = activity_set[activity_idx + 1][0]

    # Here we need to move to the next activity in the annotations data if the current
    # IMU data point lies after the next annotation time stamp.
    if current_time > next_activity_ts:
        # Move to nex activity
        activity_idx = activity_idx + 1
        next_activity_ts = activity_set[activity_idx + 1][0]

    activity_timeseries.append(activity_set[activity_idx][2])
    activity_string_timeseries.append(activity_set[activity_idx][1])






## Plot a histogram to visualise the distribution

fig, ax = plt.subplots(figsize=(10,10))
ax.hist(activity_timeseries,bins=[-0.5,0.5,1.5,2.5,3.5,4.5], rwidth=0.95)
ax.set_title("Activity Frequency Distribution")
ax.set_ylabel("Activity Count")
ax.set_xticklabels(["", "Standing", "Walking", "Jogging", "Side-Step", "Running"])








## Set the x, y, z sets to the range of start to finish data


#We are only interested in the video window
ax_set = ax_set[start_idx:end_idx]
ay_set = ay_set[start_idx:end_idx]
az_set = az_set[start_idx:end_idx]
gx_set = gx_set[start_idx:end_idx]
gy_set = gy_set[start_idx:end_idx]
gz_set = gz_set[start_idx:end_idx]






## Calculate the Signal Magnitude Area, and the Average Intensity for each label.
## Store the data into features, a target set for the training and testing data based on a 1 second interval from the video window

feature_set = []
target_set = []
window_size = 1.0

for t in range(int(time_track_segment[0]), int(time_track_segment[-1])):
    if not t + window_size in time_track_segment or not t in time_track_segment:
        continue

    # The index function finds the index of the first occurrence of the data
    window_start_idx = list(time_track_segment).index(t)
    window_end_idx = list(time_track_segment).index(t + window_size)
    ax_window = ax_set[window_start_idx:window_end_idx]
    ay_window = ay_set[window_start_idx:window_end_idx]
    az_window = az_set[window_start_idx:window_end_idx]
    gx_window = gx_set[window_start_idx:window_end_idx]
    gy_window = gy_set[window_start_idx:window_end_idx]
    gz_window = gz_set[window_start_idx:window_end_idx]

    # activity that will be assigned to the set of features
    activity_code = activity_timeseries[window_start_idx]

    # Now we can build features from the data window
    # Mean
    mu_ax, mu_ay, mu_az = statistics.mean(ax_window), statistics.mean(ay_window), statistics.mean(az_window)
    mu_gx, mu_gy, mu_gz = statistics.mean(gx_window), statistics.mean(gy_window), statistics.mean(gz_window)

    # Max
    max_ax, max_ay, max_az = max(ax_window), max(ay_window), max(az_window)
    max_gx, max_gy, max_gz = max(gx_window), max(gy_window), max(gz_window)

    # Min
    min_ax, min_ay, min_az = min(ax_window), min(ay_window), min(az_window)
    min_gx, min_gy, min_gz = min(gx_window), min(gy_window), min(gz_window)

    # Sum
    ax_abs_sum, ay_abs_sum, az_abs_sum = 0, 0, 0
    gx_abs_sum, gy_abs_sum, gz_abs_sum = 0, 0, 0

    # Sum of sqrt
    a_sum_sq = 0
    g_sum_sq = 0

    for i in range(0, len(ax_window)):
        # Add up the absolute values for the SMA
        ax_abs_sum = ax_abs_sum + abs(ax_window[i])
        ay_abs_sum = ay_abs_sum + abs(ay_window[i])
        az_abs_sum = az_abs_sum + abs(az_window[i])

        gx_abs_sum = gx_abs_sum + abs(gx_window[i])
        gy_abs_sum = gy_abs_sum + abs(gy_window[i])
        gz_abs_sum = gz_abs_sum + abs(gz_window[i])

        a_sum_sq = ((ax_window[i] ** 2) + (ay_window[i] ** 2) + (az_window[i] ** 2)) + a_sum_sq
        g_sum_sq = ((gx_window[i] ** 2) + (gy_window[i] ** 2) + (gz_window[i] ** 2)) + g_sum_sq

    # Signal Magnitude area
    a_sma = (ax_abs_sum + ay_abs_sum + az_abs_sum) / len(ax_window)
    g_sma = (gx_abs_sum + gy_abs_sum + gz_abs_sum) / len(ax_window)

    # Average intensity
    a_av_intensity = math.sqrt(a_sum_sq) / len(ax_window)
    g_av_intensity = math.sqrt(g_sum_sq) / len(ax_window)

    feature_row = [mu_ax, mu_ay, mu_az, mu_gx, mu_gy, mu_gz,
                   max_ax, max_ay, max_az, max_gx, max_gy, max_gz,
                   min_ax, min_ay, min_az, min_gx, min_gy, min_gz,
                   a_sma, g_sma, a_av_intensity, g_av_intensity]

    feature_set.append(feature_row)
    target_set.append(activity_code)







# Visualise some features, SMA and AI

fig, ax = plt.subplots(figsize=(20,10))
fig.suptitle("Example Features After Feature Engineering")
ax.plot(range(1,705),np.array(feature_set)[:,18], color='r')
ax.plot(range(1,705),np.array(feature_set)[:,19], color='g')
ax.plot(range(1,705),np.array(feature_set)[:,20], color='b')
ax.plot(range(1,705),np.array(feature_set)[:,21], color='c')
ax.set_xlabel("Time Steps")
ax.set_ylabel("Value")





## Create a re-usable function to plot ROC Curves for each of the classifiers

def plot_roc_curve(function=None):
    nclasses = 3
    classifier = function
    x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=.2)
    classifier.fit(x_test, y_test)
    y_score = classifier.predict_proba(x_test)
    y_test_bin = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nclasses):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['cyan', 'magenta', 'purple', 'blue', 'pink']
    plt.plot([0, 1], [0, 1], "r--")
    for i, color in zip(range(nclasses), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    ax.set_xlabel("False Positives")
    ax.set_ylabel("True Positives")
    ax.set_title("ROC Curve of Classifications")
    ax.legend(["y=x", "Walking", "Standing", "Running", "Side-Step", "Jogging"])
    plt.show()
    return None









## Create a function to find the best test and train split for the data

classifiers = ["Standing", "Walking", "Jogging", "Side-Step", "Running"]

c_names = [[i] for i in classifiers]

# Feature names for the tree.
labels = ["Mean Ax", "Mean Ay", "Mean Az","Mean Gx", "Mean Gy", "Mean Gz",
           "Max Ax", "Max Ay", "Max Az","Max Gx", "Max Gy", "Max Gz",
           "Min Ax", "Min Ay", "Min Az","Min Gx", "Min Gy", "Min Gz",
           "Acceleration SMA", "Gyroscope SMA" , "Acceleration AI", "Gyroscope AI"]

f_names = [[i] for i in labels]

def get_training_split():
    test_sizes = [round(float(i * .1), 2) for i in range(1, 9)].__reversed__()
    test_sizes = list(test_sizes)
    knn = neighbors.KNeighborsClassifier(n_neighbors=4)
    plt.figure()

    for test_size in test_sizes:
        scores = []
        for i in range(1, 100):
            x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=1 - test_size)
            knn.fit(x_train, y_train)
            scores.append(knn.score(x_test, y_test))
        plt.plot(test_size, np.mean(scores), "r+")
    plt.plot()
    plt.xlabel("training % split")
    plt.ylabel("predictions")
    plt.show()
    return None









# *** KNN Classifier ***
## Create a function to find the best neighbors for the KNN Classifier

def find_knn_neighbours():
    k_range = range(1,20)
    scores = []
    max_score = []
    for k in k_range:
        x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set)
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        scores.append(knn.score(x_test, y_test))
        max_score.append([(knn.score(x_test, y_test)), k])
    plt.figure()
    plt.xlabel("k neighbours")
    plt.ylabel("predictions")
    plt.scatter(k_range, scores)
    plt.grid()
    plt.xticks([i for i in range(0, 35, 5)])
    plt.xlim([0, 30])
    plt.ylim([0, 1])
    plt.show()
    max_score = max(max_score[1])
    return max_score









## Train and test the data based on the KNN Classifier and produce plots to visually inspect the performance

X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=.2, random_state=42)

# Set the number neighbours to use in the classifier
n_neighbors = find_knn_neighbours()

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

# Train the model
clf.fit(X_train, y_train)

# Return the predictions for the 3-Fold cross-validation
y_predicted = cross_val_predict(clf, X_train, y_train, cv=3)

# Return the predictions for the test set
y_test_predicted = clf.predict(X_test)

# Construct the confusion matrices
conf_mat_train = confusion_matrix(y_train, y_predicted)
conf_mat_test = confusion_matrix(y_test, y_test_predicted)

# store categorical accuracy for comparison to other models
knn_cat_accuracy = list(conf_mat_test.diagonal() / conf_mat_test.sum(axis=1))

# Print out the recall, precision and F1 scores
# There will be a value for each class
# CV Train
print("CV Train Recall:\t", recall_score(y_train, y_predicted, average=None))
print("CV Train Precision:\t", precision_score(y_train, y_predicted, average=None))
print("CV Train F1 Score:\t", f1_score(y_train, y_predicted, average=None))

# Test
print("Test Recall:\t\t", recall_score(y_test, y_test_predicted, average=None))
print("Test Precision:\t\t", precision_score(y_test, y_test_predicted, average=None))
print("Test F1 Score:\t\t", f1_score(y_test, y_test_predicted, average=None))

# Plot the confusion matrices
# cm = confusion_matrix(X_train, y_train)

# Create a ConfusionMatrixDisplay instance
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Jogging', 'Running', 'Side-Step', 'Standing'])
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)

# ax.set_title('3-Fold Training')
# # Plot the confusion matrix
# fig, ax = plt.subplots()
# disp.plot(cmap=plt.cm.Blues, ax=ax)

# # Create the second plot
# fig, ax = plt.subplots()
# cm = confusion_matrix(X_test, y_test)

# # Create a ConfusionMatrixDisplay instance
# cm = confusion_matrix(X_test, y_test)

# # Create a ConfusionMatrixDisplay instance
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Jogging', 'Running', 'Side-Step', 'Standing'])
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)

# ax.set_title('3-Fold Training')
# # Plot the confusion matrix
# fig, ax = plt.subplots()
# disp.plot(cmap=plt.cm.Blues, ax=ax)

# # Show the plot
# plt.show()
# # Rotate the labels
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)

plot_roc_curve(clf)









## Create a function to get the best training split for the RandomForestClassifier

def get_training_split_forrest():
    test_sizes = [round(float(i * .1), 2) for i in range(1, 9)].__reversed__()
    test_sizes = list(test_sizes)
    clf = RandomForestClassifier()
    plt.figure()

    for test_size in test_sizes:
        scores = []
        for i in range(1, 100):
            x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=1 - test_size)
            clf.fit(x_train, y_train)
            scores.append(clf.score(x_test, y_test))
        plt.plot(test_size, np.mean(scores), "r+")
    plt.plot()
    plt.xlabel("training % split")
    plt.ylabel("predictions")
    plt.show()
    return None

get_training_split()











# *** Random Forest Classifier ***
## Test and train the Random Forest Classifier and plot graphs to visualise the data

X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=.2, random_state=42)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)
# Return the predictions for the 3-Fold cross-validation
y_predicted = cross_val_predict(clf, X_train,y_train, cv=3)
# Return the predictions for the test set
y_test_predicted = clf.predict(X_test)
# Construct the confusion matrices
conf_mat_train = confusion_matrix(y_train, y_predicted)
conf_mat_test = confusion_matrix(y_test, y_test_predicted)

# store categorical accuracy for comparison to other models
rf_cat_accuracy = list(conf_mat_test.diagonal()/conf_mat_test.sum(axis=1))

# Print out the recall, precision and F1 scores
# There will be a value for each class
# CV Train
print("CV Train Recall:\t", recall_score(y_train,y_predicted,average=None))
print("CV Train Precision:\t",precision_score(y_train,y_predicted,average=None))
print("CV Train F1 Score:\t",f1_score(y_train,y_predicted,average=None))

# Test
print("Test Recall:\t\t",recall_score(y_test,y_test_predicted,average=None))
print("Test Precision:\t\t",precision_score(y_test,y_test_predicted,average=None))
print("Test F1 Score:\t\t",f1_score(y_test,y_test_predicted,average=None))

# Plot the confusion matrices using the pretty functions
# fig, ax = plt.subplots()
# disp = plot_cm(clf, X_train, y_train,
#                                  display_labels=['Walking','Jogging','Running', 'Side-Step', 'Standing'],
#                                  cmap=plt.cm.Blues,ax=ax)
# # Rotate the labels
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)

# ax.set_title('3-Fold Training')

# fig, ax = plt.subplots()
# disp = plot_cm(clf, X_test, y_test,
#                                  display_labels=['Walking', 'Jogging', 'Running', 'Side-Step', 'Standing'],
#                                  cmap=plt.cm.Blues,ax=ax)
# # Rotate the labels
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)












# *** Stochastic Gradient Descent Classifier ***
## Split and scale data


# split train and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, random_state=42)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)











## Train model and optimise hyper parameters through iteration

from sklearn.metrics import recall_score, precision_score

best_mean_f1 = 0
best_alpha = 0
best_max_iter = 0

alpha_test_values = [0.001,0.005,0.01,0.05]
max_iter_test_values = [100,250,500]

# Iterate through hyper parameter options
for alpha in alpha_test_values:
    for iterations in max_iter_test_values:

        # train classifier
        sgd = SGDClassifier(alpha=alpha, max_iter=iterations, random_state=42)
        sgd.fit(X_train_scaled, y_train)

        # predict using 3-fold cross validation
        y_train_predicted = cross_val_predict(sgd, X_train_scaled, y_train, cv=3)                # TODO: DO I NEED TO SPECIFY FEATURE COLUMNS HERE ??


        current_f1 = np.mean(f1_score(y_train, y_train_predicted, average=None))
        if current_f1 > best_mean_f1:
            best_mean_f1 = current_f1
            best_alpha = alpha
            best_max_iter = iterations
            best_sgd = sgd
            best_y_train_predicted = y_train_predicted


# predict using best SGD classifier
y_test_predicted = best_sgd.predict(X_test_scaled)

print("Hyper Parameter Combination")
print("Best alpha: ", best_alpha)
print("Best max_iter: ", best_max_iter, "\n")
print("CV Train Recall: \t\t", recall_score(y_train, best_y_train_predicted, average=None))
print("CV Train Precision: \t", precision_score(y_train, best_y_train_predicted, average=None))
print("CV Train F1: \t\t\t", f1_score(y_train, best_y_train_predicted, average=None))
print("Test Recall: \t\t\t", recall_score(y_test, y_test_predicted, average=None))
print("Test Precision: \t\t", precision_score(y_test, y_test_predicted, average=None))
print("Test F1: \t\t\t\t", f1_score(y_test, y_test_predicted, average=None))













## Plot Confusion Matrices

# plot confusion matrix for train set
# fig, ax = plt.subplots()
# ax.set_title("Train")
# disp = plot_cm(best_sgd, X_train_scaled, y_train,
#                              display_labels=["Standing", "Walking", "Jogging", "Side-Step", "Running"],
#                              cmap=plt.cm.Blues, ax=ax)

# # plot confusion matrix for test set
# fig, ax = plt.subplots()
# ax.set_title("Test")
# disp = plot_cm(best_sgd, X_test_scaled, y_test,
#                              display_labels=["Standing", "Walking", "Jogging", "Side-Step", "Running"],
#                              cmap=plt.cm.Blues, ax=ax)

# Store categorical accuracy for comparison to other models
conf_mat_test = confusion_matrix(y_test, y_test_predicted)
sgd_cat_accuracy = list(conf_mat_test.diagonal()/conf_mat_test.sum(axis=1))












# *** Neural Network ***
## Split and scale data



# split train and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, random_state=42)

# convert to numpy arrays
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# apply MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert target classes to categorical array for neural network
y_train_cat_array = to_categorical(y_train, num_classes=5)
y_test_cat_array = to_categorical(y_test, num_classes=5)












## Build and train NN model

# NN input layer size
input_size = len(X_test_scaled[0])
# set random state
tf.random.set_seed(42)
# declare variable for activation function to test various functions
activation_function = 'relu'

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_size, activation=activation_function),
    tf.keras.layers.Dense(128, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(128, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(128, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(64, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(32, activation=activation_function),
    tf.keras.layers.Dense(5, activation='softmax')
])

# compile model
model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['categorical_accuracy']
)

# train model
history = model.fit(
    X_train_scaled,
    y_train_cat_array,
    epochs=250,
    validation_data=(X_test_scaled, y_test_cat_array),      # Ideally a validation split would be used here, but for this example the test data has been used for validation
    batch_size=32
)












## Visualise accuracy and loss curves

# Function to plot loss and accuracy metrics for training and validation dataset
def plot_loss_curves(history_object):
    """
    Plots loss and accuracy metrics for training and validation dataset
    """

    loss = history_object.history['loss']
    val_loss = history_object.history['val_loss']
    accuracy = history_object.history['categorical_accuracy']
    val_accuracy = history_object.history['val_categorical_accuracy']
    epochs = range(len(history_object.history['loss']))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
    fig.suptitle("Loss and Accuracy Training Curves")
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.95)

    # plot loss
    ax[0].plot(epochs, loss, label='training_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel("Loss Value")
    ax[0].legend()

    # plot accuracy
    ax[1].plot(epochs, accuracy, label='training_cat_accuracy')
    ax[1].plot(epochs, val_accuracy, label='val_cat_accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel("Accuracy Score")
    ax[1].legend()
    return None










## Plot confusion matrix
# make predictions with NN model
y_predicted = model.predict(X_test_scaled)
# convert array of predicted probabilities to single column of predictions
y_predicted = np.argmax(y_predicted, axis=1)

matrix = confusion_matrix(y_test, y_predicted)
# plot_nn_cm(conf_mat=matrix, class_names=['Standing', 'Walking', 'Jogging', 'Side-Step', 'Running'], show_normed=True)
# Store categorical accuracy for comparison to other models
nn_cat_accuracy = list(matrix.diagonal()/matrix.sum(axis=1))



















# Model Performance Comparison

# chart up categorical accuracy score for each model in histogram
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
fig.suptitle("Categorical Accuracy - Model Comparison \n(Test Dataset)")
fig.subplots_adjust(left=0.05)
fig.subplots_adjust(right=0.95)

# plot categorical accuracy for each classifier
X_axis1 = np.arange(len(knn_cat_accuracy))
#X_axis1 = ["Standing", "Walking", "Jogging", "Side-Step", "Running"]
bar_width = 0.2
ax[0].bar(X_axis1 -0.0, knn_cat_accuracy, bar_width, label="KNN Classifier")
ax[0].bar(X_axis1 +0.2, rf_cat_accuracy, bar_width, label="Random Forest")
ax[0].bar(X_axis1 +0.4, sgd_cat_accuracy, bar_width, label="SGD classifier")
ax[0].bar(X_axis1 +0.6, nn_cat_accuracy, bar_width, label="Neural Network")

ax[0].set_title("Categorical Accuracy")
ax[0].set_ylabel("Categorical Accuracy Score")
ax[0].set_ylim([0.8, 1.0])
# TODO: SET X LABELS FOR EACH CLASS
ax[0].set_xticklabels(["", "Standing", "Walking", "Jogging", "Side-Step", "Running"])
ax[0].legend()

# plot average categorical accuracy for each classifier
X_axis2 = ['KNN', 'RF', 'SGD', 'NN']
avg_cat_accuracy = [np.mean(knn_cat_accuracy), np.mean(rf_cat_accuracy), np.mean(sgd_cat_accuracy), np.mean(nn_cat_accuracy)]
ax[1].bar(X_axis2, avg_cat_accuracy)
ax[1].set_ylim(0.9, 1.0)
ax[1].set_title("Average Categorical Accuracy")
ax[1].set_ylabel("Categorical Accuracy Score")








time_track = []
ax_set = []
ay_set = []
az_set = []

gx_set = []
gy_set = []
gz_set = []

activity_set = []


def map_activity(activity_string):
    if activity_string == 'Standing':
        return 0
    elif activity_string == 'Walking':
        return 1
    elif activity_string == 'Jogging':
        return 2
    elif activity_string == 'Side-Step':
        return 3
    elif activity_string == 'Running':
        return 4


with open(imu_data_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        time_track.append(float(row[0]))
        ax_set.append(float(row[1]))
        ay_set.append(float(row[2]))
        az_set.append(float(row[3]))
        gx_set.append(float(row[4]))
        gy_set.append(float(row[5]))
        gz_set.append(float(row[6]))

fig, axs = plt.subplots(2, 1, figsize=(20, 10))
fig.suptitle("Raw Data")
axs[0].plot(ax_set, color='r')
axs[0].plot(ay_set, color='g')
axs[0].plot(az_set, color='b')
axs[0].set_title("Accelerometer")
axs[0].set_xlabel("Sampling Increments")
axs[0].set_ylabel("Value")

axs[1].plot(gx_set, color='r')
axs[1].plot(gy_set, color='g')
axs[1].plot(gz_set, color='b')
axs[1].set_title("Gyroscope")
axs[1].set_xlabel("Sampling Increments")
axs[1].set_ylabel("Value")

start_idx = 825

fig, axs = plt.subplots(2, 1, figsize=(20, 10))
fig.suptitle("Raw Data After Clipping Start Point")
axs[0].plot(ax_set[start_idx:370000], color='r')
axs[0].plot(ay_set[start_idx:370000], color='g')
axs[0].plot(az_set[start_idx:370000], color='b')
axs[0].set_title("Accelerometer")
axs[0].set_xlabel("Sampling Increments")
axs[0].set_ylabel("Value")

axs[1].plot(gx_set[start_idx:370000], color='r')
axs[1].plot(gy_set[start_idx:370000], color='g')
axs[1].plot(gz_set[start_idx:370000], color='b')
axs[1].set_title("Gyroscope")
axs[1].set_xlabel("Sampling Increments")
axs[1].set_ylabel("Value")

# Now we need to find the end point
start_ts = time_track[start_idx]
print(start_ts)

# The video has been cut to 443 seconds in length - need to find
# Start_ts + 443 seconds. This should land us at the end point
# We can sanity check this by inspecting the plot

end_idx = time_track.index(start_ts + 711)

# Sanity Check
fig, axs = plt.subplots(2, 1, figsize=(20, 10))
fig.suptitle("Raw Data After Clipping")
axs[0].plot(ax_set[start_idx:end_idx], color='r')
axs[0].plot(ay_set[start_idx:end_idx], color='g')
axs[0].plot(az_set[start_idx:end_idx], color='b')
axs[0].set_title("Accelerometer")
axs[0].set_xlabel("Sampling Increments")
axs[0].set_ylabel("Value")

axs[1].plot(gx_set[start_idx:end_idx], color='r')
axs[1].plot(gy_set[start_idx:end_idx], color='g')
axs[1].plot(gz_set[start_idx:end_idx], color='b')
axs[1].set_title("Gyroscope")
axs[1].set_xlabel("Sampling Increments")
axs[1].set_ylabel("Value")

with open(annotations_path, newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for row in reader:
        time_Stamp = (row[0].split('='))[1]
        activity_set.append([float(time_Stamp), row[-1], map_activity(row[-1])])

# Now we can create an activity time track for each data point within
# the imu timeseries. We will have a list to store the numeric code and the string.
activity_timeseries = []
activity_string_timeseries = []
activity_idx = 0

# We need the time stamp for the start point
start_time = time_track[start_idx]

# The time track segment we are interested in
time_track_segment = time_track[start_idx:end_idx]

# Add an 'end' activity - this book-ends the data
activity_set.append([time_track_segment[-1] - start_time, 'Standing'])

# Zero out the time track segment to make it match the video time
time_track_segment = np.array(time_track_segment) - start_time

fig, ax = plt.subplots(figsize=(20, 10))
fig.suptitle("Raw Data - Clipped and Segmented for Each Activity")
ax.plot(time_track_segment, ax_set[start_idx:end_idx], color='r')
ax.plot(time_track_segment, ay_set[start_idx:end_idx], color='g')
ax.plot(time_track_segment, az_set[start_idx:end_idx], color='b')
ax.set_xlabel("Time Steps")
ax.set_ylabel("Value")

for act in np.array(list(zip(*activity_set)))[0, :]:
    ax.axvline(float(act))

for imu_time_track_item in time_track_segment:
    current_time = imu_time_track_item
    next_activity_ts = activity_set[activity_idx + 1][0]

    # Here we need to move to the next activity in the annotations data if the current
    # IMU data point lies after the next annotation time stamp.
    if current_time > next_activity_ts:
        # Move to nex activity
        activity_idx = activity_idx + 1
        next_activity_ts = activity_set[activity_idx + 1][0]

    activity_timeseries.append(activity_set[activity_idx][2])
    activity_string_timeseries.append(activity_set[activity_idx][1])

fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(activity_timeseries, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], rwidth=0.95)
ax.set_title("Activity Frequency Distribution")
ax.set_ylabel("Activity Count")
ax.set_xticklabels(["", "Standing", "Walking", "Jogging", "Side-Step", "Running"])

# We are only interested in the video window
ax_set = ax_set[start_idx:end_idx]
ay_set = ay_set[start_idx:end_idx]
az_set = az_set[start_idx:end_idx]
gx_set = gx_set[start_idx:end_idx]
gy_set = gy_set[start_idx:end_idx]
gz_set = gz_set[start_idx:end_idx]

"""

Calculate the Signal Magnitude Area, and the Average Intensity for each label.
Store the data into features, a target set for the training and testing data based on a 1 second interval from the video window
The Signal Magnitude Area: SMA=1T(T∑t=1|ax(t)|+T∑t=1|ay(t)|+(T∑t=1|az(t)|)
Average intensity:
AI=1T(T∑t=1(√ax(t)2+ay(t)2+ay(t)2)

"""

feature_set = []
target_set = []
window_size = 1.0

for t in range(int(time_track_segment[0]), int(time_track_segment[-1])):
    if not t + window_size in time_track_segment or not t in time_track_segment:
        continue

    # The index function finds the index of the first occurrence of the data
    window_start_idx = list(time_track_segment).index(t)
    window_end_idx = list(time_track_segment).index(t + window_size)
    ax_window = ax_set[window_start_idx:window_end_idx]
    ay_window = ay_set[window_start_idx:window_end_idx]
    az_window = az_set[window_start_idx:window_end_idx]
    gx_window = gx_set[window_start_idx:window_end_idx]
    gy_window = gy_set[window_start_idx:window_end_idx]
    gz_window = gz_set[window_start_idx:window_end_idx]

    # activity that will be assigned to the set of features
    activity_code = activity_timeseries[window_start_idx]

    # Now we can build features from the data window
    # Mean
    mu_ax, mu_ay, mu_az = statistics.mean(ax_window), statistics.mean(ay_window), statistics.mean(az_window)
    mu_gx, mu_gy, mu_gz = statistics.mean(gx_window), statistics.mean(gy_window), statistics.mean(gz_window)

    # Max
    max_ax, max_ay, max_az = max(ax_window), max(ay_window), max(az_window)
    max_gx, max_gy, max_gz = max(gx_window), max(gy_window), max(gz_window)

    # Min
    min_ax, min_ay, min_az = min(ax_window), min(ay_window), min(az_window)
    min_gx, min_gy, min_gz = min(gx_window), min(gy_window), min(gz_window)

    # Sum
    ax_abs_sum, ay_abs_sum, az_abs_sum = 0, 0, 0
    gx_abs_sum, gy_abs_sum, gz_abs_sum = 0, 0, 0

    # Sum of sqrt
    a_sum_sq = 0
    g_sum_sq = 0

    for i in range(0, len(ax_window)):
        # Add up the absolute values for the SMA
        ax_abs_sum = ax_abs_sum + abs(ax_window[i])
        ay_abs_sum = ay_abs_sum + abs(ay_window[i])
        az_abs_sum = az_abs_sum + abs(az_window[i])

        gx_abs_sum = gx_abs_sum + abs(gx_window[i])
        gy_abs_sum = gy_abs_sum + abs(gy_window[i])
        gz_abs_sum = gz_abs_sum + abs(gz_window[i])

        a_sum_sq = ((ax_window[i] ** 2) + (ay_window[i] ** 2) + (az_window[i] ** 2)) + a_sum_sq
        g_sum_sq = ((gx_window[i] ** 2) + (gy_window[i] ** 2) + (gz_window[i] ** 2)) + g_sum_sq

    # Signal Magnitude area
    a_sma = (ax_abs_sum + ay_abs_sum + az_abs_sum) / len(ax_window)
    g_sma = (gx_abs_sum + gy_abs_sum + gz_abs_sum) / len(ax_window)

    # Average intensity
    a_av_intensity = math.sqrt(a_sum_sq) / len(ax_window)
    g_av_intensity = math.sqrt(g_sum_sq) / len(ax_window)

    feature_row = [mu_ax, mu_ay, mu_az, mu_gx, mu_gy, mu_gz,
                   max_ax, max_ay, max_az, max_gx, max_gy, max_gz,
                   min_ax, min_ay, min_az, min_gx, min_gy, min_gz,
                   a_sma, g_sma, a_av_intensity, g_av_intensity]

    feature_set.append(feature_row)
    target_set.append(activity_code)

fig, ax = plt.subplots(figsize=(20, 10))
fig.suptitle("Example Features After Feature Engineering")
ax.plot(range(1, 705), np.array(feature_set)[:, 18], color='r')
ax.plot(range(1, 705), np.array(feature_set)[:, 19], color='g')
ax.plot(range(1, 705), np.array(feature_set)[:, 20], color='b')
ax.plot(range(1, 705), np.array(feature_set)[:, 21], color='c')
ax.set_xlabel("Time Steps")
ax.set_ylabel("Value")


def plot_roc_curve(function=None):
    nclasses = 5
    classifier = function
    x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=.2)
    classifier.fit(x_test, y_test)
    y_score = classifier.predict_proba(x_test)
    y_test_bin = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nclasses):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['cyan', 'magenta', 'purple', 'blue', 'pink']
    plt.plot([0, 1], [0, 1], "r--")
    for i, color in zip(range(nclasses), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    ax.set_xlabel("False Positives")
    ax.set_ylabel("True Positives")
    ax.set_title("ROC Curve of Classifications")
    ax.legend(["y=x", "Walking", "Standing", "Running", "Side-Step", "Jogging"])
    plt.show()
    return None


classifiers = ["Standing", "Walking", "Jogging", "Side-Step", "Running"]

c_names = [[i] for i in classifiers]

# Feature names for the tree.
labels = ["Mean Ax", "Mean Ay", "Mean Az", "Mean Gx", "Mean Gy", "Mean Gz",
          "Max Ax", "Max Ay", "Max Az", "Max Gx", "Max Gy", "Max Gz",
          "Min Ax", "Min Ay", "Min Az", "Min Gx", "Min Gy", "Min Gz",
          "Acceleration SMA", "Gyroscope SMA", "Acceleration AI", "Gyroscope AI"]

f_names = [[i] for i in labels]


def get_training_split():
    test_sizes = [round(float(i * .1), 2) for i in range(1, 9)].__reversed__()
    test_sizes = list(test_sizes)
    knn = neighbors.KNeighborsClassifier(n_neighbors=4)
    plt.figure()

    for test_size in test_sizes:
        scores = []
        for i in range(1, 100):
            x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=1 - test_size)
            knn.fit(x_train, y_train)
            scores.append(knn.score(x_test, y_test))
        plt.plot(test_size, np.mean(scores), "r+")
    plt.plot()
    plt.xlabel("training % split")
    plt.ylabel("predictions")
    plt.show()
    return None


def find_knn_neighbours():
    k_range = range(1, 20)
    scores = []
    max_score = []
    for k in k_range:
        x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set)
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        scores.append(knn.score(x_test, y_test))
        max_score.append([(knn.score(x_test, y_test)), k])
    plt.figure()
    plt.xlabel("k neighbours")
    plt.ylabel("predictions")
    plt.scatter(k_range, scores)
    plt.grid()
    plt.xticks([i for i in range(0, 35, 5)])
    plt.xlim([0, 30])
    plt.ylim([0, 1])
    plt.show()
    max_score = max(max_score[1])
    return max_score


X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=.2, random_state=42)

# Set the number neighbours to use in the classifier
n_neighbors = find_knn_neighbours()

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

# Train the model
clf.fit(X_train, y_train)

# Return the predictions for the 3-Fold cross-validation
y_predicted = cross_val_predict(clf, X_train, y_train, cv=3)

# Return the predictions for the test set
y_test_predicted = clf.predict(X_test)

# Construct the confusion matrices
conf_mat_train = confusion_matrix(y_train, y_predicted)
conf_mat_test = confusion_matrix(y_test, y_test_predicted)

# store categorical accuracy for comparison to other models
knn_cat_accuracy = list(conf_mat_test.diagonal() / conf_mat_test.sum(axis=1))

# Print out the recall, precision and F1 scores
# There will be a value for each class
# CV Train
print("CV Train Recall:\t", recall_score(y_train, y_predicted, average=None))
print("CV Train Precision:\t", precision_score(y_train, y_predicted, average=None))
print("CV Train F1 Score:\t", f1_score(y_train, y_predicted, average=None))

# Test
print("Test Recall:\t\t", recall_score(y_test, y_test_predicted, average=None))
print("Test Precision:\t\t", precision_score(y_test, y_test_predicted, average=None))
print("Test F1 Score:\t\t", f1_score(y_test, y_test_predicted, average=None))


# ******************************************************************************************************************

# Generate the confusion matrix
# cm = confusion_matrix(X_train, y_train)

# Create a ConfusionMatrixDisplay instance
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Jogging', 'Running', 'Side-Step', 'Standing'])
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)

# ax.set_title('Before 3-Fold Training')
# # Plot the confusion matrix
# fig, ax = plt.subplots()
# disp.plot(cmap=plt.cm.Blues, ax=ax)

# Show the plot
# plt.show()
# ******************************************************************************************************
# Generate the confusion matrix
# cm = confusion_matrix(X_test, y_test)
#
# # Create a ConfusionMatrixDisplay instance
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Jogging', 'Running', 'Side-Step', 'Standing'])
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)
#
# ax.set_title('3-Fold Training')
# # Plot the confusion matrix
# fig, ax = plt.subplots()
# disp.plot(cmap=plt.cm.Blues, ax=ax)

# Show the plot
# plt.show()
# # Rotate the labels
# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)

plot_roc_curve(clf)


def get_training_split_forrest():
    test_sizes = [round(float(i * .1), 2) for i in range(1, 9)].__reversed__()
    test_sizes = list(test_sizes)
    clf = RandomForestClassifier()
    plt.figure()

    for test_size in test_sizes:
        scores = []
        for i in range(1, 100):
            x_train, x_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=1 - test_size)
            clf.fit(x_train, y_train)
            scores.append(clf.score(x_test, y_test))
        plt.plot(test_size, np.mean(scores), "r+")
    plt.plot()
    plt.xlabel("training % split")
    plt.ylabel("predictions")
    plt.show()
    return None


get_training_split()

X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=.2, random_state=42)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)
# Return the predictions for the 3-Fold cross-validation
y_predicted = cross_val_predict(clf, X_train, y_train, cv=3)
# Return the predictions for the test set
y_test_predicted = clf.predict(X_test)
# Construct the confusion matrices
conf_mat_train = confusion_matrix(y_train, y_predicted)
conf_mat_test = confusion_matrix(y_test, y_test_predicted)

# store categorical accuracy for comparison to other models
rf_cat_accuracy = list(conf_mat_test.diagonal() / conf_mat_test.sum(axis=1))

# Print out the recall, precision and F1 scores
# There will be a value for each class
# CV Train
print("CV Train Recall:\t", recall_score(y_train, y_predicted, average=None))
print("CV Train Precision:\t", precision_score(y_train, y_predicted, average=None))
print("CV Train F1 Score:\t", f1_score(y_train, y_predicted, average=None))

# Test
print("Test Recall:\t\t", recall_score(y_test, y_test_predicted, average=None))
print("Test Precision:\t\t", precision_score(y_test, y_test_predicted, average=None))
print("Test F1 Score:\t\t", f1_score(y_test, y_test_predicted, average=None))

# Plot the confusion matrices using the pretty functions
# *********************************************************************************
# Generate the confusion matrix
cm = confusion_matrix(X_train, y_train)

# Create a ConfusionMatrixDisplay instance
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Jogging', 'Running', 'Side-Step', 'Standing'])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

ax.set_title('3-Fold Training')
# Plot the confusion matrix
fig, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Show the plot
plt.show()
# Rotate the labels
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

# *********************************************************************************
ax.set_title('3-Fold Training')

# Generate the confusion matrix
cm = confusion_matrix(X_test, y_test)

# Create a ConfusionMatrixDisplay instance
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Jogging', 'Running', 'Side-Step', 'Standing'])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

ax.set_title('3-Fold Training')
# Plot the confusion matrix
fig, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Show the plot
plt.show()
# Rotate the labels
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

# stochastic
# split train and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, random_state=42)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.metrics import recall_score, precision_score

best_mean_f1 = 0
best_alpha = 0
best_max_iter = 0

alpha_test_values = [0.001, 0.005, 0.01, 0.05]
max_iter_test_values = [100, 250, 500]

# Iterate through hyper parameter options
for alpha in alpha_test_values:
    for iterations in max_iter_test_values:

        # train classifier
        sgd = SGDClassifier(alpha=alpha, max_iter=iterations, random_state=42)
        sgd.fit(X_train_scaled, y_train)

        # predict using 3-fold cross validation
        y_train_predicted = cross_val_predict(sgd, X_train_scaled, y_train,
                                              cv=3)  # TODO: DO I NEED TO SPECIFY FEATURE COLUMNS HERE ??

        current_f1 = np.mean(f1_score(y_train, y_train_predicted, average=None))
        if current_f1 > best_mean_f1:
            best_mean_f1 = current_f1
            best_alpha = alpha
            best_max_iter = iterations
            best_sgd = sgd
            best_y_train_predicted = y_train_predicted

# predict using best SGD classifier
y_test_predicted = best_sgd.predict(X_test_scaled)

print("Hyper Parameter Combination")
print("Best alpha: ", best_alpha)
print("Best max_iter: ", best_max_iter, "\n")
print("CV Train Recall: \t\t", recall_score(y_train, best_y_train_predicted, average=None))
print("CV Train Precision: \t", precision_score(y_train, best_y_train_predicted, average=None))
print("CV Train F1: \t\t\t", f1_score(y_train, best_y_train_predicted, average=None))
print("Test Recall: \t\t\t", recall_score(y_test, y_test_predicted, average=None))
print("Test Precision: \t\t", precision_score(y_test, y_test_predicted, average=None))
print("Test F1: \t\t\t\t", f1_score(y_test, y_test_predicted, average=None))

# plot confusion matrix for train set
#######################################################################
# fig, ax = plt.subplots()
# ax.set_title("Train")
# disp = plot_cm(best_sgd, X_train_scaled, y_train,
#                display_labels=["Standing", "Walking", "Jogging", "Side-Step", "Running"],
#                cmap=plt.cm.Blues, ax=ax)

# Generate predictions for the training data
y_train_pred = best_sgd.predict(X_train_scaled)

# Compute the confusion matrix
cm = confusion_matrix(y_train, y_train_pred)

# Create a ConfusionMatrixDisplay instance and plot
fig, ax = plt.subplots()
ax.set_title("Train")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Standing", "Walking", "Jogging", "Side-Step", "Running"])
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Show the plot
plt.show()
############################################################
# plot confusion matrix for test set
# fig, ax = plt.subplots()
# ax.set_title("Test")
# disp = plot_cm(best_sgd, X_test_scaled, y_test,
#                display_labels=["Standing", "Walking", "Jogging", "Side-Step", "Running"],
#                cmap=plt.cm.Blues, ax=ax)

# Generate predictions for the test data
y_test_pred = best_sgd.predict(X_test_scaled)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Create a ConfusionMatrixDisplay instance and plot
fig, ax = plt.subplots()
ax.set_title("Test")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Standing", "Walking", "Jogging", "Side-Step", "Running"])
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Show the plot
plt.show()
# ***********************************************************
# Store categorical accuracy for comparison to other models
conf_mat_test = confusion_matrix(y_test, y_test_predicted)
sgd_cat_accuracy = list(conf_mat_test.diagonal() / conf_mat_test.sum(axis=1))

# split train and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, random_state=42)

# convert to numpy arrays
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# apply MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert target classes to categorical array for neural network
y_train_cat_array = to_categorical(y_train, num_classes=5)
y_test_cat_array = to_categorical(y_test, num_classes=5)

# NN input layer size
input_size = len(X_test_scaled[0])
# set random state
tf.random.set_seed(42)
# declare variable for activation function to test various functions
activation_function = 'relu'

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_size, activation=activation_function),
    tf.keras.layers.Dense(128, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(128, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(128, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(64, activation=activation_function),
    tf.keras.layers.Dropout(rate=0.25, seed=42),
    tf.keras.layers.Dense(32, activation=activation_function),
    tf.keras.layers.Dense(5, activation='softmax')
])

# compile model
model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['categorical_accuracy']
)

# train model
history = model.fit(
    X_train_scaled,
    y_train_cat_array,
    epochs=250,
    validation_data=(X_test_scaled, y_test_cat_array),
    # Ideally a validation split would be used here, but for our example the test data has been used for validation
    batch_size=32
)

print("*****    PERFORMANCE METRICS    ***** ")
print(" MAX VAL ACCURACY: " + str(max(history.history['val_categorical_accuracy'])))
print("            EPOCH: " + str(
    history.history['val_categorical_accuracy'].index(max(history.history['val_categorical_accuracy'])) + 1))
print("     MIN VAL LOSS: " + str(min(history.history['val_loss'])))
print("            EPOCH: " + str(history.history['val_loss'].index(min(history.history['val_loss'])) + 1))


# Function to plot loss and accuracy metrics for training and validation dataset
def plot_loss_curves(history_object):
    """
    Plots loss and accuracy metrics for training and validation dataset
    """

    loss = history_object.history['loss']
    val_loss = history_object.history['val_loss']
    accuracy = history_object.history['categorical_accuracy']
    val_accuracy = history_object.history['val_categorical_accuracy']
    epochs = range(len(history_object.history['loss']))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    fig.suptitle("Loss and Accuracy Training Curves")
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.95)

    # plot loss
    ax[0].plot(epochs, loss, label='training_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel("Loss Value")
    ax[0].legend()

    # plot accuracy
    ax[1].plot(epochs, accuracy, label='training_cat_accuracy')
    ax[1].plot(epochs, val_accuracy, label='val_cat_accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel("Accuracy Score")
    ax[1].legend()
    return None


plot_loss_curves(history)

# make predictions with NN model
y_predicted = model.predict(X_test_scaled)
# convert array of predicted probabilities to single column of predictions
y_predicted = np.argmax(y_predicted, axis=1)
matrix = confusion_matrix(y_test, y_predicted)
# plot_nn_cm(conf_mat=matrix, class_names=['Standing', 'Walking', 'Jogging', 'Side-Step', 'Running'], show_normed=True)
# Store categorical accuracy for comparison to other models
nn_cat_accuracy = list(matrix.diagonal() / matrix.sum(axis=1))
# chart up categorical accuracy score for each model in histogram
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
fig.suptitle("Categorical Accuracy - Model Comparison \n(Test Dataset)")
fig.subplots_adjust(left=0.05)
fig.subplots_adjust(right=0.95)

# plot categorical accuracy for each classifier
X_axis1 = np.arange(len(knn_cat_accuracy))
# X_axis1 = ["Standing", "Walking", "Jogging", "Side-Step", "Running"]
bar_width = 0.2
ax[0].bar(X_axis1 - 0.0, knn_cat_accuracy, bar_width, label="KNN Classifier")
ax[0].bar(X_axis1 + 0.2, rf_cat_accuracy, bar_width, label="Random Forest")
ax[0].bar(X_axis1 + 0.4, sgd_cat_accuracy, bar_width, label="SGD classifier")
ax[0].bar(X_axis1 + 0.6, nn_cat_accuracy, bar_width, label="Neural Network")

ax[0].set_title("Categorical Accuracy")
ax[0].set_ylabel("Categorical Accuracy Score")
ax[0].set_ylim([0.8, 1.0])
# TODO: SET X LABELS FOR EACH CLASS
ax[0].set_xticklabels(["", "Standing", "Walking", "Jogging", "Side-Step", "Running"])
ax[0].legend()

# plot average categorical accuracy for each classifier
X_axis2 = ['KNN', 'RF', 'SGD', 'NN']
avg_cat_accuracy = [np.mean(knn_cat_accuracy), np.mean(rf_cat_accuracy), np.mean(sgd_cat_accuracy),
                    np.mean(nn_cat_accuracy)]
ax[1].bar(X_axis2, avg_cat_accuracy)
ax[1].set_ylim(0.9, 1.0)
ax[1].set_title("Average Categorical Accuracy")
ax[1].set_ylabel("Categorical Accuracy Score")