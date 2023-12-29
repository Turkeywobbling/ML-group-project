# imports
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import cwt, ricker, morlet, welch
from scipy.stats import mode, skew, kurtosis, entropy
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from sklearn.neighbors import KNeighborsClassifier

# for random seed for consistency
random_seed = 1222

# for data preprocess-------------------------------------------------------------------------
# split the training and testing set
def split(ratio, data):
    # Group by the label column
    grouped = data.groupby('target')

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    # Split each group into training and testing sets
    for label, group in grouped:
        # Shuffle the group data
        group = group.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        # Calculate the index for splitting
        split_index = int(ratio * len(group))
        # Split into training and testing sets
        group_train = group[:split_index]
        group_test = group[split_index:]

        df_train = pd.concat([df_train, group_train])
        df_test = pd.concat([df_test, group_test])

    # Shuffle the data order
    df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return df_train, df_test

# get a better class labels
def labelset(df):
    original_labels = set(list(df['target']))
    labeldict = {element: i for i, element in enumerate(original_labels, start=0)}
    new_labels = [labeldict[key] for key in list(df['target'])]
    return new_labels

# Data/label Split
def prep(df1, df2):
    # copy the original dataset just in case
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    y_train = labelset(df1)
    y_test = labelset(df2)


    df1_copy.drop('target', inplace=True, axis=1)
    df2_copy.drop('target', inplace=True, axis=1)

    x_train = df1_copy
    x_test = df2_copy

    return x_train, y_train, x_test, y_test

# all together
def pre_features(ratio, data):
    training, testing = split(ratio, data)
    # split it into data and labels
    X_train, y_train, X_test, y_test = prep(training, testing)
    
    return X_train, y_train, X_test, y_test

# calculate the mean of each row
def cal_mean_class(df):
    return df.iloc[:, :-1].mean().values

# evlaution model---------------------------------------------------------------------------------
# knn model
def simple_knn(X_train, y_train, X_test, y_test):
  # Create a KNN classifier
  knn_classifier = KNeighborsClassifier(n_neighbors=10)

  # Train the classifier on the training data
  knn_classifier.fit(X_train, y_train)

  # Make predictions on the test data
  y_pred = knn_classifier.predict(X_test)

  # Evaluate the accuracy of the classifier
  f1 = f1_score(y_test, y_pred, average='weighted')

  return f1

# k folds function for cross validation
def k_folds(X, y, k):
    # Calculate the number of samples and indices
    num_samples = len(y)
    indices = np.arange(num_samples)

    # Shuffle the indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split the indices into k folds
    folds = np.array_split(indices, k)

    # Initialize a list to store the accuracy scores for each fold
    f1_scores = []

    # Perform k-fold cross-validation
    for i in range(k):
        validation_indices = folds[i]
        train_indices = np.concatenate([fold for j, fold in enumerate(folds) if j != i])

        # Use the current fold as the validation set
        validation_X = X.values[validation_indices]
        validation_y = [y[idx] for idx in validation_indices]
        train_X = X.values[train_indices]
        train_y = [y[idx] for idx in train_indices]

    return validation_X, validation_y, train_X, train_y

# knn with cross validation -------------------------------------------------------------------------------
def knn_cv(X_train, y_train, X_test, y_test, k):
  validation_X, validation_y, train_X, train_y = k_folds(X_train, y_train, k)

  knn_classifier = KNeighborsClassifier(n_neighbors=10)
  f1_scores = []
  for i in range(k):
    # Fit the classifier on the training set
    knn_classifier.fit(train_X, train_y)

    # Evaluate the classifier on the validation set
    y_pred = knn_classifier.predict(validation_X)

    # Evaluate the accuracy of the classifier
    f1_scores.append(f1_score(validation_y, y_pred, average='weighted'))

  return np.mean(f1_scores)

# functions that extract features --------------------------------------------------------------------------------
# function the calculate the statics of data
def transform_stats(df):
    stats = {'mean': [], 'median': [], 'mode': [], 'std': [], 'skew': [], 'kurt': []}
    for i in range(len(df)):
        row_data = df.iloc[i].values
        stats['mean'].append(np.mean(row_data))
        stats['median'].append(np.median(row_data))
        stats['mode'].append((mode(row_data).mode)[0])
        stats['std'].append(np.std(row_data))
        stats['skew'].append(skew(row_data))
        stats['kurt'].append(kurtosis(row_data))
    stats_data = pd.DataFrame(stats)
    return stats_data

# FFT Calculation
def calculate_fft(time_series):
    sampling_rate = 6000 # Our data is sampled at 6000Hz as specified by the dataset
    # add abs value to avoid imaginary numbers
    fft_result = np.fft.fft(time_series)
    # Get the power spectrum (magnitude of the FFT)
    magnitude = np.abs(fft_result)
    # Generate frequency axis data
    frequency = np.fft.fftfreq(len(power_spectrum), d=1./sampling_rate)
    # Only take the positive frequencies
    positive_frequency = frequency[:len(frequency)//2]
    positive_power_spectrum = power_spectrum[:len(power_spectrum)//2]

    return positive_frequency * magnitude


# Short-time Fourier Transform Function - Temporal Integration (Highest Stats Features) (71.3% - cross-validation)
def stft_features(signal):

    X_flattened_list = []
    signal = np.array(signal)

    # Perform STFT
    f, t_spec, Zxx = stft(signal, fs=6000, nperseg=128)

    # Summing the spectrogram along the time axis to see how frequency content changes
    temporal_int = np.sum(np.abs(Zxx), axis=1)
    flattened_spec = temporal_int.reshape(-1)
    X_flattened_list.extend(list(flattened_spec))

    return X_flattened_list

#Spectral Rolloff (F1 - 0.45)
def stft_spectral_rolloff_features(signal):

    X_flattened_list = []
    signal = np.array(signal)

    # Perform STFT
    f, t_spec, Zxx = stft(signal, fs=6000, nperseg = 128)  # You can adjust nperseg according to your requirements
    Sxx = np.abs(Zxx)

    spectral_rolloff = np.zeros(Zxx.shape[1])  # Initialize array for each time segment
    for j in range(Zxx.shape[1]):
        total_energy = np.sum(np.abs(Zxx[:, j]) ** 2)
        cumulative_energy = 0.85 * total_energy
        cumulative_sum = 0
        for k in range(len(f)):
            cumulative_sum += np.abs(Zxx[k, j]) ** 2
            if cumulative_sum >= cumulative_energy:
                spectral_rolloff[j] = f[k]
                break

    X_flattened_list.extend(list(spectral_rolloff))

    return X_flattened_list


# function to extract spectral density and frequencies
def spec_features(input):

  fs = 6000 # Because our data is sampled every 10ms

  frequencies, power_density = welch(input, fs, nperseg = 80)
  features = frequencies * power_density

  return features

# wavelet transform function
def wt_transform(input):

  widths = np.arange(1, 15)
  cwtmatr = cwt(input, ricker, widths)

  # Initialize a list to store features for each scale
  all_features = []

  # Extract features for each scale
  for i in range(len(widths)):
      scale_features = []
      scale_coeffs = cwtmatr[i, :]

      # Statistical Features
      scale_features.append(np.mean(scale_coeffs))
      scale_features.append(np.std(scale_coeffs))
      scale_features.append(np.max(scale_coeffs))
      scale_features.append(np.min(scale_coeffs))
      scale_features.append(np.max(scale_coeffs) - np.min(scale_coeffs))

      # Add the feature array for this scale to the list
      all_features.extend(scale_features)

  return all_features

# cross validate function
def cross_validation(X_train, y_train, k, classifier):
    """
    Parameters:
    - X_train: Feature matrix for the training set
    - y_train: Target variable for the training set
    - k: Number of folds
    - classifier: The classifier model

    Returns:
    - List of accuracy scores for each fold
    """

    # Calculate the number of samples and indices
    num_samples = len(y_train)
    indices = np.arange(num_samples)

    # Shuffle the indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split the indices into k folds
    folds = np.array_split(indices, k)

    # Initialize a list to store the accuracy scores for each fold
    f1_scores = []

    # Perform k-fold cross-validation
    for i in range(k):
        validation_indices = folds[i]
        train_indices = np.concatenate([fold for j, fold in enumerate(folds) if j != i])

        # Use the current fold as the validation set
        validation_X = X_train.values[validation_indices]
        validation_y = [y_train[idx] for idx in validation_indices]
        train_X = X_train.values[train_indices]
        train_y = [y_train[idx] for idx in train_indices]

        # Fit the classifier on the training set
        classifier.fit(train_X, train_y)

        # Evaluate the classifier on the validation set
        y_pred = classifier.predict(validation_X)

        # Evaluate the accuracy of the classifier
        f1_scores.append(f1_score(validation_y, y_pred, average='weighted'))

    return f1_scores

def grid_search_cv(X_train, y_train, ns, weights, num_folds):
    mean_cv_f1 = {}
    for weight in weights:
      f1_weights = []
      for n in ns:
        knn = KNeighborsClassifier(n_neighbors=n, weights=weight)
        cross_val_scores = cross_validation(X_train, y_train, num_folds, knn)
        f1_weights.append(np.mean(cross_val_scores))
#       mean_cv_f1.append(f1_weights)
      mean_cv_f1[weight] = f1_weights
    return mean_cv_f1

# for discussion ----------------------------------------------------------------------------------
def pre(train, test, ratio):
    #combined_normalized Training and Test Set
    combined_normalized = pd.concat([train, test], axis=0)
    # split it into training and testing
    training, testing = split(ratio, combined_normalized)
    # split it into data and labels
    X_train, y_train, X_test, y_test = prep(training, testing)

    return X_train, y_train, X_test, y_test
