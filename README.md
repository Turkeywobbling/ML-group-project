# INST0060 Group Project - Group 3
This is the code archive by Group 3 for the *INST0060* Group Project.

## Aim of this Project

This project aim to explore the ways of feature extractions on insect flying sounds, and evaluate them by k-Nearest Neighbors (KNN) model. Our methods particularly focus on tranforming the data into variety of representations, specifically time domain and frequency domain.

## Dataset

The dataset consists of 50,000 instances, with each signal split into 10 ms segments over a time-series of length 600. There are 10 classes of varying flying insect species, with 5,000 instances for each: Aedes\_female, Aedes\_male, Fruit\_flies, House\_flies, Quinx\_female, Quinx\_male, Stigma\_female, Stigma\_male, Tarsalis\_female and Tarsalis\_male. Figure 1 depicts the behavior of each class by presenting the average of data points within each respective class. This graphical representation offers a concise overview of the class-specific trends and patterns, derived from the aggregated mean values of the data points.

Average datapoint of each class:
<p align="center">
  <img src="https://github.com/Turkeywobbling/ML_group-project/assets/105172948/12268f50-c081-48ed-a6d3-536d33968fc7" alt="classes", width="600">
</p>

## Code sturcture

### Preprocessing

This part of code separate the label of data apart, rename the label from byte datatype to integer number (0, 1, 2... ,9), then split them into training set and testing set.

### Evaluate model

This part of code set up basic functionaliies of KNN model to evaluate different features for later uses. It also include the K-fold function which allow the model to conduct Cross-Validation in order to gain insights of the overfitting behvaiour of features.

### Methods

This part of code extracted the original data into new features. It includes:

- Baseline - Stats Features

- Fourier Transform

- Temporal Integration

- Spectral Rolloff

- Spectral Density Extraction

- Wavelets Transform

### Feature result

The fitting result of every feature shows in this section, including result table and graphs.

### Tuning Hyperparameter

This part of the code optimiised the hyperparameters of KNN by using Cross-Validation and Grid Search.

## How to Use

### Install Dependencies

Make sure basic Python Libraries are installed. If not, Libraries can be installed by following comments:

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install pandas
pip install random
pip install sklearn
pip install seaborn
```
Make sure dataset is downloaded locally. If not, please install:

```bash
pip install gdown
```
### Run the project

To conduct the projectthe [Insect Sound Dataset](https://www.timeseriesclassification.com/description.php?Dataset=InsectSound) needs to be provided.

Start the experiment with `main.ipynb` by using jupyter notebook. For instance:

```bash
jupyter notebook
```

## Project Structure
- `./insects/` contains all the auxiliary datasets.
- `./main.ipynb` the entry to the programme, see above.
- The rest are project related configurations and IDE settings


