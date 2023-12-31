# INST0060 Group Project - Group 3
This is the code archive by Group 3 for the *INST0060* Group Project.

## Aim of this Project

This project aim to explore the ways of feature extractions on insect flying sounds, and evaluate them by k-Nearest Neighbors (KNN) model. Our methods particularly focus on tranforming the data into variety of representations, specifically time domain and frequency domain.

## Result

Enhancing the initial performance, the original dataset achieved a KNN F1 score of 0.4. Through optimization efforts, the results were significantly boosted, yielding an impressive F1 score of approximately 0.7.

## Dataset

The dataset consists of 50,000 instances, with each signal split into 10 ms segments over a time-series of length 600. There are 10 classes of varying flying insect species, with 5,000 instances for each: 

`Aedes\_female`, `Aedes\_male`, `Fruit\_flies`, `House\_flies`, `Quinx\_female`, `Quinx\_male`, `Stigma\_female`, `Stigma\_male`, `Tarsalis\_female` and `Tarsalis\_male`. 

Average datapoint of each class:
<p align="center">
  <img src="https://github.com/Turkeywobbling/ML-group-project/assets/105172948/adfc1b23-f6d6-4dc1-ac03-9e8cf9775167" alt="classes", width="600">
</p>

## Code structure

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

This part of code can be found in `feature extraction.ipynb`

### Feature result

This part of code can be found in `main.ipynb` and `playground.ipynb`.

The fitting result of every feature shows in this section, including result table and graphs.

### Tuning Hyperparameter

This part of code can be found in `main.ipynb` and `playground.ipynb`.

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

This repository is 1.1 GB in total, in order to pull this repository, please make sure `lfs` installed:

```bash
git lfs install
```

else, you can install gdown can use it in `playground.ipynb`:

```bash
pip install gdown
```

### Run the project

To conduct the projectthe [Insect Sound Dataset](https://www.timeseriesclassification.com/description.php?Dataset=InsectSound) needs to be provided.

Start the experiment with `main.ipynb` by using jupyter notebook.


## Project Structure
- `./insects/` contains original datasets.
- `./features/` contains all the extracted features.
- `./Experiment_Features/` contains the model evaluation of all the extracted features as separate experiments.
- `functions.py` contains all the used functions.
- `feature extraction.ipynb` contains the process of feature engineering
- `main.ipynb` the result of the programme, see above.
- `playground.ipynb` contains the whole process of the programme.
- The rest are project related configurations and IDE settings


