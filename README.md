# Predicting Car Insurance Claims

## Project Goal

This project is based on two datasets from a French car insurer, aiming to **model the expected claim amount per policyholder and year**. This prediction forms the basis for calculating a fair insurance premium. The datasets contain risk characteristics and damage information for motor liability insurance contracts.

## Data Sources

The datasets used for this project can be found at:

- [freMTPL2freq](https://www.openml.org/d/41214): This dataset contains risk characteristics of the policyholder and the insured vehicle.
- [freMTPL2sev](https://www.openml.org/d/41215): This dataset includes the amount of individual claim expenses during the insurance period.


### Data Dictionary

#### Dataset: freMTPL2freq

- `IDpol`: Contract ID
- `ClaimNb`: Number of claims during the insurance period
- `Exposure`: Length of the insurance period (in years) (component of the dependent variable)
- `Area`: Area code of the policyholder (independent variable)
- `VehPower`: Power of the insured vehicle (independent variable)
- `VehAge`: Age of the insured vehicle (independent variable)
- `DrivAge`: Age of the policyholder (independent variable)
- `BonusMalus`: No-claim discount (French equivalent of the no-claim class) (independent variable)
- `VehBrand`: Brand of the insured vehicle (independent variable)
- `VehGas`: Drive of the insured vehicle (independent variable)
- `Density`: Number of inhabitants per km² at the place of residence of the policyholder (independent variable)
- `Region`: Region of the policyholder (independent variable)

#### Dataset: freMTPL2sev

- `IDpol`: Contract ID
- `ClaimAmount`: Amount of individual claim expenses (multiple entries per contract if multiple claims were present during the period.) (Component of the dependent variable. The dependent variable is defined as ClaimAmount / Exposure.)


## Project Structure

The thought process and results of this project are documented (in German) in the top level `presentation.ipynb` notebook.

All source code is contained in the `src` directory. The `src` directory contains the following files:

- `config.py`: Contains configuration parameters for the project like file paths and model parameters.
- `preprocessing.py`: Contains functions for data preprocessing.
- `modeling.py`: Contains functions for model training.
- `evaluation.py`: Contains functions for model evaluation.

To reproduce the results of this project, run the following files in the `src` directory in the following order:

1. `s1_read_data.py`: Reads the raw data from the data directory and saves them as dataframes to the `data` directory.
1. `s2_run_preprocessing.py`: Data Cleaning and Feature Engineering.
1. `s3_split_data.py`: Splits the data into training and test sets and saves them to the `data` directory.
1. `s4_run_models.py`: Trains all models and saves the fitted models to the `models` directory.



## Setup

This project used [pdm](https://pdm.fming.dev/) as the package and environment manager. See the [pdm documentation](https://pdm.fming.dev/latest/#installation) for installation instructions.

To setup a devlopment environment with all the required dependencies, clone the repository from GitHub, navigate to the project directory and run:

```bash
pdm install
```

This will create a new virtual environment and install all the required dependencies.

