# Repository for experienced temperature project

## Cloning the repository
To clone the repository, run

```git clone https://github.com/carnegie/clab_experiencedT.git```

and then

```cd clab_experiencedT```.

## Data
Most data is from the World Bank and can be downloaded from data.worldbank.org.

The data for the CDD predictions is from https://github.com/marina-andrijevic/coolinggap.

## Setting up the environment
The environment can be set up using conda. To create the environment, run

```conda env create -f env.yaml```.

Then activate the environment by running

```conda activate experiencedT_env```.

## Running main script
The main script for creating all plots is the interactive jupyter notebook ```experienced_cdd_predictions.ipynb```.

The underlying exposure function is derived in ```derive_exposure_function.ipynb``` (this is not required to run the main script but shows the derivation of the assumptions used in the main script).