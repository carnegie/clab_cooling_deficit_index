# Repository for experienced temperature project

## Cloning the repository
To clone the repository, run

```git clone https://github.com/carnegie/clab_experiencedT.git```

and then

```cd clab_experiencedT```.

## Setting up the environment
The environment can be set up using conda. To create the environment, run

```conda env create -f env.yaml```.

Then activate the environment by running

```conda activate experiencedT_env```.


## Running main script
The main script for creating all plots is the interactive jupyter notebook ```experienced_cdd_predictions.ipynb```.

The underlying function describing air conditioning adoption depending on cooling degree days and per capita GDP is derived in ```derive_ac_adoption_function.ipynb``` (it is not necessary to this to be able to run the main script but fit parameters used in the main script are obtained with this script).