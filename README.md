# Repository for experienced temperature project

## Data
The script is using the ERA5 temperature data downloaded by Lei to the Caltech server, so it's currently set up to run there. All other data is in this repository.

## Setting up the environment
The environment can be set up using conda. To create the environment, run

```conda env create -f env.yml```.

## Running main script
The main script is run by doing

```python historicalT.py```

### Other scripts
The script ```AC_penetration.ipynb``` holds an interactive jupyter notebook to illustrate the estimated AC penetration as a function of cooling degree days and GDP per capita.

The script ```experiencedT.ipynb``` holds some first calculations for this project and is currently abandoned.
