# Repository for "Economic development, air conditioning and adaptation to warming"

## Cloning the repository
To clone the repository, run

```git clone https://github.com/carnegie/clab_cooling_deficit_index.git```

and then

```cd clab_cooling_deficit_index```

## Setting up the environment
The environment can be set up using conda. To create the environment, run

```conda env create -f config/env.yaml```

Then activate the environment by running

```conda activate CDI_env```.


## Running main script
The main script for creating all plots is the interactive jupyter notebook 

```cooling_deficit_index.ipynb```

Open and run this notebook to generate the plots of this publication.

### Deriving the Air Conditioning Adoption Function

The underlying function describing air conditioning adoption depending on cooling degree days and per capita GDP is derived in 

```derive_ac_adoption_function.ipynb```

Although it's not necessary to run this notebook before running the main script, the fit parameters used in the main script are obtained from it.

## License
This project is licensed under the MIT License. See the LICENSE file for details.