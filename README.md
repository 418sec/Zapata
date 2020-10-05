# Zapata
## Computational and Mapping Library   

A revolutionary library for analysis and plotting of meteorological data. The mapping is based on cartopy and GEOCAT libraries from NCAR.
It uses `xarray` as a core data structure for all processing. 

The directory `examples_notebook` contains working Jupyter Notebooks to illustarte different applications of Zapata.

## Documentation

Zapata documentation is created using `sphinx` and it is sourced from the code itself.
HTML documentation can be accessed at `docs/build/html/index.html`

To update the documetation of the tool go to `docs` folder and type `gmake html`.

## Setup working environemt
Python working environment can be setup using the provided conda environment.yml file:

`conda env create -f environment.yml`

`conda activate zapata`

To update your `zapata` environment with following updates use

`conda env update -f environment.yml`

# Library content

## zapata package
Contains computation modules and plotting modules along with dedicated functions to read input data form different datasets (local and remote).
    
- **computation** : Routines for averaging and various computations
    
- **data** : Information on data sets and routine to get data from the data sets

- **mapping** : Mapping routines based on *Cartopy*
    
- **lib** : Utilties for the rest of the modules.


##  klus package
Contains algorithms for data analysis contributed by Stefan Klus
    
- **algorithms** : Routines for averaging and various computations
    
- **kernel** : Information on data sets and routine to get data from the data sets

- **osbervables** : Mapping routines based on *Cartopy*


## interp module
subroutines for data interpolation

## Zeus module
Functions to access and remote working on Zeus cluster
