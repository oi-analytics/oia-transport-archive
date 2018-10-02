# Transport Preprocessing codes (internal only)

Data processing, modelling, analysis and visualisation for a Vietnam Transport Risk Analysis.

[![vtra on github](https://img.shields.io/badge/github-oi--analytics%2Fvietnam--transport-brightgreen.svg)](https://github.com/oi-analytics/vietnam-transport/)

[![Documentation Status](https://readthedocs.org/projects/vietnam-transport-risk-analysis/badge/?version=latest)](https://vietnam-transport-risk-analysis.readthedocs.io/en/latest/?badge=latest)


Documentation is available at
[ReadTheDocs](https://vietnam-transport-risk-analysis.readthedocs.io)


## Requirements

### Python and libraries

Python is required to run the scripts in this project. We suggest using
[miniconda](https://conda.io/miniconda.html) to set up an environment and manage library
dependencies.

Create a conda environment from the `environment.yml` definition:

    conda env create -f environment.yml
    conda install python-igraph

See http://igraph.org/python/ for instructions on Windows installation of `python-igraph`.

Activate the environment:

    conda activate vietnam-transport

Set up the `vtra` package (this project) for development use:

    python setup.py develop


### GAMS

The economic model uses [GAMS](https://www.gams.com/) (General Algebraic Modeling System) via
its python API. GAMS provide [installation and
licensing](https://www.gams.com/latest/docs/UG_MAIN.htm) instructions.


### Postgres and PostGIS

Much of the data processing requires access to a [PostgreSQL](https://www.postgresql.org/)
database with the [PostGIS](http://postgis.net/) extension installed and enabled.


## Configuration

The location of data and output files are configured by a `config.json` file.  Copy
`config.template.json` and edit the file paths and database connection details to locate
the files on your system and access your database.


## Development notes

### Notebooks in git

Make sure not to commit data inadvertently if working with jupyter notebooks. Suggest using
[nbstripout](https://github.com/kynan/nbstripout) to automatically strip output.

Install git hooks to filter notebooks when committing to git:

    cd /path/to/vietnam-transport
    nbstripout --install


## Acknowledgements

This project has been developed by Oxford Infrastructure Analytics as part of a project funded
by the World Bank.

All code is copyright Oxford Infrastructure Analytics, licensed MIT (see the `LICENSE` file for
details) and is available on GitHub at
[oi-analytics/vietnam-transport](https://github.com/oi-analytics/vietnam-transport).
