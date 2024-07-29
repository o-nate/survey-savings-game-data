# Data processing for surveys experiment in the Savings Game

## Virtual environment
To run this script you must have certain libraries installed through a virtual environment named `survey_sg_data`.

To do install the virtual environment and its libraries, run `conda env create --name recoveredenv --file environment.yml`.

Then, run `conda activate survey_sg_data` to activate the environment.

## Install `pip` package
To ensure `src` resources can be accessed across the project, run:
`pip install -e .`

## Logging
Each module has specific logging settings defined. The function `src.utils.logging_helpers.set_external_module_log_levels` disables other modeules' loggers up to defined `level` parameter. Define the logging level you need for the external modules using this function. Define the logging level you need for your current module using `logger.setLevel`.