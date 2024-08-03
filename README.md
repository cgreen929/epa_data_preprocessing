# EPA PM2.5 Data Preprocessing
## Introduction
This guide will walk you through the steps required to use the scripts in this repository to create a complete hourly set of PM2.5 data for your desired location and time span, using data downloaded from the [EPA Air Quality](https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw) repository. Initially, the scripts use linear interpolation to fill data gaps less than 4 hours in duration. Then, for gaps greater than 4 hours, data is sourced from the nearest available weather stations to complete your dataset. Leap year days are removed to create an 8760 hour dataset for each year.

Currently, only PM2.5 data is supported, but future versions will allow for handling of other air quality parameters. It is also slow, and future updates will aim to increase the speed.
## 1. Requirements
You will need the following packages:
- [pandas](https://pypi.org/project/pandas/)
- [numpy](https://pypi.org/project/numpy/)
- [geopy](https://pypi.org/project/geopy/) - Calculating distance
## 2. Usage Instructions
#### 2.1 Retrieve data
To begin, download the annual hourly PM2.5 data you are interested in [here](https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw), along with the data for the previous year (e.g., if you want to complete a dataset from 2018, also include the data from 2017 in your directory). Place these files in the "hourly_data" folder.
#### 2.2 Create target sites CSV
Create a CSV file following the template in target_sites.csv, using the coordinates for your city/cities of interest. The example file contains information for a weather station in San Francisco, CA; you may replace this information and/or add subsequent rows for additional cities. This list will be used to identify the primary station 
#### 2.3 Fill the data
Open weather_data_fill.py and specify the years you are interested in, in ascending order. Example:
```sh
years = [2018,2019]
```
Several of the datasets are sparse, so, to speed up computation time and reduce the variability of source location, weather stations with dataset completeness less than a specified threshold are removed from the candidate dataset. The default value is 0.90, but you can set this to any value you like by changing: 
```sh
percent_completeness_threshold = 0.90
```
In PM2.5 data, negative values are not expected. The script removes any datasets with a specified percentage of PM2.5 measurements below -2 (allowing for some uncertainty around low measurement values). Specify your acceptable percentage of measurement values <-2 by specifying _percent_negative_threshold_ (default is 0.98).
```sh
percent_negative_threshold = 0.98
```
Run the file. The folder "annual_data_complete" will contain the completed results files for each city/year combination. 


## License

MIT
