# Anomalous Sound Detection for predictive maintenance of industrial machines

## Table of Contents
1. [Description](#description)
1. [Objectives](#objectives)
	1. [Challenges](#challenges)
	2. [Limitations](#limitations)
	3. [Further developments](#further-developments)
1. [Repo Architecture](#repo-architecture)
1. [Installation](#installation)
1. [Usage](#usage)
1. [Visuals](#visuals)
1. [Timeline](#timeline)
1. [Personal situation](#personal-situation)

## Description
This project is a part of the Becode.org AI Bootcamp programme. The goal is to use unsupervised ML method for anomalous sound detection in industrial machines for a fictional company Acme Corporation. Data samples of normal and abnormal sounds of valves, pumps, fans and sliders are provided. The samples have been labelled. The goal of the task is to a) verify if clustering audio files may work for detecting normal and abnormal sounds b) verify if clustering can detect transition between normal and abnormal sounds. The project is a continuation of [the previous project]((https://github.com/kpranke/machine-monitoring-conditions)) related to the same data, but with the use of a supervised classifier. 

![factory](https://images.unsplash.com/photo-1513828583688-c52646db42da?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80)

## Objectives

- Be able to work and process data from audio format
- Find insights from data, build hypothesis and define conclusions
- Build machine learning models for predictive classification and/or regression
- Select the right performance metrics for your model
- Evaluate the distribution of data points and evaluate its influence in the model
- Be able to identify underfitting or overfitting that might exist on the model
- Tuning parameters of the model for better performance
- Select the model with better performance and following your customer's requirements
- Define the strengths and limitations of the model

### Strengths

- The model works for all the types of machines with acceptable accuracy.
- The model provides intial insights to detect abnormalities.

### Limitations

- Undersampled examples of abnormal sounds
- Overfitting in the current model 

### Further Developments

- Finetuning the model per machine
- Trying different machine learning techniques

## Repo Architecture

- *(1) README.md* Project documentation
- *(2) Download folders with audio data.ipynb* A Jupyter Notebook file with code to download data samples from provided resource
- *(3) Create a dataframe with files.ipynb* A Jupyter Notebook file with code to create a dataframe and .csv file with filenames
- *(4) anomaly_files.csv* .csv file with filenames of provided samples
- *(5) Extract feature 6dB machine.ipynb* A Jupyter Notebook file with code to extract features and targets for 6dB samples of all machines 
- *(6) ASD_model.ipynb* A Jupyter Notebook file with code to make a prediction ML model
- *(7) ASD_model.pkl* A pickelised ML model to easily deploy

## Installation

 *git clone* the repo 


## Usage

To use this model run (2) and (3) to collect the data samples and extract the features by running file (5). Next run (6) or use (7) to run a prediction.

## Timeline

The project took 4 working days to write a code and was presented on the 5th day.

## Personal situation

Contributors: [mokegg](https://github.com/mokegg), [kpranke](https://github.com/kpranke)

We are currently participating in the Becode.org AI Bootcamp to upskill into a career in data science.

**[Back to top](#table-of-contents)**
