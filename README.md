# Anomalous Sound Detection for predictive maintenance of industrial machines - unsupervised learning

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
This project is a part of the Becode.org AI Bootcamp programme. The goal is to use unsupervised ML method for anomalous sound detection in industrial machines for a fictional company Acme Corporation. Data samples of normal and abnormal sounds of valves, pumps, fans and sliders are provided. The samples have been labelled. The goal of the task is to a) verify if clustering audio files may work for detecting normal and abnormal sounds b) verify if clustering can detect transition between normal and abnormal sounds. The project is a continuation of [the previous project](https://github.com/kpranke/machine-monitoring-conditions) which was about classifying the same data, but with the use of a supervised method. 

![factory](https://images.unsplash.com/photo-1513828583688-c52646db42da?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80)

## Objectives


- Find insights from data, build hypothesis and define conclusions
- Build machine learning models for clustering
- Select the right performance metrics for your model
- Define the strengths and limitations of the model
- Verify if it is possible to automatically label the operative conditions
- Verify if it is possible to identify transitory states

### Strengths

- The model provides trustable results for some of the machines.
- The model provides intial insights to cluster sounds into normal/abnormal.
- The model provides initial insights to identify the transitory states.

### Limitations


### Further Developments

- Finetuning the model per machine
- Trying different features with kMeans
- Trying different clustering techniques

## Repo Architecture

- *README.md* a .md file contains the documentation of the project, this is the file you are currently reading
- *Extract feature 6dB machine.ipynb* a jupyter notebook file that contains functions necessary to extract features from data per machine and save them to .csv files (the files are also provided in the repo)
-  *ML model - one machine id.ipynb* a jupyter notebook file with analysis of both labelled targets as well as kmeans clustering results for one machine: slider id_00 (used with 2 and 3 clusters for insights on correlation between labels and clusters as well as identifying a possibility of detecting a transitional cluster)
-  *Validation for all machines.ipynb* a jupyter notebook file containing code necessary to run the kmeans algorithm for all macines and machine types (16 in total) and calculating corellation between labels and clusters
- *anomaly_files.csv* a csv file necessary to run *Extract feature 6dB machine.ipynb* notebook
- *Download folders with audio data.ipynb* a jupyter notebook file necessary to download audio files
- *df_6dB_fan_fe_ta_imb.csv*, *df_6dB_pump_fe_ta_imb.csv*, *df_6dB_slider_fe_ta_imb.csv*, *df_6dB_valve_fe_ta_imb.csv* .csv files with features of respective machines: fan, pump, slider, valve
- */images* a folder containing visuals included in README.txt
- *kmeans_model_2.sav* a pickle file containing kmeans model with nr of clusters = 2
- *kmeans_model_3.sav* a pickle file containing kmeans model with nr of clusters = 3
## Installation

 *git clone* the repo 


## Usage

## Findings

During [the first part of the project](https://github.com/kpranke/machine-monitoring-conditions), in collaboration with my colleague, we used the sklearn classifier to be able to detect the normal and abnormal sound labels. AI model was trained with 70% of the data, 15% of data used for testing, and 15% for validation. We achieved 0.89 F1-score and detected overfitting. One of the reasons for overfitting was undersampled data of abnormal sounds. We concluded the reliability of the model should be imprved, among others, by addressing the undersampling and overfitting.

During the current part of the challenge entailing using unsupervised learning, I worked on my own. I tested using kmeans unsupervised ML algorithm with the number of clusters 2 and 3. I also compared the available labels (normal\abnormal sounds) with the results of running kmeans with 2 clusters. Below are my findings:

*Kmeans nr_clusters = 2* 
In order to evaluate the esults of running kmeans algorithm with 2 clusters, a correlation between provided labels (normal/abnormal) and clusters have been calculated per each machine type. The table below presents the correlation per machine type:
![mach_id_corr](https://github.com/kpranke/machine-monitoring-conditions-unsupervised/blob/main/images/mach_id_corr.png)


## Timeline

The project took 4 working days.

## Personal situation

Contributors: [kpranke](https://github.com/kpranke)

I am currently participating in the Becode.org AI Bootcamp to upskill into a career in data science.

**[Back to top](#table-of-contents)**
