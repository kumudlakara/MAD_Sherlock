# MAD-Sherlock: MultiAgent Debates for Misinformation Detection based on Out of Context Image Use
This repository contains the code for MAD-Sherlock. It contains all the notebooks as well as final scripts used to replicate the results. This project has been developed as part of my master's thesis for the MSc in Advanced Computer Science course at the University of Oxford. 

The main goal of this project is to develop an autonomous AI agent capable of detecting as well as explaining instances of misinformation on the internet. It uses an external retrieval module to form an understanding of the original context in which the image appears on the Internet. This helps the system better explain its predictions.

## Installation
The code can be setup by cloning this repository and setting up a virtual environment as follows:

```bash
cd MAD_Sherlock
pip install -r requirements.txt
```

All experiments using the LLaVA model will also require the LLaVA to be installed. This can be done by following the instructions available on the official LLaVA repository [here](https://github.com/haotian-liu/LLaVA)
## Code Structure
The code in this repository is structured to include baseline results and analysis, various experiments including result analysis and model comparison, scripts to run experiments and various utility files containing modules for result compilation, external retrieval and prompting. The contents of different directories are detailed below:

1. ```baselines```: contains the code for setting up different baseline methods and experimenting with them in a notebook environment. We also provide scripts to run the experiments presented in the thesis. The baselines directory further contains the `baseline_result_analysis.ipynb` notebook which includes the result analysis for different baseline methods as presented in the thesis.

2. `experiments`: contains various notebooks in a sequential order in which the experiments were carried out through out the course of this project. A notebook of particular importance in this directory is the `exp10_result_analysis.ipynb` which contains the results and their analysis as presented in the thesis. The directory also includes different debate setups defined in the thesis. These notebooks are a great way to experiment with the code base and try out different image-text pairs.

3. `scripts`: contains all the scripts that were used to run the experiments on the entire inference dataset for the project.

4. `utils`: contains the code for utility components of the system. This directory has modules that facilitate external retrieval, data access, prompting and stored retrieval.

## Dataset
For this project we report results on a subset of the NewsCLIPpings dataset available [here](https://github.com/g-luo/news_clippings?tab=readme-ov-file): 
The NewsCLIPpings dataset is built based on the VisualNews dataset available [here](https://github.com/FuxiaoLiu/VisualNews-Repository)
The Visual- News dataset consists of image-caption pairs from four news agencies: BBC, USA Today, The Guardian and The Washington Post. The NewsCLIPpings dataset is created by generating out-of-context samples by replacing an image in one image- caption pair with a semantically related image from a different image-caption pair. Access to both datasets is required in order to replicate the experiments in this project.  
