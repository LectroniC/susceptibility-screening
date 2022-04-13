# UIUC CS598DLH Reproducibility Project

## [Identify Susceptible Locations in Medical Records via Adversarial Attacks on Deep Predictive Models](https://arxiv.org/abs/1802.04822)

## Introduction

This project is to reproduce [Identify Susceptible Locations in Medical Records via Adversarial Attacks on Deep Predictive Models](https://arxiv.org/abs/1802.04822). The code would be added soon.

To train the LSTM model mentioned in the paper, you can use the following sampled data (provided by the author and further sampled by us).
You would need an @illinois.edu account to access the [data](https://drive.google.com/file/d/1BPwtfLnRe4bgKQ439eANFxKDvnkzgDNH/view?usp=sharing).

## Project Layout
- [mimic3gen](../mimic3gen): This folder contains our data cleaning and extraction efforts. 
- main.py: Train the target model (LSTM). The code references the original repo below and is made compatible in TF2. We follow the structure of the original code for clarity.
  
## Reference

- [med-attack](https://github.com/illidanlab/med-attack)