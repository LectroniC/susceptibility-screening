# mimic3gen


## Background and Usage

This folder includes the efforts to generate the data from the raw MIMIC3 dataset. This is a challenging task and is still an ongoing process. Here we document our progress:

To avoid starting from scratch, we start with [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks).

To customize the dataset so it contains only the features we want:

Within `mimic3-benchmarks`,

You need to replace `itemid_to_varibale_map.csv` under `mimic3benchmark/resources`. 

You need to replace `channel_info.json` and `discretizer_config.json` under `mimic3models/resources`. 
 
You need to modify the `clean_fns` in `mimic3benchmark/preprocessing.py` accordingly. The `clean_lab` can process pretty much all the features since they are all numerical.