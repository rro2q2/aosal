# Open Set Active Learning Project
Official code of the paper title "Adaptive Open-Set Active Learning with Distance-Based Out-of-Distribution Detection for Robust Task-Oriented Dialog System'.

## Getting Starts:
Will detail resources later!

## Configurations
The file `main.yaml` contains full configurations for running expreiments

Config args (Important ones):
* IND dataset: `ind_dataset=<str>` Choices (`clinc`, `rostd`, `imdb`, `sst2`)
* OOD dataset: `ood_dataset=<str>` Choices (`clinc`, `rostd`, `imdb`, `sst2`)
* noise_ratio: `noise_ratio=<float>` Interval (`0.1, 1.0`)
* epochs: `epochs=<int>` Interval (`1, ∞`)
* strategy: `strategy=<str>` Choices (`aosal`, `random`, `entropy`, `bertkm`, `badge`, `cal`)
* percentile `percentile=<int>` Choices (`90`, `95`, `97`)
* inf_measure `inf_measure=<int>` Choices (`uncertainty`, `diversity`)

## How to run
### Run default code
`python src/main.py`

### Running AOSAL
To run AOSAL use the command below:

`python src/main.py ind_dataset=<str>, ood_dataset=<str>, noise_ratio=<float>, epochs=<int> strategy=<str>, percentile=<int> distance=<str> inf_measure=<str>`

For example, if you wanted to use the `AOSAL` AL strategy to train a model on CLINC-FULL over 5 epochs and test on ROSTD with 10% noise ratio, 95 FPR, Mahalanobis distance and uncertainty measure for informative query selection, the following code would look as:

`python src/main.py ind_dataset=clinc, ood_dataset=rostd, noise_ratio=0.1, epochs=5 strategy=aosal, percentile=95 distance=mahalanobis inf_measure=uncertainty`

### Running Baselines
To run AL baselines use the following command below:

`python src/main.py ind_dataset=<str>, ood_dataset=<str>, noise_ratio=<float>, epochs=<int> strategy=<str>`

For example, if you wanted to use the `Entropy` AL strategy to train a model on CLINC-FULL over 5 epochs and test on ROSTD with 10% noise ratio, the following code would look as:

`python src/main.py ind_dataset=clinc, ood_dataset=rostd, noise_ratio=0.1, epochs=5 strategy=entropy`

### Train Configurations per Dataset
In terms of `epochs`, the best configuration are as follows for each IND dataset:
* CLINC-Full: `epochs=5`
* ROSTD: `epochs=1`
* IMDB: `epochs=1`,
* SST2: `epochs=1`

## Citing This Work
If you would like to cite this work, please use the BibTeX syntax shown below:
```
@inproceedings{goruganthu2024adaptive,
  title={Adaptive Open-Set Active Learning with Distance-Based Out-of-Distribution Detection for Robust Task-Oriented Dialog System},
  author={Goruganthu, Sai Keerthana and Oruche, Roland R and Calyam, Prasad},
  booktitle={Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue},
  pages={357--369},
  year={2024}
}
```
