<div align="center">

<h1>PLUE: Language Understanding Evaluation Benchmark for Privacy Policies in English</h1>

This repository contains code for downloading data and implementations of baseline systems for PLUE.

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#build-a-baseline-system">Build a baseline system</a> •
  <a href="#license">License</a> • 
  <a href="#citation">Citation</a>
</p>

</div>

## Setup

Setting up a conda environment is recommended to run experiments. We assume [anaconda](https://www.anaconda.com/) is
installed. The additional requirements noted in requirements.txt can be installed by running
the following script:

```
bash install_env.sh
```

The next step is to download the data. Run the following command to download the datasets.

```
bash data/download.sh
```

## Build a baseline system

### OPP-115, APP-350  (Sentence Classification)

``` 
bash scripts/run.sh [MODEL] [opp-115,app-350]
```

### PolicyQA, PrivacyQA (Question Answering)

``` 
bash scripts/run.sh [MODEL] [policyqa,privacyqa]
```

### PIExtract (Named Entity Recognition)

``` 
bash scripts/run.sh [MODEL] piextract
```

### PolicyIE (Intent Detection and Slot Filling)

``` 
bash scripts/run.sh [MODEL] policyie
```

## License

Contents of this repository is under the [MIT license](https://opensource.org/licenses/MIT). The license applies to the
released model checkpoints as well.

## Citation

```
@article{chi2022plue,
  title={PLUE: Language Understanding Evaluation Benchmark for Privacy Policies in English},
  author={Chi, Jianfeng and Ahmad, Wasi Uddin and Tian, Yuan and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2212.10011},
  year={2022}
}
```

