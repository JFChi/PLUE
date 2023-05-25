<div align="center">

<h1>Privacy Policies Language Understanding Evaluation</h1>

This repository contains code for downloading data and implementations of baseline systems for PLUE.

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#Usage">Usage</a> •
  <a href="#license">License</a> • 
  <a href="#citation">Citation</a>
</p>

</div>

## Setup

### Downloading PLUE Datasets 

We have already provided all PLUE datasets and the correponding preprocessing scripts in the [data folder](https://github.com/JFChi/PLUE/tree/main/data). Except PolicyIE, we have also uploaded the all pre-processed datasets. 

To pre-process policyIE, please run
```
cd data/policyie
bash run.sh 
```

If you want to checkout how we preprocess PrivacyQA, PIExtract, APP-350, and OPP-115, please run
```
cd data
bash setup.sh
```

### Downloading Pre-training Corpus

To download our pre-training corpus, please run
```
cd pretraining/data
bash download.sh 
```

## Usage

### Pre-training

In each pre-trained model folder, please download all the required dependencies
```
pip install -r requirements.txt
```

Note that the dependencies are associated with each pre-trained models. After all dependencies are properly installed, please run

```
bash train.sh
```

### Fine-tuning

All fine-tuning tasks share the same environment
```
cd finetuning
pip install -r requirements.txt
```

for each task, please run the run.sh in the corresponding folder. For example, if we want to run APP-350 with pp-roberta, we run
```
cd finetuning/classification/app350/
bash run.sh 0 policy_roberta # 0 indicate the gpu_id
```

## License

Contents of this repository is under the [MIT license](https://opensource.org/licenses/MIT). The license applies to the
released model checkpoints as well.

## Citation

```
@inproceedings{chi2023plue,
  author = {Chi, Jianfeng and Ahmad, Wasi Uddin and Tian, Yuan and Chang, Kai-Wei},
  title = {PLUE: Language Understanding Evaluation Benchmark for Privacy Policies in English},
  booktitle = {ACL (short)},
  year = {2023}
}
```

