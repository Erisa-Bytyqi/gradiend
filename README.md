# GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models
> Jonathan Drechsel, Steffen Herbold
[![arXiv](https://img.shields.io/badge/arXiv-2502.01406-blue.svg)](https://arxiv.org/abs/2502.01406)

The official source code for the training and evaluation of GRADIEND can be found [here](https://github.com/aieng-lab/gradiend).


## Grammatical Gender in German

This repository extends the original GRADIEND framework to address grammatical gender in German.

---

## Usage
To train and evaluate a GRADIEND model for grammatical gender in German:

```bash
python train.py mode=gradiend pairing=MFN num_runs=2
```


**Arguments:**
- `--mode`: Specifies the training method to use.
- `--pairing`: Selects the grammatical gender pairs (options: MF, FN, MN, MFN).
- `--mode.model_config.num_runs`: Sets how many models to train.


This ensures you have the required version for grammatical gender support.
You can override any configuration in the `/conf` directory. The example command above demonstrates a minimalist setup to train two GRADIEND models for all three german grammatical genders using the original GRADIEND method.

## Installing DERBI

To use this repository, install DERBI using the command below. 

Upstream (reference only): [DERBI](https://github.com/maxschmaltz/DERBI)

Install from fork (required):

```bash
pip install git+https://github.com/Erisa-Bytyqi/DERBI.git@pip-install
```

