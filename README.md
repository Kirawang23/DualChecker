# DualChecker

This repository contains the codebase for the paper titled **"[Interactive DualChecker for Mitigating Hallucinations in Distilling Large Language Models](https://arxiv.org/abs/2408.12326)."**

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Prepare the Data](#2-prepare-the-data)
    - [3. Run the Framework](#3-run-the-framework)
    - [4. Fine-tuning the Student Model](#4-fine-tuning-the-student-model)
- [Citation](#citation)

## Overview

This repository implements the DualChecker framework designed to mitigate hallucinations and enhance the performance of both teacher and student models in knowledge distillation. 
## Directory Structure

The repository is organized as follows:

```
DualChecker/
│
├── data/
│   ├── cls_train.json      # Green innovation identification task (included as an example in the repo due to size constraints, but will be uploaded through Google Drive later)
│   ├── ce_train.json       # Technological causality extraction task (included as an example in the repo due to size constraints, but will be uploaded through Google Drive later)
│   └── path_train.json     # Environmental Path Identification task (included as an example in the repo due to size constraints, but will be uploaded through Google Drive later)
│
├── model/                  # The backbone models (not included in this repo due to size constraints, but will be uploaded through Google Drive later)
│
├── script/
│   ├── dataloader.py        # Script for loading data
│   ├── get_student.py       # Script to initialize the student model
│   ├── get_teacher.py       # Script to initialize the teacher model
│   ├── prompt_template.py   # Template for generating prompts
│   ├── run.py               # Main script to run the DualChecker framework
│   ├── student_finetune.py  # Script to fine-tune the student model
│
├── main.py                  # Entry point for running the project
├── main.sh                  # Shell script for executing the main pipeline
├── requirements.txt         # List of dependencies
└── readme.md                # This file

```

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Transformers
- Other dependencies as listed in `requirements.txt`

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Kirawang23/DualChecker.git
cd DualChecker
```

### 2. Prepare the Data

Ensure your training data is placed in the `data/` directory and formatted according to the examples provided (`ce_train.json`, `cls_train.json`, and `path_train.json`).

Since we use the original patents in Japanese, we plan to provide a translated version in English for a broader audience.

### 3. Run the Framework

You can execute the entire pipeline using the `main.sh` script:

```bash
bash main.sh
```

Alternatively, you can run the Python scripts individually by following the instructions in `main.py`.

### 4. Fine-tuning the Student Model

To fine-tune the student model, use the following command:

```bash
python script/student_finetune.py
```
## Citation

Please cite our paper if you use this code or our work in your research.

To cite this work in BibTeX format, you can use the following reference:

```bibtex
@misc{wang2024interactivedualcheckermitigatinghallucinations,
      title={Interactive DualChecker for Mitigating Hallucinations in Distilling Large Language Models}, 
      author={Meiyun Wang and Masahiro Suzuki and Hiroki Sakaji and Kiyoshi Izumi},
      year={2024},
      eprint={2408.12326},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.12326}, 
}
