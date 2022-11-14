# 유방암의 임파선 전이 예측 AI경진대회

Dacon : https://dacon.io/competitions/official/236011/overview/description


# Environment

    conda create -n cancer python=3.8
    conda activate cancer

    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

    pip install wandb, albumentations, sklearn, pandas, tqdm


# Directory

    ├── data
    │   ├── train_imgs
    │   │   ├── BC_01_0001.png
    │   │
    │   ├── train_masks
    │   │   ├── BC_01_0015.png
    │   │
    │   ├── test_imgs
    │   │   ├── BC_01_0011.png
    │   │
    │   ├── clinical_info.xlsx
    │   ├── sample_submission.csv
    │   ├── train.csv
    │   ├── train_heuristic.csv
    │   ├── train_heuristic_5fold.csv
    │   ├── test.csv
    │   └── test_heuristic.csv
    │
    ├── submit
    │
    ├── work_dirs
    │   ├── exp0
    │       ├── config.yaml
    │       ├── best.pt