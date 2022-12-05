## Modelling

Multilabel classification in task planet-understanding-the-amazon-from-space.

### Dataset

![](./assets/examples.png)

Download the dataset from [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)

### Подготовка пайплайна

1. Create and activate environment
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Install packages

    from activated environment:
    ```
    pip install -r requirements.txt
    ```

3.  Split dataset on train/val/test samples:
    ```
    python train_test_split.py -i path/to/dataframe -o path/to/save/splited/dataframes -drop-cols tags -col_id image_name
    ```

4. ClearML setting
    - [in your ClearML profile](https://app.community.clear.ml/profile) click "Create new credentials"
    - write down `clearml-init` and continue by instruction steps

### Training with Catalyst
Start with `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification nohup python train.py > log.out
```

Start without `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train.py
```

### Training with pytorch-lightning
Start with `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification nohup python train_pl.py > log.out
```

Start without `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ROOT_PATH=/data/movie-genre-classification python train_pl.py
```

