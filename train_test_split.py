
import argparse
import logging
import os

import pandas as pd
from src.dataset_splitter import stratify_shuffle_split_subsets


def preproc_df(df: pd.DataFrame):
    # Build list with unique labels
    label_list = []
    for tag_str in df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)

    # Add onehot features for every label
    for label in label_list:
        df[label] = df["tags"].apply(lambda x: 1 if label in x.split(' ') else 0)

    return df


def split_and_save_datasets(df: pd.DataFrame, save_path: str):
    logging.info(f"Original dataset: {len(df)}")
    
    df = preproc_df(df)

    df = df.drop_duplicates()
    df = df.drop(["tags"], axis=1)
    logging.info(f"Final dataset: {len(df)}")

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(
        df,
        img_path_column="image_name",
        train_fraction=0.8,
        verbose=True,
    )
    logging.info(f"Train dataset: {len(train_df)}")
    logging.info(f"Valid dataset: {len(valid_df)}")
    logging.info(f"Test dataset: {len(test_df)}")

    train_df.to_csv(os.path.join(save_path, "train_df.csv"), index=False)
    valid_df.to_csv(os.path.join(save_path, "valid_df.csv"), index=False)
    test_df.to_csv(os.path.join(save_path, "test_df.csv"), index=False)
    logging.info("Datasets successfully saved!")


def split_and_save_datasets():

    parser = argparse.ArgumentParser(description='Demo script')
    parser.add_argument('-i', type=str, help='input dataframe path', dest='input_data')
    parser.add_argument('-o', type=str, help='specific output dir', dest='output_dir', default=None)
    parser.add_argument('-col_id', type=str, help='camera name', dest='column_id', default='image_name')
    parser.add_argument('-drop-cols', '--drop-cols', nargs='+', type=str, default=['tags'], help='columns to delete')
    parser.add_argument('-train-share', type=float, help='dataset train share', dest='train_share', default=0.8)
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False, help='verbose dataset split')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    base_path = os.path.join(os.environ.get("ROOT_PATH"))

    df = pd.read_csv(os.path.join(base_path, args.input_data))

    if args.output_dir is not None:
        save_path = os.path.join(base_path, args.output_dir)
    else: 
        save_path = os.path.dirname(os.path.join(base_path, args.input_data))

    logging.info(f"Original dataset: {len(df)}")

    df = preproc_df(df)

    df = df.drop_duplicates()
    print(args.drop_cols)
    df = df.drop(["tags"], axis=1)
    logging.info(f"Final dataset: {len(df)}")

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(
        df,
        img_path_column=args.column_id,
        train_fraction=args.train_share,
        verbose=args.verbose,
    )
    logging.info(f"Train dataset: {len(train_df)}")
    logging.info(f"Valid dataset: {len(valid_df)}")
    logging.info(f"Test dataset: {len(test_df)}")

    train_df.to_csv(os.path.join(save_path, "train_df.csv"), index=False)
    valid_df.to_csv(os.path.join(save_path, "valid_df.csv"), index=False)
    test_df.to_csv(os.path.join(save_path, "test_df.csv"), index=False)
    logging.info("Datasets successfully saved!")


if __name__ == "__main__":
    split_and_save_datasets()
