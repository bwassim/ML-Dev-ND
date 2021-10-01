#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(project="exercise_5", job_type="process_data")

    ## YOUR CODE HERE
    logger.info("Download artifact")
    artifact = run.use_artifact(args.input_artifact)
    local_path = artifact.file()

    logger.info("Reading dataframe")
    df = pd.read_parquet(local_path)

    logger.info("Starting preprocessing: Drop duplicates")
    df.drop_duplicates().reset_index(drop=True)

    # - Create a new feature by concatenating them, after replacing all missing values with the empty string.
    logger.info("Create a new feature")
    df["title"].fillna(value="", inplace=True)
    df["song_name"].fillna(value="", inplace=True)
    df["text_feature"] = df["title"] + " " + df["song_name"]

    filename = args.artifact_name
    df.to_csv(filename)

    # save the preprocessed data into an artifact
    logger.info("Save the preprocessed data into an artifact")
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(filename)

    logging.info("Logging artifact")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
