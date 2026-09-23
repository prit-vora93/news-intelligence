import json
from pathlib import Path

import pandas as pd


# DATA_PATH = Path(
#     "NewsMTSC/NewsSentiment/experiments/default/"
#     "datasets/newsmtsc-rw/train.jsonl"
# )


# rows = []


# with DATA_PATH.open("r", encoding="utf-8") as file:
#     for line in file:
#         record = json.loads(line)

#         sentence = record["sentence_normalized"]

#         for target in record["targets"]:
#             mention = target["mention"]
#             polarity = target["polarity"]

#             rows.append(
#                 {
#                     "sentence": sentence,
#                     "target": mention,
#                     "polarity": polarity,
#                 }
#             )
# df = pd.DataFrame(rows)

def prepare_split(data_path):
    rows = []

    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            sentence = record["sentence_normalized"]

            for target in record["targets"]:
                # rows.append(
                #     {
                #         "sentence": sentence,
                #         "target": target["mention"],
                #         "polarity": target["polarity"],
                #     }
                # )

                rows.append(
                    {
                        "sentence": sentence,
                        "target": target["mention"],
                        "target_from": target["from"],
                        "target_to": target["to"],
                        "polarity": target["polarity"],
                    }
                )

    return pd.DataFrame(rows)
DATA_DIR = Path(
    "NewsMTSC/NewsSentiment/experiments/default/"
    "datasets/newsmtsc-rw"
)

TRAIN_PATH = DATA_DIR / "train.jsonl"
DEV_PATH = DATA_DIR / "dev.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

train_df = prepare_split(TRAIN_PATH)
dev_df = prepare_split(DEV_PATH)
test_df = prepare_split(TEST_PATH)

train_clean = train_df.drop_duplicates().reset_index(drop=True)


print("Dataset Shapes")
print("--------------")
print(f"Train:      {train_df.shape}")
print(f"Validation: {dev_df.shape}")
print(f"Test:       {test_df.shape}")
# print(df.head())
# print(f"\nShape: {df.shape}")

print("\nDuplicate Rows")
print("--------------")
print(f"Train:      {train_df.duplicated().sum()}")
print(f"Validation: {dev_df.duplicated().sum()}")
print(f"Test:       {test_df.duplicated().sum()}")


train_clean = train_df.drop_duplicates().reset_index(drop=True)


# #misssing values
# print("\nMissing Values")
# print("--------------")
# print(df.isnull().sum())

# #duplicate values

# print("\nDuplicate Rows")
# print("--------------")
# print(df.duplicated().sum())

# print("\nDuplicate Examples")
# print("------------------")

# duplicates = df[df.duplicated(keep=False)]

# # print(duplicates.head(10).to_string(index=False))

# print("\nUnique Sentences Involved In Duplicates")
# print("---------------------------------------")

# print(duplicates["sentence"].nunique())

# print("\nUnique Targets Involved In Duplicates")
# print("-------------------------------------")

# print(duplicates["target"].nunique())


# #####################conflicting polarity duplicates

# label_conflicts = (
#     df.groupby(["sentence", "target"])["polarity"]
#     .nunique()
# )

# conflicts = label_conflicts[label_conflicts > 1]

# print("\nLabel Conflicts")
# print("---------------")
# print(f"Number of conflicting sentence-target pairs: {len(conflicts)}")

# if len(conflicts) > 0:
#     print("\nConflicting Examples:")

#     for (sentence, target), _ in conflicts.items():
#         labels = df[
#             (df["sentence"] == sentence)
#             & (df["target"] == target)
#         ]["polarity"].unique()

#         print(f"\nSentence: {sentence}")
#         print(f"Target:   {target}")
#         print(f"Labels:   {labels}")

# #clean the duplicate data

# df_clean = df.drop_duplicates().reset_index(drop=True)

# print("\nAfter Removing Exact Duplicates")
# print("-------------------------------")
# print(f"Rows before: {len(df)}")
# print(f"Rows after:  {len(df_clean)}")
# print(f"Removed:     {len(df) - len(df_clean)}")

# print("\nPolarity Distribution After Cleaning")
# print("-------------------------------------")
# print(df_clean["polarity"].value_counts().sort_index())

# #save the dataframe

# output_path = Path("data/processed/train_clean.csv")

# df_clean.to_csv(output_path, index=False)

# print(f"\nSaved cleaned dataset to: {output_path}")

# train_clean.to_csv(
#     "data/processed/train_clean.csv",
#     index=False,
# )

# dev_df.to_csv(
#     "data/processed/validation.csv",
#     index=False,
# )

# test_df.to_csv(
#     "data/processed/test.csv",
#     index=False,
# )

train_clean.to_csv(
    "data/processed/train_target.csv",
    index=False,
)

dev_df.to_csv(
    "data/processed/validation_target.csv",
    index=False,
)

test_df.to_csv(
    "data/processed/test_target.csv",
    index=False,
)
# print("\nSaved Files")
# print("-----------")
# print("data/processed/train_clean.csv")
# print("data/processed/validation.csv")
# print("data/processed/test.csv")

