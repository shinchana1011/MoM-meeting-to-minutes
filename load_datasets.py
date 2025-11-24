from datasets import load_dataset

ds = load_dataset(
    "csv",
    data_files={
        "train": r"data\raw\samsum\train.csv",
        "validation": r"data\raw\samsum\validation.csv",
        "test": r"data\raw\samsum\test.csv",
    }
)
print(ds)
