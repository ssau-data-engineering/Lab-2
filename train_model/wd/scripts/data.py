import os
import pandas as pd

# Preprocess text (username and link placeholders)
def preprocess(text: str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["text"] = df["text"].apply(preprocess)
    return df[["target", "text"]]

def split_df(df: pd.DataFrame) -> tuple[list, list]:
    return df["text"].tolist(), df["target"].tolist()


if __name__ == "__main__":
    DATASET_COLUMNS=['target','ids','date','flag','user','text']
    DATASET_ENCODING = "ISO-8859-1"

    data_path = "/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/train_data"

    data_files = os.listdir(data_path)


    df = pd.read_csv(data_path, encoding=DATASET_ENCODING,names=DATASET_COLUMNS)



    split_size = 100_000
    split_data_path = "/home/anteii/projects/ssau-data-engineering/Prerequisites/airflow/data/lab2/train_data"
    for i in range(5):
        split_path = os.path.join(split_data_path, f"chunk_{i}.csv")
        df.iloc[i * split_size: (i + 1) * split_size].to_csv(split_path)


    #print(len(df))
    #print(df.columns)
    #print(df.head(3))