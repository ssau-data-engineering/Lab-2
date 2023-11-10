
import logging
import os

from data import load_data, split_df
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from train import train
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

log_file_path = "results/log.log"
model_path = "model"
data_path = "data"

EPOCHS = 5
BACKBONE_LEARNING_RATE = 5e-6
HEAD_LEARNING_RATE = 5-5
BATCH_SIZE = 8

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S', 
                    filename=log_file_path, filemode='w')


files = os.listdir(data_path)
logging.info(f"Detected {len(files)} files")

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
logging.info("Tokenizer successfully loaded")

label2id = {"negative": 0, "positive": 1}
id2label = {v: k for k, v in label2id.items()}
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, label2id=label2id, id2label=id2label)

logging.info("Model successfully loaded")

# NOTE remove [:1]
for file in tqdm(os.listdir(data_path)[:1]):
    df_path = os.path.join(data_path, file)

    df = load_data(df_path)

    logging.info(f"DataFrame ({df_path}) successfully read")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                                        stratify=df["target"])

    X_train, y_train = split_df(train_df)
    X_valid, y_valid = split_df(val_df)

    logging.info("Data successfully splitted into train/dev sets")

    #NOTE remove data splits
    history = train(model, tokenizer, 
        X_train[:50], X_valid[:20], y_train[:50], y_valid[:20], 
        "AdamW", batch_size=BATCH_SIZE, epochs=EPOCHS, 
        backbone_lr=BACKBONE_LEARNING_RATE, head_lr=HEAD_LEARNING_RATE)

model.save_pretrained(model_path)