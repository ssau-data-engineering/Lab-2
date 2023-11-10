import random

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers.optimization import get_linear_schedule_with_warmup

RANDOM_SEED = 42

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s', 
                    datefmt='%a, %d %b %Y %H:%M:%S', 
                    filename='results/log.log', filemode='w')

# Класс датасета для модели torch
class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, y, tokenizer, max_length=512, **kwargs):
        super().__init__(**kwargs)
        self.labels = y
        self.texts = [tokenizer(text, padding='max_length', max_length=max_length, 
                                truncation=True, return_tensors="pt") for text in X]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return self.labels[idx]

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

# Функция создания toch dataloader с заданными сидами (для детерминизма)
def create_torch_dataloader(torch_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=False):
  # Функция инициализации воркера (для воспроизводимости)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    #sampler = RandomSampler(train_dataset, generator=g)
    dataloader = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=shuffle,
                          worker_init_fn=seed_worker, generator=g, 
                          num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

def train(model, tokenizer, X_train, X_valid, y_train, y_valid, 
          optimizer_name, batch_size=32, epochs=5,
          learning_rate=5e-5, backbone_lr=5e-5, head_lr=5e-3,
          save_best=False, save_path="./saved_model", trial=None):
    
    train_ds = Dataset(X_train, y_train, tokenizer)
    val_ds = Dataset(X_valid, y_valid, tokenizer)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Use cuda: {use_cuda}")
    # num_workers don't go well along with tokenizer from transformers
    train_dataloader = create_torch_dataloader(train_ds, batch_size=batch_size,
                                               shuffle=True, num_workers=0, pin_memory=use_cuda)
    val_dataloader = create_torch_dataloader(val_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=0, pin_memory=use_cuda)
    
    plist = [
        {'params': model.base_model.parameters(), 'lr': backbone_lr},
        {'params': model.classifier.parameters(), 'lr': head_lr}
        ]
    
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = Adam(plist, lr=learning_rate)
    optimizer = getattr(
        torch.optim, optimizer_name
    )(plist, lr=learning_rate)
    
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    best_f1_val_score = 0
    
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):
            
            predicts = []
            labels = []

            total_acc_train = 0
            total_loss_train = 0
            
            model.train()
            
            for train_input, train_label in tqdm(train_dataloader):
                # Clear gradients before backward pass
                #model.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                #for param in model.parameters():
                #    param.grad = None
                # Move to device
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                # Forward pass
                #with autocast(device_type="cuda" if use_cuda else "cpu"):
                logits = model(input_id, mask).logits
                # Calculate loss
                
                batch_loss = criterion(logits, train_label)
                total_loss_train += batch_loss.item()
                # Cache values for metrics
                predicts.extend([x.item() for x in logits.argmax(dim=1)])
                labels.extend([x.item() for x in train_label])
                acc = (logits.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                
                # Backward pass
                batch_loss.backward()
                #scaler.scale(batch_loss).backward()
                # Clip gradient
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                # Update learning rate
                scheduler.step()
                                       
            total_acc_val = 0
            total_loss_val = 0
            
            predicts = []
            labels = []
            
            model.eval()
            
            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    logits = model(input_id, mask).logits

                    batch_loss = criterion(logits, val_label)
                    total_loss_val += batch_loss.item()
                    
                    predicts.extend([x.item() for x in logits.argmax(dim=1)])
                    labels.extend([x.item() for x in val_label])

                    acc = (logits.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
     
            history["loss"].append(total_loss_train / len(y_train))
            history["val_loss"].append(total_loss_val / len(y_valid))
            history["acc"].append(total_acc_train / len(y_train))
            history["val_acc"].append(total_acc_val / len(y_valid))
            
            logger.info("Epochs: {} | Train Loss: {:.3f} | Train Accuracy: {:.3f} | Val Loss: {:.3f} | Val Accuracy: {:.3f}".
                  format(epoch_num + 1, total_loss_train / len(y_train), total_acc_train / len(y_train), total_loss_val / len(y_valid), 
                        total_acc_val / len(y_valid)))

    return history



