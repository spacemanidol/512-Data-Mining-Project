import numpy as np  
import wget
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import wandb


wandb.init(project='ORCAS-EXP', entity='spacemanidol')

config = wandb.config
config.learning_rate = 0.01


wandb.log({"loss": loss})




url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
wget.download(url, './cola_public_1.1.zip') 
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, 
names=  ['sentence_source', 'label', 'label_notes', 'sentence'])

print('Number of training sentences: {:,}\n'.format(df.shape[0]))
df.sample(10)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_len = 0
for sent in sentences:

    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    max_len = max(max_len, len(input_ids))
input_ids = []
attention_masks = []
encoded_dict = tokenizer.encode_plus(
  sent,  add_special_tokens = True,  max_length = 64,  pad_to_max_length = True, 
return_attention_mask = 
  True,  return_tensors = 'pt',   
                   )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)



# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# WANDB PARAMETER
def ret_dataloader():
    batch_size = wandb.config.batch_size
    print('batch_size = ', batch_size)
    train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_datase       t),  batch_size = batch_size)

    validation_dataloader = DataLoader( val_dataset,  sampler = SequentialSampler(val_     dataset),   batch_size = batch_size)
    return train_dataloader,validation_dataloaderfrom torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# WANDB PARAMETER
def ret_dataloader():
    batch_size = wandb.config.batch_size
    print('batch_size = ', batch_size)
    train_dataloader = DataLoader(train_dataset,sampler = RandomSampler(train_datase       t),  batch_size = batch_size)

    validation_dataloader = DataLoader( val_dataset,  sampler = SequentialSampler(val_     dataset),   batch_size = batch_size)
    return train_dataloader,validation_dataloader



def ret_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels = 2, 
        output_attentions = False, 
        output_hidden_states = False,
    )
    return model

def ret_optim(model):
    print('Learning_rate = ',wandb.config.learning_rate )
    optimizer = AdamW(model.parameters(),
                      lr = wandb.config.learning_rate, 
                      eps = 1e-8 
                    )
    return optimizer

def ret_scheduler(train_dataloader,optimizer):
    epochs = wandb.config.epochs
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,           
num_training_steps = total_steps)
    return scheduler

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [ 5e-5, 3e-5, 2e-5]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'epochs':{
            'values':[2, 3, 4]
        }
    }
}
sweep_id = wandb.sweep(sweep_config)

import random
import numpy as np

def train():
    wandb.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ret_model()
    model.to(device)
    train_dataloader,validation_dataloader = ret_dataloader()
    optimizer = ret_optim(model)
    scheduler = ret_scheduler(train_dataloader,optimizer)
    training_stats = []
    total_t0 = time.time()
    epochs = wandb.config.epochs
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len                 (train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
        #Log the train loss
            wandb.log({'train_batch_loss':loss.item()})
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)
        #Log the Avg. train loss
        wandb.log({'avg_train_loss':avg_train_loss})
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                      token_type_ids=None, 
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                
        
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)
        #Log the Avg. validation accuracy
        wandb.log({'val_accuracy':avg_val_accuracy,'avg_val_loss':avg_val_loss})
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
def main(args):
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Duplicate query')
    
    args = parser.parse_args()
    main(args)