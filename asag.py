from utils.configuration import train_config
from utils.data import get_train_dataloader, get_test_dataloader
from utils.training import train_epoch, validate
from fire import Fire
import transformers
import torch
import gc
import os
import wandb

def train():
    cuda = torch.cuda.is_available()
    config = train_config()
    wandb.init(project = 'test',
               group = config.model_path,
               config = config)
    config = wandb.config
    model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_path, num_labels = config.num_labels)
    train_dataloader = get_train_dataloader(
        model_path = config.model_path,
        num_workers = config.num_workers if cuda else 0,
        percent = config.train_percent,
        batch_size = config.batch_size,
        drop_last = False)
    test_dataloader = get_test_dataloader(
        model_path = config.model_path,
        num_workers = config.num_workers if cuda else 0,
        percent = config.test_percent,
        batch_size = config.batch_size,
        drop_last = False)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.max_lr, weight_decay = config.weight_decay)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 1024, 17000)
    if cuda:
        model.cuda()
    best_f1 = 0.0
    patience = 0
    epoch = 0
    try:
        while epoch < config.max_epochs:
            epoch += 1
            av_epoch_loss = train_epoch(train_dataloader, model, optimizer, lr_scheduler, config.num_labels, cuda)
            gc.collect()
            torch.cuda.empty_cache()
            metrics_weighted, metrics_macro = validate(model, test_dataloader, cuda)
            p,r,f1,val_acc = metrics_weighted
            p_m, r_m, f1_m, val_acc_m = metrics_macro
            print(f'epoch: {epoch} | av_epoch_loss {av_epoch_loss:.5f} | f1: {f1:.5f} | accuracy: {val_acc:.5f}')
            wandb.log({'f1': f1, 'f1-macro': f1_m, 'accuracy': val_acc, 'accuracy-macro': val_acc_m})
            if f1 > best_f1:
                this_model =  os.path.join(wandb.run.dir,'best_f1.pt')
                print("saving to: ", this_model)
                torch.save([model.state_dict(), config.__dict__], this_model)
                wandb.save('*.pt')
                best_f1 = f1
                patience = 0 #max((0, patience-1))
            elif config.max_patience:
                patience +=1
                if patience >= config.max_patience:
                    break
        # Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.max_lr)
        gc.collect()
        torch.cuda.empty_cache()
        #return model   #Gives Error

    except KeyboardInterrupt:
        if config.log:
            wandb.save('*.pt')
        model.cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.max_lr)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    Fire(train)
