import torch
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#    import wandb


def train_epoch(loader, model, optimizer, lr_scheduler, num_labels, cuda, log = False, token_types = False):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    with tqdm(total=len(loader.batch_sampler)) as pbar:
        epoch_loss = 0.
        for i, batch in enumerate(loader):
            if cuda:
                batch.cuda()
            optimizer.zero_grad()
            logits = model(input_ids = batch.input_ids, attention_mask = batch.generate_mask()).logits
            loss = loss_fn(logits.view(-1, num_labels), batch.labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            lr_scheduler.step()
            if batch.labels.size(0)>1:
                acc = accuracy_score(batch.labels.cpu(), logits.cpu().detach().argmax(-1).squeeze())
            else:
                acc = 0.
            epoch_loss += loss.item()

            wandb.log({"Train Accuracy": acc, "Train Loss": loss.item(), "Learning Rate": optimizer.param_groups[0]['lr']})
            pbar.set_description(f'global_step: {lr_scheduler.last_epoch}| loss: {loss.item():.4f}| acc: {acc*100:.1f}%| epoch_av_loss: {epoch_loss/(i+1):.4f} |')
            pbar.update(1)
        #  move stuff off GPU
        batch.cpu()
        logits = logits.cpu().detach().argmax(-1).squeeze()
        return epoch_loss/(i+1)


def metrics(predictions, y_true, metric_params):
    precision = precision_score(y_true, predictions, **metric_params)
    recall = recall_score(y_true, predictions, **metric_params)
    f1 = f1_score(y_true, predictions, **metric_params)
    accuracy = accuracy_score(y_true, predictions)
    return precision, recall, f1, accuracy

@torch.no_grad()
def validate(model, loader, cuda,  token_types = False):
    model.eval()
    # batches = list(loader)
    preds = []
    true_labels = []
    with tqdm(total= len(loader.batch_sampler)) as pbar:
        for i,batch in enumerate(loader):
            if cuda:
                batch.cuda()
            logits = model(input_ids =  batch.input_ids, attention_mask = batch.generate_mask()).logits
            preds.append(logits.argmax(-1).squeeze().cpu())
            true_labels.append(batch.labels.cpu())
            pbar.update(1)
    preds = torch.cat(preds)
    y_true = torch.cat(true_labels)
    model.train()
    metric_params_weighted = {'average':'weighted', 'labels':list(range(model.config.num_labels))}
    metric_params_macro =  {'average':'macro', 'labels':list(range(model.config.num_labels))}
    return metrics(preds, y_true, metric_params_weighted), metrics(preds, y_true, metric_params_macro)
