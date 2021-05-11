# coding: utf-8
import argparse
import math
import torch
import torch.nn as nn
import time


from data import Corpus
from Transformer import TransformerModel
from RNN import RNNModel


parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=35, metavar='N',
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# load data
data_loader = Corpus(train_batch_size=args.train_batch_size,
                     eval_batch_size=args.eval_batch_size,
                     bptt=args.bptt)



# WRITE CODE HERE within two '#' bar
########################################
# bulid your language model here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nvoc = data_loader.get_ntokens()
model = RNNModel(nvoc,128,128,2)
#model = TransformerModel(nvoc,ninp=128,nhead=8,nhid=128,nlayers=4)
model = model.to(device)
#print(model)
########################################
isRNN = True

criterion = nn.CrossEntropyLoss()
lr = 1  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train():
    #model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()#
    log_interval = 200
    if isRNN:
        hidden = model.init_hidden(args.train_batch_size)
    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)):
        data, targets = data_loader.get_batch(data_loader.train_data, i)
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        #output就是所有时间序列对应的输出集合，所以直接与target进行比较即可
        ########################################
        ######Your code here########
        ########################################
        if isRNN:
            hidden = repackage_hidden(hidden)
            output,hidden = model(data, hidden)
        #output = model(data)
        loss = criterion(output.view(-1,nvoc), targets.view(-1))
        loss.backward()
        optimizer.step()
        #scheduler.step()

        total_loss += loss.item() * data.size(0)

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data_loader.train_data) // args.bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    if isRNN:
        hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)
            data = data.to(device)
            targets = targets.to(device)
            ########################################
            ######Your code here########
            ########################################
            if isRnn:
                hidden = repackage_hidden(hidden)
                output,hidden = model(hidden)
            else:
                output = model(data)
            output = output.view(-1, nvoc)
            loss = criterion(output, targets.view(-1))

            total_loss += loss.item() * len(data)
    return total_loss/(len(data_source) - 1)




# Train Function
best_val_loss = float("inf")
best_model = None

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, data_loader.val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

########################################

test_loss = evaluate(best_model, data_loader.test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
torch.save(best_model, 'transformers_epoch_40.pt')
