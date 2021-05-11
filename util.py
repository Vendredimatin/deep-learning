import argparse
import math
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('transformers_epoch_40.pt',map_location=torch.device('cpu'))
#print(model)
newModel = nn.Sequential(*list(model.children()))
#print(newModel)
embedingLayer = nn.Sequential(*list(model.children()))[0]
PELayer = nn.Sequential(*list(model.children()))[1]
EncoderLayer = nn.Sequential(*list(model.children()))[3]
DecoderLayer = nn.Sequential(*list(model.children()))[4]

TransformerEncoder = nn.Sequential(*list(EncoderLayer.children())[0])
TransformerEncoder_layer1 = nn.Sequential(*list(TransformerEncoder.children())[:1])
print(TransformerEncoder_layer1)
TransformerEncoder_layer1 = nn.Sequential(*list(TransformerEncoder_layer1.children()))[0]
print(TransformerEncoder_layer1)
TransformerEncoder_layer1_myltiHead = nn.Sequential(*list(TransformerEncoder_layer1.children()))[0]
print(TransformerEncoder_layer1_myltiHead)
TransformerEncoder_layer1_myltiHead_others = nn.Sequential(*list(TransformerEncoder_layer1.children())[1:])
#print(TransformerEncoder_layer1_myltiHead_others)

vocab = data_loader.get_vocab()
print(vocab.itos[10])
def train():
    #model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()#
    log_interval = 200
    model.eval()
    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)):
        data, targets = data_loader.get_batch(data_loader.train_data, i)
        data = data.to(device)
        targets = targets.to(device)

        embeding_out = embedingLayer(data)
        pe_out = PELayer(embeding_out)
        encoder_layer1_multiHead_output, attn_output_weights = TransformerEncoder_layer1_myltiHead(pe_out,pe_out,pe_out) 

        data_0 = data[:,0]
        attn_output_weights_0 = attn_output_weights[0]
        words = []
        for idx in data_0:
            idx = idx.item()
            words.append(vocab.itos[idx])


        plt.imshow(attn_output_weights_0.detach().numpy())

        plt.show()

        encoder_layer1_multiHead_others_output = TransformerEncoder_layer1_myltiHead_others(pe_out)
       

train()