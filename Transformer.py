import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        ########################################
        ######Your code here########
        ########################################
        d_model = ninp
        self.embedding = nn.Embedding(ntoken, ninp)
        self.pos_encoder_src = PositionalEncoding(d_model= ninp,max_len=35)
        #self.pos_encoder_target = PositionalEncoding(d_model= ninp,max_len=35)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=nlayers,norm=encoder_norm)

        # Decoder
        #decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dropout=dropout)
        #decoder_norm = nn.LayerNorm(d_model)
        #self.decoder = nn.TransformerDecoder(decoder_layer,num_layers=nlayers,norm=decoder_norm)

        self.fc = nn.Linear(d_model,ntoken)

        self.d_model = d_model
        self.nhead = nhead

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        ########################################
        ######Your code here########
        ########################################
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        ########################################
        ######Your code here########
        ########################################
        device = src.device
        self.src_mask = self.generate_square_subsequent_mask(len(src)).to(device)
        
        src = self.embedding(src)
        src = self.pos_encoder_src(src)
        memory = self.encoder(src, mask=self.src_mask)

        # 代表 start of sentence
        #target = "<s>"
        #target = self.embedding(target)
        #target = self.pos_encoder_target(target)

        #output = self.decoder(target, memory, memory_mask=self.src_mask)
        
        output = self.fc(memory)
        return F.log_softmax(output, dim=2)#softmax(output, dim=2)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        ########################################
        ######Your code here########
        ########################################
        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        pass

    def forward(self, x):
        ########################################
        ######Your code here########
        ########################################
        a = self.pe[:x.size(0), :]
        b = self.pe[:, :x.size(1)]
        x = x   + b #self.pe[:x.size(0), :]
        return self.dropout(x)