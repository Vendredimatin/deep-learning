import torch
import torch.nn as nn

class RNNModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, ninput, nhid, nlayers):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, ninput)
        # Construct you RNN model here. You can add additional parameters to the function.
        ########################################
        ######Your code here########
        self.lstm = nn.LSTM(input_size=ninput,hidden_size=nhid,num_layers=nlayers,batch_first=False)
        ########################################
        

        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def init_hidden(self, batch_size):
        if self.lstm.bidirectional:
            return (torch.rand(self.nlayers*2,batch_size,self.nhid),torch.rand(self.nlayers*2,batch_size,self.nhid))
        else:
            return (torch.rand(self.nlayers,batch_size,self.nhid),torch.rand(self.nlayers,batch_size,self.nhid))


    def forward(self, input, hidden):
        embeddings = self.drop(self.encoder(input))
        #self.hidden = self.init_hidden(input.size(1))
    
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes

        ########################################
        ######Your code here########
        ########################################
        output, hidden = self.lstm(embeddings,hidden)
    

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

