import torch.nn as nn
from .rnn_encoder import RNNEncoder


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.0, use_gpu=False):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self.rnn = RNNEncoder(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout)

        if self.use_gpu:
            self.cuda()

    def forward(self, input):

        assert isinstance(input, tuple)
        assert self.bidirectional

        _, last_hidden = self.rnn(input)

        output = self.output_layer(last_hidden)
        prob = self.sigmoid(output)
        return prob.view(-1)



