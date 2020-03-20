import torch
import torch.nn as nn


class Chen_IEEE(nn.Module):
    def __init__(self,
                 input_size=6,
                 hidden_size=256,
                 num_layers=2,
                 output_size=4):
        super(Chen_IEEE, self).__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        intermediate     = hidden_size // 2
        self.FC_in       = nn.Linear(input_size, intermediate)
        self.BiLSTM      = nn.LSTM(input_size=intermediate,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False)
        self.FC_out      = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden == None:
            self.batch_size = x.shape[1]
            hidden          = self.init_hidden()

        x = self.FC_in(x)
        x, hidden = self.BiLSTM(x, hidden)

        # I believe this is how it should be done...
        # Note that hidden is a list, where the first
        # value is the hidden, and the second is the cell.
        x = self.FC_out(hidden[0])
        # x = self.FC_out(x)
        return x, hidden

    def init_hidden(self):
        hidden_state = torch.randn(self.num_layers,
                                   self.batch_size,
                                   self.hidden_size)
        cell_state = torch.randn(self.num_layers,
                                 self.batch_size,
                                 self.hidden_size)

        return hidden_state, cell_state
