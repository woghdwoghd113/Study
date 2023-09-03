import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from modules import LSTM, CustomCNN

class ConvLSTM(nn.Module):
    def __init__(self, input_length, output_length, num_classes, cnn_layers=None,
                 cnn_input_dim=1, rnn_input_dim=256,
                 cnn_hidden_size=256, rnn_hidden_size=512, rnn_num_layers=1, rnn_dropout=0.0,):
        # NOTE: you can freely modify hyperparameters argument
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_input_dim = cnn_input_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.input_length = input_length
        self.output_length = output_length
        self.num_classes = num_classes
        
        self.conv = CustomCNN(input_dim=cnn_input_dim, hidden_size=cnn_hidden_size)
        self.embedding = nn.Embedding(num_embeddings=num_classes+2, embedding_dim=rnn_input_dim)
        self.lstm_model = LSTM(input_dim=rnn_input_dim, hidden_size=rnn_hidden_size,
                                 num_layers=rnn_num_layers, vocab_size=num_classes, dropout=rnn_dropout)


    def forward(self, inputs):
        """
        input is (images, labels) (training phase) or images (test phase)
        images: sequential features of (Batch x Input_length, Channel=1, Height, Width)
        labels: (Batch x Output_length,)
        outputs should be a size of (Batch x Output_length, Num_classes)
        """
        have_labels = False
        if len(inputs) == 2:
            have_labels = True
            images, labels = inputs
        else:
            images = inputs

        batch_size = images.size(0) // self.input_length
        hidden_state = torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hidden_size)).to(images.device)
        cell_state = torch.zeros((self.rnn_num_layers, batch_size, self.rnn_hidden_size)).to(images.device)
        lstm_input = torch.zeros(batch_size, self.rnn_hidden_size).to(images.device)

        # CNN forward pass
        cnn_output = self.conv(images)
        cnn_output = cnn_output.view(batch_size, self.input_length, -1)
        cnn_output = cnn_output.permute(1,0,2)
        _, hidden_state, cell_state = self.lstm_model(cnn_output,hidden_state,cell_state)

        start_token = 27
        if have_labels:
            labels = labels.view(batch_size, -1)
            labels = torch.cat([torch.full((labels.size(0), 1), start_token).to(images.device), labels], dim=1).to(images.device)
            
        outputs = [torch.full((batch_size, 1), start_token).to(images.device)]
        for i in range(self.output_length):
            if have_labels:
                lstm_input = labels[:, i].unsqueeze(1)
            else:
                if i == 0:
                    lstm_input = outputs[-1]
                else:
                    lstm_input = torch.argmax(outputs[-1], dim=-1).view(batch_size, 1)
            
            embedded = self.embedding(lstm_input)
            embedded = embedded.permute(1,0,2)
            output, hidden_state, cell_state = self.lstm_model(embedded, hidden_state, cell_state)
            outputs.append(output.squeeze(1))

        outputs = outputs[1:]
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2).reshape(-1, self.num_classes)
        
        return outputs