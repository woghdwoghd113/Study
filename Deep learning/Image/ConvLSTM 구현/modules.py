import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math

class CustomCNN(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 3 * 3, hidden_size)

    def forward(self, inputs):
        inputs = inputs.view(-1, 1, 28, 28)
        inputs = inputs.float()

        out = self.conv1(inputs)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()

        # Define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # LSTM cell
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.fc_in = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, feature, h, c):
        output, (h_next, c_next) = self.lstm(feature, (h, c))
        output = self.fc_in(output)
        output = self.fc_out(output)

        return output, h_next, c_next
