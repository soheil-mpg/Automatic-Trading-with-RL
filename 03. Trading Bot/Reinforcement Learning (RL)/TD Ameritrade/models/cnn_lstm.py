
# Import the libraries
import torch

# Network architecture
class Network(torch.nn.Module):

    # Constructor
    def __init__(self, input_size, actions_n):
    
        # Initialize the parent't constructor
        super(Network, self).__init__()
        
        # CNN layers
        self.conv1 = torch.nn.Conv1d(in_channels = input_size, out_channels = 512, kernel_size = 1)
        self.conv2 = torch.nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1)
        self.conv3 = torch.nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 1)
        self.conv_output_size = self.conv2(self.conv1(torch.zeros(1, input_size, 1))).shape[-1]
        
        # LSTM layers
        self.lstm = torch.nn.LSTM(input_size = self.conv_output_size, hidden_size = 512, num_layers = 4, bidirectional = True)
        self.lstm_output_size = self.lstm(self.conv2(self.conv1(torch.zeros(1, input_size, 1))))[0].shape[-1]
        
        # Relu layer
        self.relu = torch.nn.ReLU()
        
        # FCL layer
        self.fcl1 = torch.nn.Linear(in_features = self.lstm_output_size, out_features = 512)
        
        # FCL layers - Value
        self.val1 = torch.nn.Linear(in_features = 512, out_features = 512)
        self.val2 = torch.nn.Linear(in_features = 512, out_features = 1)
        
        # FCL layers - Advantage Value
        self.adv1 = torch.nn.Linear(in_features = 512, out_features = 512)
        self.adv2 = torch.nn.Linear(in_features = 512, out_features = actions_n)
        
    # Forward function
    def forward(self, x):
        
        # Convert input type to float32
        x = torch.tensor(x, dtype = torch.float32)
        
        # Add an extra dimension
        x = x.unsqueeze(-1)
        
        # Feedforward to CNN layers
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))
        
        # Feedforward to LSTM layers
        lstm_out, (hn, cn) = self.lstm(conv3_out)
        lstm_out = self.relu(lstm_out)
        
        # Feedforward to FCL layer
        fc1_out = self.relu(self.fcl1(lstm_out))
        
        # Output the value
        val1_out = self.relu(self.val1(fc1_out))
        val2_out = self.val2(val1_out)
        
        # Output the advantage value
        adv1_out = self.relu(self.adv1(fc1_out))
        adv2_out = self.adv2(adv1_out)
        
        # Get the final output
        output = val2_out + (adv2_out - adv2_out.mean(dim = 1, keepdim = True))
        
        # Get the mean along the dimension 1
        output = torch.mean(output, dim = 1)
        
        return output
