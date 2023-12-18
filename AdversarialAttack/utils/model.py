import torch
import torch.nn.functional as F
from torch.nn import Embedding, Conv1d, LSTM, Linear, BCELoss, Conv2d, ZeroPad2d

# Take the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.embedder1 = Embedding(num_embeddings=316, embedding_dim=16)
        self.embedder2 = Embedding(num_embeddings=289, embedding_dim=8)

        self.cnn1_1 = Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 16), stride=1, padding=(1, 0))
        self.cnn1_2 = Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=1)
        self.cnn1_3 = Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 16), stride=1, padding=(2, 0))

        self.cnn2 = Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 8), stride=4)

        self.lstm = LSTM(input_size=512, hidden_size=100, bidirectional=True, batch_first=True)

        self.lin1 = Linear(200, 64)
        self.lin2 = Linear(64, 32)
        self.lin3 = Linear(32, 1)

    def forward(self, data):
        x_name, x_behavior = data.split([1000, 4000], 1)

        x_name = self.embedder1(x_name)
        x_behavior = self.embedder2(x_behavior)

        x_name = x_name.unsqueeze(1)
        x_behavior = x_behavior.unsqueeze(1)

        pad = ZeroPad2d(padding=(0, 0, 2, 1))
        x_name_pad = pad(x_name)

        x_name_cnn1 = F.relu(self.cnn1_1(x_name)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn2 = F.relu(self.cnn1_2(x_name_pad)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn3 = F.relu(self.cnn1_3(x_name)).squeeze(-1).permute(0, 2, 1)

        x_behavior = F.relu(self.cnn2(x_behavior)).squeeze(-1).permute(0, 2, 1)

        x = torch.cat([x_name_cnn1, x_name_cnn2, x_name_cnn3, x_behavior], dim=-1)

        x, (h_n, c_n) = self.lstm(x)

        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]

        x = torch.cat([output_fw, output_bw], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.sigmoid(self.lin3(x))

        return x


# Define a custom class for the model
class CustomNet(Net):
    # Differentiable forward function for the model
    # Note: the sigmoid has been removed to increase gradient stability
    def diff_forward(self, name_embeddings, behavior_embeddings):
        x_name = name_embeddings.unsqueeze(1)
        x_behavior = behavior_embeddings.unsqueeze(1)

        pad = ZeroPad2d(padding=(0, 0, 2, 1))
        x_name_pad = pad(x_name)

        x_name_cnn1 = F.relu(self.cnn1_1(x_name)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn2 = F.relu(self.cnn1_2(x_name_pad)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn3 = F.relu(self.cnn1_3(x_name)).squeeze(-1).permute(0, 2, 1)

        x_behavior = F.relu(self.cnn2(x_behavior)).squeeze(-1).permute(0, 2, 1)

        x = torch.cat([x_name_cnn1, x_name_cnn2, x_name_cnn3, x_behavior], dim=-1)

        x, (h_n, c_n) = self.lstm(x)

        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]

        x = torch.cat([output_fw, output_bw], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)

        return x
  

    # Function to perform the inference of a single api sequence
    # Input: api sequence (name + semantic) as Numpy
    # Output: binary prediction
    def single_inference(self, name_embeddings, behavior_embeddings):
        self.eval() # set the eval switch

        # From Numpy to Tensor
        name_embeddings = torch.from_numpy(name_embeddings)
        behavior_embeddings = torch.from_numpy(behavior_embeddings)

        # Send data to the device
        name_embeddings = name_embeddings.to(device)
        behavior_embeddings = behavior_embeddings.to(device)

        # pred = self.diff_forward(name_embeddings, behavior_embeddings)[0][0]
        pred = self.diff_forward(name_embeddings, behavior_embeddings)
        pred = F.sigmoid(pred)
        pred = 1 if pred >= 0.5 else 0
        return pred
    
    def single_inference_thr(self, name_embeddings, behavior_embeddings):
        self.eval() # set the eval switch

        # From Numpy to Tensor
        name_embeddings = torch.from_numpy(name_embeddings)
        behavior_embeddings = torch.from_numpy(behavior_embeddings)

        # Send data to the device
        name_embeddings = name_embeddings.to(device)
        behavior_embeddings = behavior_embeddings.to(device)

        # pred = self.diff_forward(name_embeddings, behavior_embeddings)[0][0]
        pred = self.diff_forward(name_embeddings, behavior_embeddings)
        pred = F.sigmoid(pred)
        pred_ = 1 if pred >= 0.001 else 0
        if pred_ == 0:
            print('--------------------')
            print(pred)
        return pred_
    
    def single_score(self, name_embeddings, behavior_embeddings):
        self.eval() # set the eval switch

        # From Numpy to Tensor
        name_embeddings = torch.from_numpy(name_embeddings)
        behavior_embeddings = torch.from_numpy(behavior_embeddings)

        # Send data to the device
        name_embeddings = name_embeddings.to(device)
        behavior_embeddings = behavior_embeddings.to(device)

        # pred = self.diff_forward(name_embeddings, behavior_embeddings)[0][0]
        pred = self.diff_forward(name_embeddings, behavior_embeddings)
        pred = F.sigmoid(pred)
        return pred
    
    def pred_given_split(self, x_name, x_behavior):
        x_name = self.embedder1(x_name)
        x_behavior = self.embedder2(x_behavior)

        x_name = x_name.unsqueeze(1)
        x_behavior = x_behavior.unsqueeze(1)

        pad = ZeroPad2d(padding=(0, 0, 2, 1))
        x_name_pad = pad(x_name)

        x_name_cnn1 = F.relu(self.cnn1_1(x_name)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn2 = F.relu(self.cnn1_2(x_name_pad)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn3 = F.relu(self.cnn1_3(x_name)).squeeze(-1).permute(0, 2, 1)

        x_behavior = F.relu(self.cnn2(x_behavior)).squeeze(-1).permute(0, 2, 1)

        x = torch.cat([x_name_cnn1, x_name_cnn2, x_name_cnn3, x_behavior], dim=-1)

        x, (h_n, c_n) = self.lstm(x)

        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]

        x = torch.cat([output_fw, output_bw], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.sigmoid(self.lin3(x))  
        x = 1 if x >= 0.5 else 0
        return x        
    
    def score_given_split(self, x_name, x_behavior):
        x_name = self.embedder1(x_name)
        x_behavior = self.embedder2(x_behavior)

        x_name = x_name.unsqueeze(1)
        x_behavior = x_behavior.unsqueeze(1)

        pad = ZeroPad2d(padding=(0, 0, 2, 1))
        x_name_pad = pad(x_name)

        x_name_cnn1 = F.relu(self.cnn1_1(x_name)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn2 = F.relu(self.cnn1_2(x_name_pad)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn3 = F.relu(self.cnn1_3(x_name)).squeeze(-1).permute(0, 2, 1)

        x_behavior = F.relu(self.cnn2(x_behavior)).squeeze(-1).permute(0, 2, 1)

        x = torch.cat([x_name_cnn1, x_name_cnn2, x_name_cnn3, x_behavior], dim=-1)

        x, (h_n, c_n) = self.lstm(x)

        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]

        x = torch.cat([output_fw, output_bw], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.sigmoid(self.lin3(x))  
        return x        