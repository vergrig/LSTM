from LSTMCell.py import LSTMCell

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = LSTMCell(input_dim, hidden_dim)  
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    
    def forward(self, x, h, c):
        sz = x.shape[1]
        #print("h, lstm = ", h.shape)
        
        for i in range(sz):
            h, c = self.lstm(x[:, i], (h, c))
        
        h = self.fc(h)
        return h
