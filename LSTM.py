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

   
class SimpleLSTM(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()

        self.emb = embedding_layer
        self.bert_dim = 768 # берт по умолчанию
        self.hidden_dim = 384
        self.classes = 4
        self.lstm = LSTM(self.bert_dim, self.hidden_dim, self.classes)

    def forward(self, input):
        x = self.emb(input)

        h = torch.zeros((x.shape[0], self.hidden_dim)).cuda()
        c = torch.zeros((x.shape[0], self.hidden_dim)).cuda()
        #print("h, simple =", h.shape)
        x = self.lstm(x, h, c)

        return x
