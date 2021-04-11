class LSTMCell(nn.Module):

    def __init__(self, input_sz, hidden_sz):
        super().__init__()

        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        
        # матрица W
        self.w_ii = nn.Parameter(torch.zeros(input_sz, hidden_sz))
        self.w_hi = nn.Parameter(torch.zeros(hidden_sz, hidden_sz))
        self.w_if = nn.Parameter(torch.zeros(input_sz, hidden_sz))
        self.w_hf = nn.Parameter(torch.zeros(hidden_sz, hidden_sz))
        self.w_ig = nn.Parameter(torch.zeros(input_sz, hidden_sz))
        self.w_hg = nn.Parameter(torch.zeros(hidden_sz, hidden_sz))
        self.w_io = nn.Parameter(torch.zeros(input_sz, hidden_sz))
        self.w_ho = nn.Parameter(torch.zeros(hidden_sz, hidden_sz))        
        
        # BIAS
        self.b_ii = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_if = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_io = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_sz))

        # Функциональные слои
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / self.hidden_sz ** 0.5
        for cur in self.parameters():
            cur.data.uniform_(-stdv, stdv)
    
    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        #print("h, cell = ", h_prev.shape)

        i_t = x @ self.w_ii + self.b_ii + h_prev @ self.w_hi + self.b_hi
        i_t = self.sigm(i_t)

        f_t = x @ self.w_if + self.b_if + h_prev @ self.w_hf + self.b_hf
        f_t = self.sigm(f_t)

        g_t = x @ self.w_ig + self.b_ig + h_prev @ self.w_hg + self.b_hg
        g_t = self.tanh(g_t)

        o_t = x @ self.w_io + self.b_io + h_prev @ self.w_ho + self.b_ho
        o_t = self.sigm(o_t)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * self.tanh(c_t)
        
     
        return (h_t, c_t)
