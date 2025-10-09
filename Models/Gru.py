import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h_prev)) # (Hidden,)
        r = torch.sigmoid(self.Wr(x) + self.Ur(h_prev)) # (Hidden,)
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h_prev)) # (Hidden,)
        h = (1 - z) * h_prev + z * h_tilde # (Hidden,)
        return h
    

class NaiveGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=5):
        super().__init__()
        
        self.hidden_size = hidden_size # Hidden
        self.num_layers = num_layers # Layers
        
        self.cells = nn.ModuleList([GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h = None):
        num_frames, channels = x.size() # (NumFrames, C)

        if h is None:
            h = [torch.zeros(self.hidden_size).to(x.device) for _ in range(self.num_layers)] # list of (Hidden,) tensors

        out = []
        for t in range(num_frames):
            x_t = x[t, :] # (C,)
            for i in range(self.num_layers):
                h[i] = self.cells[i](x_t, h[i]) # (Hidden,)
                x_t = h[i]
            out.append(h[-1].unsqueeze(0)) # (1, Hidden)
        
        out = torch.cat(out, dim=0) # (NumFrames, Hidden)
        out = self.fc(out) # (NumFrames, 2_classes)
        return out, h # (NumFrames, 2_classes), list of (Hidden,) tensors
    

class EfficientGRUModel(nn.Module):
    r"""
    a wrapper around nn.GRU
    """

    def __init__(self, input_size = 768, hidden_size = 512, output_size = 6, number_layers = 5):
        super().__init__()

        # TODO: nn.gru also has dropout which you might want to use
        self.gru = nn.GRU(input_size, hidden_size, num_layers=number_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        # TODO: for efficiency, utlize the batch dim, but should we fold the frames?
        # TODO: should we fold the frames using view?
        # TODO: how did they implement GRU without for loop in?
        # x: (NumFrames, C) -> add batch dim at dim=1
        x = x.unsqueeze(1) # (NumFrames, 1, C)

        if h is None:
            h0 = None
        else:
            # already (Layers, 1, Hidden)
            h0 = h.to(x.device)

        y, hn = self.gru(x, h0) # y:(NumFrames, 1, Hidden), hn:(Layers, 1, Hidden)
        y = y.squeeze(1) # (NumFrames, Hidden)
        out = self.fc(y) # (NumFrames, output_size)
        return out, hn # (NumFrames, output_size), (Layers, 1, Hidden)
    

class GRUEncoder(nn.Module):
    r"""
    GRU encoder for stage 2
    """

    def __init__(self, input_size = 768, hidden_size = 512, number_layers = 5):
        super().__init__()

        # bidirectional GRU
        self.num_layers = number_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=number_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, hidden_size)
        self.ln_hid = nn.LayerNorm(hidden_size*2)
        self.ln = nn.LayerNorm(hidden_size*2)

    def forward(self, x, h=None):
        # x: (NumFrames, C) -> add batch dim at dim=1
        x = x.unsqueeze(1) # (NumFrames, 1, C)

        y, hn = self.gru(x, h) # y:(NumFrames, 1, Hidden*2), hn:(2*Layers, 1, Hidden)

        hn = hn.view(self.num_layers, 2, 1, self.hidden_size) # (Layers, 2, 1, Hidden)
        h_forward = hn[:, 0] # (Layers, 1, hidden)
        h_backward = hn[:, 1] # (Layers, 1, hidden)
        hidden = torch.cat((h_forward, h_backward), dim=-1) # (Layers, 1, 2*hidden)
        hidden = self.fc(self.ln_hid(hidden)) # (Layers, 1, hidden)

        y = y.squeeze(1) # (NumFrames, Hidden*2)
        out = self.ln(y) # (NumFrames, hidden_size*2)

        return out, hidden # (NumFrames, hidden_size*2), (Layers, 1 hidden)
        

class GRUDecoder(nn.Module):
    r"""
    GRU decoder for stage 2
    """

    def __init__(self, hidden_size = 512, vocab_size = 14, number_layers = 5):
        super().__init__()

        self.hidden_size = hidden_size

        self.embd = nn.Embedding(vocab_size, hidden_size)

        self.att_proj_q = nn.Linear(hidden_size, hidden_size)
        self.att_proj_k = nn.Linear(hidden_size*2, hidden_size)
        self.att_proj_v = nn.Linear(hidden_size*2, hidden_size)

        self.gru = nn.GRU(2*hidden_size, hidden_size, num_layers=number_layers, batch_first=False)

        self.ln_hid = nn.LayerNorm(hidden_size)
        self.ln_out = nn.LayerNorm(3*hidden_size)
        self.fc_out = nn.Linear(3*hidden_size, vocab_size)

    def forward(self, enc_output, shot, hidden):
        # enc_output: (NumFrames, hidden_size*2)
        # shot: (,)
        # hidden: (Layers, 1 hidden_size)

        shot = shot.view(1, 1) # (1, 1)
        tok_emb = self.embd(shot) # (1, 1, hidden_size)
        s = hidden[-1] # (1, hidden_size)

        query = self.att_proj_q(s) # (1, hidden_size)
        key = self.att_proj_k(enc_output) # (NumFrames, hidden_size)
        att = key @ query.transpose(1, 0) # (NumFrames, 1)
        alphas = F.softmax(att, dim=0) # (NumFrames, 1)

        value = self.att_proj_v(enc_output) # (NumFrames, hidden_size)
        c = value.transpose(1,0) @ alphas # (hidden_size, 1)
        c = c.view(1, 1, -1) # (1, 1, hidden_size)

        gru_input = torch.cat((tok_emb, c), dim = 2) #  # (1, 1, 2*hidden_size)
        output, h = self.gru(gru_input, hidden) # (1, 1, hidden_size), (Layers, 1, hidden_size)

        h = self.ln_hid(h)

        pre_pred = torch.cat((output, gru_input), dim=2).squeeze(1) # (1, 3*hidden_size)
        prediction = self.fc_out(self.ln_out(pre_pred)) # (1, vocab_size)

        return prediction.squeeze(0), h # (1, vocab_size), (Layers, 1, hidden_size)
        

class GRUStage2(nn.Module):
    r"""
    GRU for stage 2
    """

    def __init__(self, input_size = 768, hidden_size = 512, vocab_size = 14, number_layers = 5):
        super().__init__()

        self.encoder = GRUEncoder(input_size=input_size, hidden_size=hidden_size, number_layers=number_layers)
        self.decoder = GRUDecoder(hidden_size= hidden_size, vocab_size=vocab_size, number_layers=number_layers)

    def forward(self, x, y, h=None):
        # x: (NumFrames, C) -> add batch dim at dim=1
        # y: (Num of shots in this point,)

        # append start of the point token
        point_token = torch.tensor([13], dtype=torch.long, device=y.device)
        y = torch.cat((point_token, y), dim=0) # (Num of shots in this point+1,)

        enc_output, h = self.encoder(x) # (NumFrames, hidden_size*2), (Layers, 1 hidden)

        outputs = [] # a list of (vocab_size,) tensors
        for i in range(len(y)-1):
            pred, h = self.decoder(enc_output, y[i], h)
            outputs.append(pred)

        logits = torch.stack(outputs, dim=0) # (num of shots, vocab_size)
        return logits
