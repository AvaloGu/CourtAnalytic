import torch
import torch.nn as nn

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