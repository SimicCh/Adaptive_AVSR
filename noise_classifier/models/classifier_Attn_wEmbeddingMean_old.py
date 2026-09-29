import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class Encoder_Block(nn.Module):
    def __init__(self, 
                 Attn_num_heads=8,
                 Attn_num_features=512,
                 Attn_dropout=0.1,
                 FF_hidden_dim=2048,
                 FF_dropout=0.1): 
        super().__init__()

        # Multi-Head Attention Block
        self.attn = MultiHeadedAttention(Attn_num_heads, Attn_num_features, dropout_rate=Attn_dropout)
        self.attn_dropout = nn.Dropout(Attn_dropout)
        self.attn_norm = nn.LayerNorm(Attn_num_features)

        # Feed-Forward Block
        self.ff = nn.Sequential(
            nn.Linear(Attn_num_features, FF_hidden_dim),
            nn.ReLU(),  # Aktivierungsfunktion aus dem Paper
            nn.Dropout(FF_dropout),
            nn.Linear(FF_hidden_dim, Attn_num_features),
            nn.Dropout(FF_dropout)
        )
        self.ff_norm = nn.LayerNorm(Attn_num_features)

    def forward(self, x, mask):
        # Multi-Head Attention Block
        x_attn_out = self.attn(x, x, x, mask)
        x = self.attn_norm(x + self.attn_dropout(x_attn_out))  # Residual + Norm

        # Feed-Forward Block
        x_ff_out = self.ff(x)
        x = self.ff_norm(x + x_ff_out)  # Residual + Norm

        return x





class Categorial_FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()
        
        # Feed-Forward Block
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),  # Aktivierungsfunktion aus dem Paper
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout_rate)
        )
        self.ff_norm = nn.LayerNorm(in_dim)

        self.out_fc = nn.Linear(in_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x):
        # Feed-Forward Block
        x_ff_out = self.ff(x)
        x = self.ff_norm(x + x_ff_out)

        x = self.out_fc(x)
        x = self.softmax(x)
        
        return x


class Regression_FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate):
        super().__init__()
        
        # Feed-Forward Block
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),  # Aktivierungsfunktion aus dem Paper
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout_rate)
        )
        self.ff_norm = nn.LayerNorm(in_dim)

        self.out_fc = nn.Linear(in_dim, 1)



    def forward(self, x):
        # Feed-Forward Block
        x_ff_out = self.ff(x)
        x = self.ff_norm(x + x_ff_out)

        x = self.out_fc(x)
        
        return x




class Noise_Classifier_Attn_wEmbMean_old(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        
        input_dim = model_config['input_dim']
        processing_dim = model_config['processing_dim']
        out_dim = model_config['out_dim']
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        hidden_dim = model_config['hidden_dim']
        attn_dropout = model_config['attn_dropout']
        ff_dropout = model_config['ff_dropout']
        classifier_num_classes = model_config['classifier_num_classes']

        self.inp_fc = nn.Linear(input_dim, processing_dim)

        self.layers = nn.ModuleList([
            Encoder_Block(
                Attn_num_heads=num_heads,
                Attn_num_features=processing_dim,
                Attn_dropout=attn_dropout,
                FF_hidden_dim=hidden_dim,
                FF_dropout=ff_dropout
            ) for _ in range(num_layers)
        ])

        self.fc1 = nn.Linear(processing_dim, out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

        self.classifier = Categorial_FeedForward(out_dim, int(out_dim // 2), classifier_num_classes, ff_dropout)
        self.regressor = Regression_FeedForward(out_dim, int(out_dim // 2), ff_dropout)
        # self.softmax = nn.Softmax(dim=1)


    def create_attention_mask(self, lens, device):
        """
        Erstelle eine Maske basierend auf den Sequenzlängen ohne Padding.
        
        :param lens: Tensor mit den Längen der Sequenzen (batch_size,)
        :return: Eine Maske mit Form (batch_size, 1, 1, max_len), wo 0 für "real" und 1 für Padding-Positionen steht.
        """
        # lens = torch.tensor(lens)
        max_len = torch.max(lens)
        # Maske erstellen, die 1 für Padding und 0 für echte Sequenz-Positionen hat
        mask = torch.arange(max_len).unsqueeze(0).repeat(lens.shape[0], 1).to(device) < lens.unsqueeze(1).to(device)
        
        # Die Maske umformen, damit sie für MHA funktioniert (batch_size, 1, 1, max_len)
        mask = mask.unsqueeze(1)  # (batch_size, 1, max_len)
        
        return mask.to(device)
        


    def forward(self, x, lens):
        """
        Forward pass of the Noise_Classifier_Attn model.
        
        :param x: Input tensor of shape (batch_size, input_dim, seq_len)
        :param lens: List or tensor of sequence lengths (batch_size,)
        :return: Tuple of (classification_output, regression_output)
        """
        # Shorten x if possible
        x = x[:,:,:torch.max(lens)]
        # Prepare mask
        mask = self.create_attention_mask(lens.to(x.device), device=x.device)
        # mask = mask.to(x.device)
        # Permute input tensor to processing shape
        x = x.permute(0, 2, 1)

        x = self.inp_fc(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        mask = mask.squeeze(1).unsqueeze(-1)
        x_embedding = x * mask
        x_embedding = x_embedding.sum(1) / mask.sum(1)

        x_class = self.classifier(x_embedding)
        x_regr = self.regressor(x_embedding)

        return x_class, x_regr, x_embedding
        


    def get_embedding(self, x, lens=None):
        """
        Forward pass of the Noise_Classifier_Attn model.
        
        :param x: Input tensor of shape (batch_size, input_dim, seq_len)
        :param lens: List or tensor of sequence lengths (batch_size,)
        :return: Tuple of (classification_output, regression_output)
        """
        # Shorten x if possible
        if lens is None:
            lens = torch.ones(x.shape[0], device=x.device) * x.shape[2]
            
        x = x[:,:,:int(torch.max(lens).item())]
        # Prepare mask
        mask = self.create_attention_mask(lens.to(x.device), device=x.device)
        # mask = mask.to(x.device)
        # Permute input tensor to processing shape
        x = x.permute(0, 2, 1)

        x = self.inp_fc(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        mask = mask.squeeze(1).unsqueeze(-1)
        x_embedding = x * mask
        x_embedding = x_embedding.sum(1) / mask.sum(1)

        return x_embedding








if __name__ == "__main__":

    model_config = {}
    model_config['input_dim'] = 80
    model_config['processing_dim'] = 256
    model_config['out_dim'] = 256
    model_config['num_layers'] = 4
    model_config['num_heads'] = 8
    model_config['hidden_dim'] = 512
    model_config['attn_dropout'] = 0.1
    model_config['ff_dropout'] = 0.1
    model_config['classifier_num_classes'] = 4  

    model = Noise_Classifier_Attn(model_config)

    # get number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")


    # params = list(model.parameters())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)

    inp = torch.rand(2, 80, 36)
    len_list = torch.tensor([20, 36])
    out = model(inp, len_list)

    print(out[0].shape)
    print(out[1].shape)

    print(out[0])
    print(out[1])

    print('Done')
