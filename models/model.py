import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, output_size=1, dropout_prob=0.2, activation_function=None, norm=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.output_size = output_size
        if activation_function is None:
            activation_function= nn.ReLU
        in_size = input_size[0] * input_size[1] if len(input_size) == 2 else input_size[0]

        self.layers.append(nn.Linear(in_size, hidden_size))
        self.layers.append(activation_function())
        if norm == 'batch':
            self.layers.append(nn.BatchNorm1d(hidden_size))
        elif norm == 'instance':
            self.layers.append(nn.InstanceNorm1d(hidden_size))
        self.layers.append(nn.Dropout(dropout_prob))

        current_hidden_size = hidden_size
        for _ in range(layers - 1):
            next_hidden_size = current_hidden_size // 2 if current_hidden_size > 1 else 1
            self.layers.append(nn.Linear(current_hidden_size, next_hidden_size))
            self.layers.append(activation_function())
            if norm == 'batch':
                self.layers.append(nn.BatchNorm1d(next_hidden_size))
            elif norm == 'instance':
                self.layers.append(nn.InstanceNorm1d(next_hidden_size))
            self.layers.append(nn.Dropout(dropout_prob))
            current_hidden_size = next_hidden_size
        
        # Output layer
        self.layers.append(nn.Linear(current_hidden_size, output_size))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        if self.output_size == 1:
            return x.view(-1, 1)
        else:
            return x
        
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, output_size=1,dropout_prob=0.2,activation_function=None):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size[1], hidden_size, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        if activation_function is None:
            activation_function= nn.ReLU
        self.activation_function = activation_function()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_last_step = lstm_out[:, -1, :]
        lstm_out_last_step = self.dropout(lstm_out_last_step)

        lstm_out_last_step = self.activation_function(lstm_out_last_step)

        return self.fc(lstm_out_last_step)

############################ CNN1D ############################### CNN1D ############################ CNN1D #####################
class CNN1D(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, output_size=1, dropout_prob=0.2, activation_function=None, norm=None):
        super(CNN1D, self).__init__()
        self.layers = nn.ModuleList()
        self.output_size = output_size
        if input_size[0] < kernel_size:
            kernel_size = int(input_size[0] // 2)
        if input_size[0] == 1:
            kernel_size = 1
        if activation_function is None:
            activation_function = nn.ReLU
        self.layers.append(nn.Conv1d(input_size[1], hidden_size, kernel_size=kernel_size))
        if norm == 'batch':
            self.layers.append(nn.BatchNorm1d(hidden_size))
        elif norm == 'instance':
            self.layers.append(nn.InstanceNorm1d(hidden_size))
        self.layers.append(activation_function())
        self.layers.append(nn.Dropout(dropout_prob))
       
        layer_count = 1
        
        max_layers = int(input_size[0] // kernel_size)
        print(f"max_layers: {max_layers}, input_size: {input_size[0]}, kernel_size: {kernel_size}")
        while hidden_size > 1 and layer_count < max_layers:
            next_hidden_size = hidden_size // 4 if layer_count >= 3 and hidden_size > 512 else hidden_size // 2
            if next_hidden_size < kernel_size:
                next_hidden_size = kernel_size
            self.layers.append(nn.Conv1d(hidden_size, next_hidden_size, kernel_size=kernel_size, padding=1))
            if norm == 'batch':
                self.layers.append(nn.BatchNorm1d(next_hidden_size))
            elif norm == 'instance':
                self.layers.append(nn.InstanceNorm1d(next_hidden_size))
            self.layers.append(activation_function())
            self.layers.append(nn.Dropout(dropout_prob))
            hidden_size = next_hidden_size
            layer_count += 1

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=2)
        if self.output_size == 1:
            return self.fc(x).reshape(-1, 1)
        else:
            return self.fc(x)
        
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, output_size=1,dropout_prob=0.2,activation_function=None):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size[1], hidden_size, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation_function = activation_function() if callable(activation_function) else nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_last_step = lstm_out[:, -1, :]
        lstm_out_last_step = self.dropout(lstm_out_last_step)

        lstm_out_last_step = self.activation_function(lstm_out_last_step)

        return self.fc(lstm_out_last_step)
    

############################ CNNLSTM ############################# CNNLSTM ########################## CNNLSTM #####################
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers=1, output_size=1,dropout_prob=0.2):
        super(CNNLSTM, self).__init__()
        dropout_prob=dropout_prob
        sequence_length, input_channels = input_size  # Extracting the sequence length and number of channels
        self.conv_layers = nn.ModuleList()
        kernel_size = min(3, input_channels)
        self.conv_layers.append(nn.Conv1d(input_channels, hidden_size, kernel_size=kernel_size, padding=1))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.Dropout(dropout_prob))
        self.dropout = nn.Dropout(dropout_prob)
        layer_count = 1
        while hidden_size > 1 and layer_count < 5:
            next_hidden_size = hidden_size // 4 if layer_count >= 3 and hidden_size > 512 else hidden_size // 2
            self.conv_layers.append(nn.Conv1d(hidden_size, next_hidden_size, kernel_size=kernel_size, padding=1))
            self.conv_layers.append(nn.ReLU())
            hidden_size = next_hidden_size
            layer_count += 1

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permuting the dimensions to (batch_size, channels, sequence_length)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)  # Permuting back to (batch_size, sequence_length, channels)
        lstm_out, _ = self.lstm(x)
        lstm_out_last_step = lstm_out[:, -1, :]
        lstm_out_last_step = self.dropout(lstm_out_last_step)
        return self.fc(lstm_out_last_step)
    
############################ NBEATS ############################## NBEATS ########################### NBEATS #####################
class GenericBlock(nn.Module):
    def __init__(self, input_size, hidden_size=256, theta=20, layers=4, output_size=1, dropout_prob=0.2, activation_function=None, norm=None):
        super(GenericBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_prob)
        in_size = input_size[0] * input_size[1] if len(input_size) == 2 else input_size[0]
        self.layers.append(nn.Linear(in_size, hidden_size))
        if norm == 'batch':
            self.layers.append(nn.BatchNorm1d(hidden_size))
        elif norm == 'instance':
            self.layers.append(nn.InstanceNorm1d(hidden_size))
        if activation_function is None:
            activation_function = nn.ReLU
        self.layers.append(activation_function())
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            if norm == 'batch':
                self.layers.append(nn.BatchNorm1d(hidden_size))
            elif norm == 'instance':
                self.layers.append(nn.InstanceNorm1d(hidden_size))
            self.layers.append(activation_function())

        # Define two output layers with 4 units each
        self.output_layer1 = nn.Sequential(
                                nn.Linear(hidden_size, theta),
                                nn.Linear(theta, in_size))
        self.output_layer2 = nn.Sequential(
                                nn.Linear(hidden_size, theta),
                                nn.Linear(theta, output_size))
        
    def forward(self, x):
        orig_x=x
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x) 
        backcast = self.output_layer1(x)
        forecast = self.output_layer2(x)
        return (orig_x-backcast), forecast

class GenericStack(nn.Module):
    def __init__(self, input_size, hidden_size, theta, layers, n_blocks, output_size, dropout_prob=0.2, activation_function=None, norm=None):
        super(GenericStack, self).__init__()
        self.generic_blocks = nn.ModuleList([
            GenericBlock(input_size, hidden_size, theta, layers, output_size, dropout_prob, activation_function, norm)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        n,h,f=x.shape
        x = x.flatten(1)
        backcast = x
        combined_forecast = 0
        
        for block in self.generic_blocks:
            backcast, forecast = block(backcast)
            combined_forecast += forecast
        backcast=backcast.reshape(n,h,f)
        return backcast, combined_forecast

class NBEATS(nn.Module):
    def __init__(self, input_size, hidden_size, theta, layers, n_blocks, output_size, dropout_prob=0.2, activation_function=None, norm=None):
        super(NBEATS, self).__init__()
        self.generic_stack = GenericStack(input_size, hidden_size, theta, layers, n_blocks, output_size, dropout_prob, activation_function, norm)

    def forward(self, x):
        generic_backcast, generic_forecast = self.generic_stack(x)        
        return generic_backcast, generic_forecast

############################ NHITS ############################### NHITS ############################ NHITS #####################
class MaxPoolBlock(nn.Module):
    def __init__(self, input_size, hidden_size=256, theta=20, layers=4, m_pool_k=1, output_size=1, dropout_prob=0.2, activation_function=None, norm=None):
        super(MaxPoolBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        n_hist,n_feat=input_size
        orig_in_size = input_size[0] * input_size[1] if len(input_size) == 2 else input_size[0]
        pooled_size=n_hist-m_pool_k
        new_in_size = pooled_size * input_size[1] if len(input_size) == 2 else pooled_size
        
        self.adaptivepool=nn.AdaptiveMaxPool1d(pooled_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layers.append(nn.Linear(new_in_size, hidden_size))
        if norm == 'batch':
            self.layers.append(nn.BatchNorm1d(hidden_size))
        elif norm == 'instance':
            self.layers.append(nn.InstanceNorm1d(hidden_size))
        if activation_function is None:
            activation_function = nn.ReLU
        self.layers.append(activation_function())
        self.layers.append(self.dropout)
        
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            if norm == 'batch':
                self.layers.append(nn.BatchNorm1d(hidden_size))
            elif norm == 'instance':
                self.layers.append(nn.InstanceNorm1d(hidden_size))
            self.layers.append(activation_function())
            self.layers.append(self.dropout)

        self.output_layer1 = nn.Sequential(
                                nn.Linear(hidden_size, theta),
                                nn.Linear(theta, orig_in_size))
        self.output_layer2 = nn.Sequential(
                                nn.Linear(hidden_size, theta),
                                nn.Linear(theta, output_size))
    
    def reshape_tensor(tensor, channels, height):
        return tensor.reshape(-1, channels, height)
        
    def forward(self, x):
        orig_x=x
        n,h,f=x.shape
        if h>2:
            x_pool=x.permute(0, 2, 1)
            x_pool=self.adaptivepool(x_pool)            
            x_pool=x_pool.permute(0, 2, 1)
            x = x_pool.reshape(n, -1)
        else:
            x=x.reshape(n,h*f)
        for layer in self.layers:
            x = layer(x)
        backcast = self.output_layer1(x)
        forecast = self.output_layer2(x)        
        backcast=backcast.reshape(n,h,f)
        return (orig_x-backcast), forecast

class NhitsStack(nn.Module):
    def __init__(self, input_size, hidden_size, theta, layers, n_blocks, m_pool_k, output_size, dropout_prob=0.2, activation_function=None, norm=None):
        super(NhitsStack, self).__init__()
        self.generic_blocks = nn.ModuleList([
            MaxPoolBlock(input_size, hidden_size, theta, layers, m_pool_k, output_size, dropout_prob, activation_function, norm)
            for _ in range(n_blocks)
        ])
    def forward(self, x):
        backcast = x
        combined_forecast = 0
        for block in self.generic_blocks:
            backcast, forecast = block(backcast)
            combined_forecast += forecast
        return backcast, combined_forecast

class NHITS(nn.Module):
    def __init__(self, input_size, hidden_size, theta, layers, n_blocks, output_size, dropout_prob=0.2, activation_function=None, norm=None):
        super(NHITS, self).__init__()
        n_hist,n_feat=input_size
        
        m_pool_k_list = [n_hist // 2, n_hist // 3, 1, 0]  # Define kernel sizes as required
        self.NhitsStack = nn.ModuleList([
            NhitsStack(input_size, hidden_size, theta, layers, n_blocks, m_pool_k, output_size, dropout_prob, activation_function, norm)
            for m_pool_k in m_pool_k_list
        ])
    def forward(self, x):
        n, h, f = x.shape  # extract the sequence length 'h'
        backcast = x
        combined_forecast = 0
        if h in [1, 2]:
            backcast, forecast = self.NhitsStack[-1](backcast)  # Using only the first stack
            combined_forecast += forecast
        else:
            for stack in self.NhitsStack:
                backcast, forecast = stack(backcast)
                combined_forecast += forecast
        
        return backcast, combined_forecast
    
############################ NLINEAR ############################# NLINEAR ########################## NLINEAR #####################
class LTSF_NLinear(nn.Module):
    def __init__(self, window_size, forecast_size, individual, feature_size, dropout_prob=0.2, activation_function=None, norm=None):
        super(LTSF_NLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forecast_size
        self.individual = individual
        self.channels = feature_size
        self.dropout = nn.Dropout(p=dropout_prob)
        if activation_function is None:
            activation_function = nn.ReLU
        self.activation_function = activation_function()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(feature_size)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(feature_size)
        else:
            self.norm = None
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = nn.Linear(self.window_size, self.forcast_size)
    def forward(self, x):
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(self.activation_function(x.permute(0,2,1))).permute(0,2,1)
        if self.norm is not None:
            x = self.norm(x)
        x = self.dropout(x)
        x = x + seq_last
        return x

class NLINEAR(LTSF_NLinear):
    def __init__(self, window_size, forecast_size, individual, feature_size, dropout_prob=0.2, activation_function=None, norm=None):
        super(NLINEAR, self).__init__(window_size, forecast_size, individual, feature_size, dropout_prob, activation_function, norm)
        self.final_layer = nn.Linear(feature_size, 1)
    def forward(self, x):
        x = super(NLINEAR, self).forward(x)
        x = self.final_layer(self.activation_function(x))
       
        return x.squeeze(-1)
    
############################ TSTMODEL ############################ TSTMODEL ######################### TSTMODEL #####################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert d_model % 2 == 0, "d_model should be even for positional encoding."
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x) 
    
class TSTMODEL(nn.Module):
    def __init__(self, iw,feat, ow, d_model, nhead, nlayers, dropout=0.2,activation_function=None):
        super(TSTMODEL, self).__init__()
        
        self.d_model = d_model + (nhead - d_model % nhead) if d_model % nhead != 0 else d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        if activation_function is None:
            activation_function = nn.ReLU
        self.activation_function = activation_function()
            
        self.encoder = nn.Sequential(
            nn.Linear(feat, self.d_model//2),
            self.activation_function,
            nn.Linear(self.d_model//2, self.d_model)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            self.activation_function,
            nn.Linear(self.d_model//2, 1)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            self.activation_function,
            nn.Linear((iw+ow)//2, ow)
        )
        
    def _add_norm_layer(self, size, norm_type):
        if norm_type == 'batch':
            return nn.BatchNorm1d(size)
        elif norm_type == 'instance':
            return nn.InstanceNorm1d(size)
        else:
            return nn.Identity()  # No normalization

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        device = src.device
        srcmask = self.generate_square_subsequent_mask(src.shape[0]).to(device)

        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output
    
############################ TFTMODEL ############################ TFTMODEL ######################### TFTMODEL ####################
class TFTMODEL(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.2):
        super(TFTMODEL, self).__init__()

        # Ensure d_model is divisible by nhead
        self.d_model = d_model + (nhead - d_model % nhead) if d_model % nhead != 0 else d_model
        
        # Calculate the input size for the embedding layer
        in_size = input_dim[0] * input_dim[1] if isinstance(input_dim, (list, tuple)) and len(input_dim) == 2 else input_dim
        self.in_size = in_size
        # Embedding layer to transform input to d_model size
        self.embedding_layer = nn.Linear(in_size, self.d_model)

        # Define the Transformer Encoder and Decoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dropout=dropout, batch_first=True)

        # Define the Transformer Encoder and Decoder
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Final fully connected layer
        self.fc = nn.Linear(self.d_model, output_dim)

    def forward(self, x):
        # Embedding the input

        x = x.view(-1, self.in_size)

        x = x.unsqueeze(1) 
        
        x_embedded = self.embedding_layer(x)
        
        # Encoding
        encoder_output = self.encoder(x_embedded)

        # Decoding
        decoder_output = self.decoder(x_embedded, encoder_output)
        
        # Pass through the final fully connected layer
        out = self.fc(decoder_output[:, -1, :])
        return out
    

class WeightAdaptingNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout_prob=0.2, activation_function=None, norm=None):
        super(WeightAdaptingNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.in_size = input_size[0] * input_size[1] if len(input_size) == 2 else input_size[0]

        # Set activation function
        if activation_function is None:
            activation_function = nn.ReLU
        self.activation_function = activation_function()

        # Layers
        self.fc1 = nn.Linear(self.in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Optional normalization
        self.norm1 = self._add_norm_layer(hidden_size, norm) if norm else nn.Identity()
        self.norm2 = self._add_norm_layer(hidden_size, norm) if norm else nn.Identity()

        # Adaptability component (weights that adapt based on input magnitude)
        self.adapt_weights = nn.Parameter(torch.ones(hidden_size))

    def _add_norm_layer(self, size, norm_type):
        if norm_type == 'batch':
            return nn.BatchNorm1d(size)
        elif norm_type == 'instance':
            return nn.InstanceNorm1d(size)
        else:
            return nn.Identity()  # No normalization

    def forward(self, x):
        x = x.view(-1, self.in_size)  # Flatten the data
        x = self.fc1(x)
        x = self.activation_function(x)
        x = self.norm1(x)

        # Adapt the weights based on the input magnitude (simplified concept)
        adapt_factor = x.abs().mean(dim=1, keepdim=True)
        adapted_weights = self.adapt_weights * adapt_factor

        x = self.fc2(x * adapted_weights)
        x = self.activation_function(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, output_size=1,dropout_prob=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.dropout = nn.Dropout(dropout_prob)
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size[1], hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.leakyrelu=nn.ReLU()
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        # Initialize hidden and cell states
        device = x.device  # Get the device of the input tensor
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)  # *2 for bidirectional

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(self.leakyrelu(self.dropout(out[:, -1, :])))

        return out