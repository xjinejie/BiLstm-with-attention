import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意力机制模块
class Attention(nn.Module): # 注意力机制
    def __init__(self, hidden_dim): # 定义注意力模块需要的层
        super(Attention, self).__init__()
        # 这里的hidden_dim是Bi-LSTM输出的维度 (2 * hidden_size)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states):# 前向计算注意力权重并生成上下文向量
        # hidden_states: [batch, seq_len, hidden_dim]
        
        # energy: [batch, seq_len, hidden_dim]
        energy = torch.tanh(self.attn(hidden_states))
        
        # attention_scores: [batch, seq_len, 1]
        attention_scores = self.v(energy)
        
        # weights: [batch, seq_len, 1]
        weights = F.softmax(attention_scores, dim=1)
        
        # context_vector: [batch, hidden_dim]
        # sum(weights * hidden_states) across seq_len
        context_vector = torch.sum(weights * hidden_states, dim=1)
        
        return context_vector, weights
    

class BiLSTMAttnModel(nn.Module): # Bi-LSTM + Attention 模型
    def __init__(self, vocab_size, config):
        super(BiLSTMAttnModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=0)
        
        self.dropout = nn.Dropout(config.dropout)
        self.use_residual = config.use_residual
        self.use_layer_norm = config.use_layer_norm
        
        # 手动定义多层LSTM以支持残差连接
        self.lstm_layers = nn.ModuleList()
        input_dim = config.embed_dim
        
        for _ in range(config.n_layers):
            self.lstm_layers.append(
                nn.LSTM(input_dim, config.hidden_dim, batch_first=True, bidirectional=True)
            )
            # 下一层的输入是上一层的输出 (双向: 2 * hidden)
            input_dim = config.hidden_dim * 2
            
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim * 2) for _ in range(config.n_layers)])

        self.attention = Attention(config.hidden_dim * 2)
        
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)
        
        # 残差投影：如果输入维度 != 输出维度，需要投影
        self.residual_projections = nn.ModuleList()
        if self.use_residual:
            curr_in = config.embed_dim
            for _ in range(config.n_layers):
                curr_out = config.hidden_dim * 2
                if curr_in != curr_out:
                    self.residual_projections.append(nn.Linear(curr_in, curr_out))
                else:
                    self.residual_projections.append(nn.Identity())
                curr_in = curr_out

    def forward(self, x):
        # x: [batch, seq_len]
        out = self.embedding(x) # [batch, seq_len, embed_dim]
        out = self.dropout(out)
        
        for i, lstm in enumerate(self.lstm_layers):
            prev_out = out
            lstm_out, _ = lstm(out) # [batch, seq_len, hidden*2]
            
            # 这里的残差连接应用在LSTM层之间
            if self.use_residual:
                # 调整prev_out维度以匹配lstm_out
                res_in = self.residual_projections[i](prev_out)
                out = lstm_out + res_in
            else:
                out = lstm_out
            
            if self.use_layer_norm:
                out = self.layer_norms[i](out)
                
            out = self.dropout(out)
            
        # Attention
        context_vector, _ = self.attention(out) # [batch, hidden*2]
        
        logits = self.fc(context_vector)
        return logits