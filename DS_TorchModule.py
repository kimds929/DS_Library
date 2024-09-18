import torch

####################################################################################################################################
# (Python) Transformer Code 230328
# ★★★ Transformer
class Transformer(torch.nn.Module):
    def __init__(self, vocab_size_X, vocab_size_y, X_pad_idx=0, y_pad_idx=0,
                 embed_dim=256, n_layers=1, dropout=0.1, n_heads=4, posff_dim=512, pos_encoding='sinusoid'):
        super().__init__()
        self.X_pad_idx = X_pad_idx
        self.y_pad_idx = y_pad_idx
        
        self.encoder = Encoder(vocab_size_X, embed_dim, n_layers, n_heads, posff_dim, dropout, pos_encoding=pos_encoding)
        self.decoder = Decoder(vocab_size_y, embed_dim, n_layers, n_heads, posff_dim, dropout, pos_encoding=pos_encoding)
        self.fc_layer = torch.nn.Linear(embed_dim, vocab_size_y)
    
    def forward(self, X, y):
        # X : (batch_seq, X_word)
        # y : (batch_seq, y_word)
        
        if y is not None:
            with torch.no_grad():
                self.y_shape = y.shape
                self.init = y[0,0].to('cpu').detach() # 학습시 초기값 저장

        # mask
        self.X_mask = make_mask(X, self.X_pad_idx).unsqueeze(1).unsqueeze(1) if self.X_pad_idx is not None else None
        self.y_mask = make_tril_mask(y, self.y_pad_idx).unsqueeze(1) if self.y_pad_idx is not None else None
        
        # encoder
        self.encoder_output = self.encoder(X, self.X_mask)
        # decoder
        self.decoder_output = self.decoder(y, self.encoder_output, self.X_mask, self.y_mask)
        
        # # fully connected layer 
        self.output = self.fc_layer(self.decoder_output)
        
        # attention_score
        with torch.no_grad():
            # self.attention_scores = [layer.attention_score for layer_name, layer in self.decoder.decoder_layers.named_children()]
            self.attention_score = self.decoder.decoder_layers[-1].attention_score
        
        return self.output

    def predict(self, X, max_len=50, eos_word=None):
        # X : (batch_seq, X_word)
        with torch.no_grad():
            X_mask = make_mask(X, self.X_pad_idx).unsqueeze(1).unsqueeze(1)
            encoder_output = self.encoder(X, X_mask)

            output = torch.LongTensor([self.init]).repeat(X.shape[0],1).to(X.device)

            for _ in range(max_len-1):
                y_mask = make_tril_mask(output, self.y_pad_idx).unsqueeze(1)

                decoder_output = self.decoder(output, encoder_output, X_mask, y_mask)
                predict_output = self.fc_layer(decoder_output)

                # 출력 문장에서 가장 마지막 단어만 사용
                pred_word = predict_output.argmax(2)[:,[-1]]
                output = torch.cat([output, pred_word], axis=1)

        return output



# ★ masking function ------------------------------------------------------------
# mask
def make_mask(x, pad_idx=0):
    # x : (batch_seq, x_word)
    mask = (x != pad_idx).to(x.device)    # (batch_seq, X_word)
    return mask   # (batch_seq, x_word)

# tril_mask
def make_tril_mask(x, pad_idx=0):
    # x : (batch_seq, x_word)
    pad_mask = (x != pad_idx).unsqueeze(1).to(x.device)     # (batch_seq, 1, x_word)
    
    tril_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]))).bool().to(x.device)  # (batch_seq, batch_seq)
    
    mask = (pad_mask & tril_mask)    # (batch_seq, x_word, x_word)
    # (diagonal 이용하여) batch_seq에 따라 순차적  mask적용 
    
    return mask   # (batch_seq, x_word, x_word)



# ★★ Encoder
class Encoder(torch.nn.Module):
    def __init__(self, vocab_size_X, embed_dim=256, n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, pos_encoding=None):
        super().__init__()
        self.embed_layer = EmbeddingLayer(vocab_size_X, embed_dim)
        self.posembed_layer = PositionalEncodingLayer(encoding=pos_encoding)
        self.dropout = torch.nn.Dropout(dropout)

        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])

    def forward(self, X, X_mask=None):
        # X : (batch_Seq, X_word)
        
        # embedding layer
        self.X_embed = self.embed_layer(X)  # (batch_seq, X_word, emb)
        
        # positional encoding
        self.X_posembed = self.posembed_layer(self.X_embed).unsqueeze(0).repeat(X.shape[0], 1, 1)     # (batch_seq, X_word, emb)
        
        if X_mask is not None:
            mask = X_mask.squeeze().unsqueeze(-1).repeat(1, 1, self.X_posembed.shape[-1])
            self.X_posembed.masked_fill_(mask==0, 0)
        
        # sum of X_emb_scaled and pos_emb_X
        self.X_input = self.dropout(self.X_embed + self.X_posembed)     # (batch_seq, X_word, emb)

        # encoder layer
        next_input = self.X_input
        
        for enc_layer in self.encoder_layers:
            next_input = enc_layer(next_input, X_mask)
        self.encoder_output = next_input

        return self.encoder_output  # (batch_seq, X_word, emb)


# ★ EmbeddingLayer
class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embed_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.scaled = embed_dim ** (1/2)
        
        # attribute
        self.weight = self.embed_layer.weight
    
    def forward(self, X):
        self.emb_scaled = self.embed_layer(X) * self.scaled   # (batch_seq, X_word, emb)

        return self.emb_scaled



# ★ PositionalEncodingLayer
# https://velog.io/@sjinu/Transformer-in-Pytorch#3-positional-encoding
# functional : postional_encoding
def positional_encoding(x, encoding=None):
    """
     encoding : None, 'sinusoid'
    """
    if x.ndim == 2:
        batch_size, seq_len = x.shape
        
        if encoding is None:
            pos_encode = torch.arange(seq_len).requires_grad_(False).to(x.device)
        elif 'sin' in encoding:
            pos_encode = torch.sin(torch.arange(seq_len)).requires_grad_(False).to(x.device)
            
    elif x.ndim == 3:
        batch_size, seq_len, embed_dim = x.shape
        
        if encoding is None:
            pos_encode = torch.arange(seq_len).unsqueeze(-1).repeat(1, embed_dim).requires_grad_(False).to(x.device)
        elif 'sin' in encoding:
            pos_encode = torch.zeros(seq_len, embed_dim).requires_grad_(False).to(x.device)         # (seq_len, emb)
            pos_inform = torch.arange(0, seq_len).unsqueeze(1) # (seq_len, 1)
            index_2i = torch.arange(0, embed_dim, step=2)       # (emb)
            pos_encode[:, ::2] = torch.sin(pos_inform/(10000**(index_2i/embed_dim)))       # (seq_len, emb)

            if embed_dim % 2 == 0:
                pos_encode[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i/embed_dim)))
            else:
                pos_encode[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i[:-1]/embed_dim)))
    return pos_encode

# Class : PositionalEncodingLayer
class PositionalEncodingLayer(torch.nn.Module):
    def __init__(self, encoding=None):
        super().__init__()
        self.encoding = encoding
        
    def forward(self, x):
        return positional_encoding(x, self.encoding)


# class LinearPositionalEncodingLayer(torch.nn.Module):
#     def __init__(self, max_len=300, embed_dim=256):
#         super().__init__()
#         pos_embed = torch.arange(0, max_len).unsqueeze(1).repeat(1,embed_dim).requires_grad_(False)

#         # self.pos_embed = pos_embed    # (max_len, emb)
#         self.register_buffer('pos_embed', pos_embed)      # 학습되지 않는 변수로 등록

#     def forward(self, x):
#         # x : (batch_Seq, x_word, emb)
#         self.pos_embed = self.pos_embed[:x.shape[1]]
#         self.pos_embed_output = torch.autograd.Variable(self.pos_embed, requires_grad=False).to(x.device)
#         return self.pos_embed_output       # (x_word, emb)

# class PositionalEncodingLayer(torch.nn.Module):
#     def __init__(self, max_len=300, embed_dim=256):
#         super().__init__()

#         pos_embed = torch.zeros(max_len, embed_dim).requires_grad_(False)         # (max_len, emb)
#         pos_inform = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
#         index_2i = torch.arange(0, embed_dim, step=2)       # (emb)
#         pos_embed[:, ::2] = torch.sin(pos_inform/(10000**(index_2i/embed_dim)))       # (max_len, emb)

#         if embed_dim % 2 == 0:
#             pos_embed[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i/embed_dim)))
#         else:
#             pos_embed[:, 1::2] = torch.cos(pos_inform/(10000**(index_2i[:-1]/embed_dim)))

#         # self.pos_embed = pos_embed    # (max_len, emb)
#         self.register_buffer('pos_embed', pos_embed)      # 학습되지 않는 변수로 등록

#     def forward(self, x):
#         # x : (batch_Seq, x_word, emb)
#         self.pos_embed = self.pos_embed[:x.shape[1]]
#         self.pos_embed_output = torch.autograd.Variable(self.pos_embed, requires_grad=False).to(x.device)
#         return self.pos_embed_output       # (x_word, emb)







# ★ Encoder_Layer
class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, posff_dim=512, dropout=0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.self_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.self_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.posff_layer = PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout)
        self.posff_layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, X_emb, X_mask=None):
        # X_emb : (batch_seq, X_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)
        
        # (Self Attention Layer) ------------------------------------------------------------------
        self.X_self_att_output  = self.self_att_layer((X_emb, X_emb, X_emb), mask=X_mask)
        self.self_attention_score = self.self_att_layer.attention_score
        
        #  (batch_seq, X_word, fc_dim=emb), (batch_seq, n_heads, X_word, key_length=X_word)
        self.X_skipconnect_1 = X_emb + self.dropout(self.X_self_att_output)   # (batch_seq, X_word, emb)
        # embeding+pos_input 값을 self_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.X_layer_normed_1 = self.self_att_layer_norm(self.X_skipconnect_1)  # layer normalization
        
        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.X_posff = self.posff_layer(self.X_layer_normed_1)    # (batch_seq, X_word, emb)
        self.X_skipconnect_2 = self.X_layer_normed_1 + self.dropout(self.X_posff)     # (batch_seq, X_word, emb)
        # layer_norm_X와 positional_feedforward를 통과한 결과를 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.X_layer_normed_2 = self.posff_layer_norm(self.X_skipconnect_2)

        return self.X_layer_normed_2   # (batch_seq, X_word, emb), (batch_seq, n_heads, X_word, key_length)


# ☆ MultiHeadAttentionLayer
class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, dropout=0, same_qkv=False):
        super().__init__()
        assert embed_dim % n_heads == 0, 'embed_dim은 n_head의 배수값 이어야만 합니다.'

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

        self.att_layer = ScaledDotProductAttention(embed_dim ** (1/2), dropout)
        self.fc_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.same_qkv = same_qkv

    def forward(self, x, mask=None):
        if self.same_qkv:
            query, key, value = (x,x,x)
        else:
            query, key, value = x
        # query, key, value : (batch_seq, len, emb)
        with torch.no_grad():
            batch_size = query.shape[0]

        self.query = self.query_layer(query)    # (batch_seq, query_len, emb)
        self.key   = self.key_layer(key)        # (batch_seq, key_len, emb)
        self.value = self.value_layer(value)    # (batch_seq, value_len, emb)

        self.query_multihead = self.query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, query_len, head_emb_dim)   ←permute←  (batch_seq, query_len, n_heads, head_emb_dim)
        self.key_multihead   =   self.key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, key_len, head_emb_dim)   ←permute←  (batch_seq, key_len, n_heads, head_emb_dim)
        self.value_multihead = self.value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)     # (batch_seq, n_heads, value_len, head_emb_dim)   ←permute←  (batch_seq, value_len, n_heads, head_emb_dim)

        self.weighted, self.attention_score = self.att_layer((self.query_multihead, self.key_multihead, self.value_multihead), mask=mask)
        # self.weightd          # (B, H, QL, HE)
        # self.attention_score  # (B, H, QL, QL)     ★

        self.weighted_arange = self.weighted.permute(0,2,1,3).contiguous()        # (B, QL, H, HE) ← (B, H, QL, HE)
        self.weighted_flatten = self.weighted_arange.view(batch_size, -1, self.embed_dim)   # (B, QL, E) ← (B, H, E)

        # self.multihead_output = self.fc_layer(self.weighted_flatten)       # (B, QL, FC)
        self.multihead_output = self.fc_layer(self.weighted_flatten).view(query.size())       # (B, QL, FC) to input shape
        return self.multihead_output       #  (batch_seq, query_length, fc_dim)


# * ScaledDotProductAttention
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, scaled=1, dropout=0):
        super().__init__()
        self.scaled = scaled
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None, epsilon=1e-10):
        query, key, value = x

        self.energy = torch.matmul(query, key.transpose(-1,-2)) / self.scaled    # (B, ..., S, S) ← (B, ..., S, W), (B, ..., W, S)
        # * summation of muliply between embedding vectors : Query에 해당하는 각 Length(단어) embedding이 어떤 key의 Length(단어) embedding과 연관(내적)되는지?

        if mask is not None:
            # masking 영역(==0)에 대해 -epsilon 으로 채우기 (softmax → 0)
            self.energy = self.energy.masked_fill(mask==0, -epsilon)

        self.attention_score = torch.softmax(self.energy, dim=-1)

        self.weighted = torch.matmul(self.dropout_layer(self.attention_score), value)    # (B, ..., S, W) ← (B, ..., S, S), (B, ..., S, W)
        # * summation of muliply between softmax_score and embeding of value

        return self.weighted, self.attention_score


# ☆ Positionalwise_FeedForward_Layer
class PositionwiseFeedForwardLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, posff_dim=512, dropout=0, activation='ReLU'):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.fc_layer_1 = torch.nn.Linear(embed_dim, posff_dim)
        self.activation =  eval(f"torch.nn.{activation}()") if type(activation) == str else (activation() if isinstance(activation, type) else None)
        self.fc_layer_2 = torch.nn.Linear(posff_dim, embed_dim)
        
    def forward(self, X):
        # X : (batch_seq, X_word, emb)
        self.ff_output_1 = self.dropout(self.activation(self.fc_layer_1(X)))    # (batch_seq, X_word, posff_dim)
        self.ff_output_2 = self.fc_layer_2(self.ff_output_1)    # (batch_seq, X_word, emb)
        return self.ff_output_2  # (batch_seq, X_word, emb)


# ★★ Decoder 
class Decoder(torch.nn.Module):
    def __init__(self, vocab_size_y, embed_dim=256, n_layers=1, n_heads=4, posff_dim=512, dropout=0.1, pos_encoding=None):
        super().__init__()
        
        self.embed_layer = EmbeddingLayer(vocab_size_y, embed_dim)
        self.posembed_layer = PositionalEncodingLayer(encoding=pos_encoding)
        self.dropout = torch.nn.Dropout(dropout)

        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(embed_dim, n_heads, posff_dim, dropout) for _ in range(n_layers)])
        
    def forward(self, y, context_matrix, X_mask=None, y_mask=None):
        # y : (batch_seq, y_word)
        # X_mask : (batch_seq, 1, ,1, X_word)
        # y_mask : (batch_seq, 1, y_word, y_word)
        # context_matrix : (batch_seq, X_word, emb)
        

        # embedding layer
        self.y_embed = self.embed_layer(y)  # (batch_seq, y_word, emb)
        
        # positional encoding
        self.y_posembed = self.posembed_layer(self.y_embed).unsqueeze(0).repeat(y.shape[0], 1, 1)     # (batch_seq, y_word, emb)
        
        # if y_mask is not None:
            # mask = y_mask.squeeze().unsqueeze(-1).repeat(1, 1, self.y_posembed.shape[-1])
            # self.y_posembed.masked_fill_(mask==0, 0)
            
        # sum of X_emb_scaled and pos_emb_X
        self.y_input = self.dropout(self.y_embed + self.y_posembed)     # (batch_seq, y_word, emb)
        
        # decoder layer
        next_input = self.y_input
        
        for dec_layer in self.decoder_layers:
            next_input = dec_layer(next_input, context_matrix, X_mask, y_mask)
        self.decoder_output = next_input
        
        return self.decoder_output

# ★ Decoder_Layer
class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, posff_dim=512, dropout=0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        
        self.self_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.self_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.enc_att_layer = MultiHeadAttentionLayer(embed_dim, n_heads, dropout)
        self.enc_att_layer_norm = torch.nn.LayerNorm(embed_dim)
        
        self.posff_layer = PositionwiseFeedForwardLayer(embed_dim, posff_dim, dropout)
        self.posff_layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, y_emb, context_matrix, X_mask=None, y_mask=None):
        # y_emb : (batch_seq, y_word, emb)
        # X_mask : (batch_seq, 1, ,1, X_word)
        # y_mask : (batch_seq, 1, y_word, y_word)
        # context_matrix : (batch_seq, X_word, emb)     # encoder output
        
        # (Self Attention Layer) -------------------------------------------------------------------
        self.y_self_att_output = self.self_att_layer((y_emb, y_emb, y_emb), mask=y_mask)
        #  (batch_seq, y_word, fc_dim=emb)        
        self.self_attention_score = self.self_att_layer.attention_score
        # (batch_seq, n_heads, y_word, key_length=y_word)
        
        self.y_skipconnect_1 = y_emb + self.dropout(self.y_self_att_output)   # (batch_seq, y_word, emb)
        # embeding+pos_input 값을 self_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_1 = self.self_att_layer_norm(self.y_skipconnect_1)  # layer normalization
        
        # (Encoder Attention Layer) ----------------------------------------------------------------
        self.y_enc_att_output = self.enc_att_layer((self.y_layer_normed_1, context_matrix, context_matrix), mask=X_mask)
        #  (batch_seq, y_word, fc_dim=emb)
        self.attention_score = self.enc_att_layer.attention_score
        # (batch_seq, n_heads, y_word, key_length=y_word)
        
        self.y_skipconnect_2 = self.y_layer_normed_1 + self.dropout(self.y_enc_att_output)   # (batch_seq, y_word, emb)
        # embeding+pos_input 값을 encoder_attention 결과와 더해준다.
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_2 = self.enc_att_layer_norm(self.y_skipconnect_2)  # layer normalization

        # (Positional FeedForward Layer) -----------------------------------------------------------
        self.y_posff = self.posff_layer(self.y_layer_normed_2)    # (batch_seq, y_word, emb)
        
        # (Layer Normalization) --------------------------------------------------------------------
        self.y_layer_normed_3 = self.posff_layer_norm(self.y_posff)
        # layer_norm_X와 positional_feedforward를 통과한 결과를 더해준다.
        
        return self.y_layer_normed_3    # (batch_seq, y_word, emb)
#######################################################################################################################################


# x = torch.tensor(randint(3,15,high=6, sign='+'))
# y = torch.tensor(randint(3,10,high=2, sign='+'))

# x = (torch.rand(3,15)*6).type(torch.long)
# y = (torch.rand(3,10)*2).type(torch.long)

# tr = Transformer(7, 5, 0, 0, n_layers=3, pos_encoding=None)
# tr = Transformer(7, 5, 0, 0, pos_encoding='sin')
# tr(x,y).shape
# tr.predict(x).shape
# tr.attention_score.shape
 