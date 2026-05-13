from typing import Optional, Tuple
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


# (masking function) #################################################################################################################
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
####################################################################################################################################


# a = torch.rand(10,5)
# a[a < 0.3] = 0
# make_mask(a)      # (10, 5)
# make_tril_mask(a) # (10, 5, 5)





# (Container Block)  ##################################################################################################
def module_helper_for_kwarg(module, *args, **kwargs):
    """
    module.forward의 시그니처를 보고, 사용할 수 있는 kwargs만 골라서 호출
    사용 예: y = call_with_filtered_kwargs(layer, x, mask=mask, time_mask=time_mask)
    """
    sig = inspect.signature(module.forward)
    param_names = list(sig.parameters.keys())[1:]  # self 제외

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
    return module(*args, **filtered_kwargs)

class KwargSequential(nn.Sequential):
    """
    A Sequential container that transparently forwards additional keyword arguments
    (e.g., `mask`, `time_mask`, `scale`, etc.) to submodules **only if** they accept them.

    일반 nn.Sequential과 달리, MultiInputSequential은 forward(x, **kwargs)로 입력받으며,
    kwargs 내부 인자 이름이 각 submodule.forward()의 signature에 존재하는 경우에만
    해당 인자를 전달합니다.

    이를 통해 "mask-aware" 모듈(MaskedConv1d, ResidualConnection 등)과
    일반 모듈(nn.ReLU, nn.BatchNorm1d 등)을 하나의 pipeline 안에서 자연스럽게 혼합하여
    사용할 수 있습니다.

    Example:
        seq = KwargSequential(
            MaskedConv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            ResidualConnection(
                KwargSequential(
                    MaskedConv1d(32, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                )
            ),
        )

        out = seq(x, mask=mask)  # mask는 mask-aware 모듈로만 전달됨

    Args:
        x (Tensor): 주 입력 텐서. Conv1d라면 (B, C, T).
        **kwargs: 추가 keyword 인자. 필요한 submodule에서만 선택적으로 사용됨.

    Returns:
        Tensor: 마지막 모듈의 출력.
    """
    def forward(self, x, **kwargs):
        """
        x : 메인 입력 (Conv1d feature 등)
        kwargs : mask, time_mask, scale 등 부가 인자들
        """
        for module in self:
            # module.forward의 시그니처를 보고, 받을 수 있는 인자만 필터링
            sig = inspect.signature(module.forward)
            param_names = list(sig.parameters.keys())[1:]  # self 제외

            filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}

            # 해당 모듈이 그 인자를 받는다면 kwargs 전달
            if filtered_kwargs:
                x = module(x, **filtered_kwargs)
            else:
                x = module(x)
        return x



 ####################################################################################################################################




# (Layer for ResidualConnection)  ##################################################################################################
class ResidualConnection(nn.Module):
    """
    A wrapper module that applies a residual skip connection around a given block.

    block(x, **filtered_kwargs) + shortcut(x)
    형태의 출력을 반환하며, block.forward()가 받을 수 없는 인자는 자동으로 필터링됩니다.

    MultiInputSequential과 함께 사용하면, block이 mask-aware인지 여부와 관계없이
    동일한 구조로 residual connection을 구성할 수 있습니다.

    Args:
        block (nn.Module): Residual main branch에 해당하는 모듈.
        shortcut (Callable, optional): Skip-connection. 기본은 identity(x).

    Example:
        block = MultiInputSequential(
            MaskedConv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        res = ResidualConnection(block)
        out = res(x, mask=mask)
    """
    def __init__(self, block, shortcut=None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut or (lambda x: x)
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)
 ####################################################################################################################################






# (Layer for Embedding)  ##################################################################################################

# class CategoricalEmbedding(nn.Module):
#     def __init__(self, n_features, num_embeddings, embedding_dim):
#         super().__init__()
#         self.embedding_weights = nn.Parameter(torch.randn((n_features, num_embeddings, embedding_dim)))
    
#     def forward(self, x):
#         x_shape = x.shape
#         n_features = x_shape[-1]
        
#         feature_idx = torch.arange(n_features).view(*([1] * (x.ndim - 1)), n_features).expand(*x_shape)
#         return self.embedding_weights[feature_idx, x]

class CategoricalEmbedding(nn.Module):
    def __init__(self, n_features, num_embeddings, embedding_dim):
        super().__init__()
        self.nf, self.ne, self.ed = n_features, num_embeddings, embedding_dim
        self.embedding = nn.Embedding(n_features * num_embeddings, embedding_dim)

    def forward(self, x):
        # x: (..., n_features), long
        *batch, F = x.shape
        device = x.device
        feature_idx = torch.arange(F, device=device).view(*([1]*len(batch)), F).expand(*x.shape)
        flat_idx = feature_idx * self.ne + x                        # (..., F)
        out = self.embedding(flat_idx)                                      # (..., F, ed)
        return out





# Feature 마다 독립적으로 Embedding을 부여 : input feature → feature × embedding으로 linear하게 mapping
#   . Embedding 이후 Feature간 connection이 없이 embedding 부여 가능
class ContinuousEmbedding(nn.Module):
    """
    ContinuousEmbeddingLayer
    ------------------------
    연속형 입력 feature를 embedding 공간으로 확장 및 사상하는 Layer.
    각 feature 값에 대해 (feature_dim → feature_dim × embed_dim) 형태의 임베딩 벡터를 생성함.
    """
    
    def __init__(self, feature_dim, embed_dim, bias=True, sine=False, independent=False, expand=True):
        """
        Args:
            feature_dim (int):
                입력 feature의 차원 수. 
                예: 입력 x의 shape이 (batch_size, feature_dim) 일 때, 그 feature_dim에 해당.

            embed_dim (int):
                각 feature를 임베딩할 차원 수. 즉, 출력 텐서의 마지막 차원 크기.

            bias (bool, default=True):
                True일 경우, 각 feature-embedding에 bias term(weight_bias)을 추가함.
                False일 경우, 순수 weight * x 형태로 계산.

            sine (bool, default=False):
                True일 경우, embedding 결과에 torch.sin() 함수를 적용하여
                주기적 표현(예: positional encoding 스타일)을 생성함.

            independent (bool, default=False):
                True → 각 feature가 입력값 x와 무관하게 독립적인 embedding 값을 가짐.
                False → 입력값 x에 따라 embedding이 선형적으로 변함 (weight * x 형태).

            expand (bool, default=True):
                embedding weight를 batch 차원에 맞게 확장(expand)하여 broadcasting 적용.
                False → 확장하지 않고 (feature_dim, embed_dim) 형태 그대로 반환.
                * independent=True일 때만 작동
        """
        super().__init__()
        self.bias = bias
        self.sine = sine
        self.independent = independent
        self.expand = expand
        self.embed_dim = embed_dim

        if independent is False:
            self.weight = nn.Parameter(torch.randn(feature_dim, embed_dim))  # (feature_dim, embed_dim, 1)
            nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

        if (independent is True) or (bias is True):
            self.weight_bias = nn.Parameter(torch.randn(feature_dim, embed_dim))       # (feature_dim, embed_dim)
            nn.init.uniform_(self.weight_bias, -1, 1)

    def forward(self, x):
        *x_shape, f_dim = x.shape   # (batch_dim, feature_dim)
        x_unsqueezed = x.unsqueeze(-1)
        if self.independent:
            x_embed = self.weight_bias     # (feature_dim, embed_dim)
            if self.expand:
                x_embed = x_embed.expand(*x_shape, f_dim, self.embed_dim)
        else:
            x_embed = self.weight * x_unsqueezed
            if self.bias:
                x_embed += self.weight_bias
        
        if self.sine:
            x_embed = torch.sin(x_embed) 

        return x_embed

# ce = ContinuousEmbedding(feature_dim=5, embed_dim=2) 
# ce( torch.rand(10,5) ).shape   # (10,5,2).shae
 

# Embedding단위에서 independent하게 fc layer를 통과
class EmbeddingLinear(nn.Module):
    """
    EmbeddingLinear
    ----------------
    각 feature dimension마다 독립적인 선형 변환(Linear layer)을 수행하는 embedding 모듈.
    즉, 각 feature 단위로 (feature_dim × in_embedding → feature_dim × out_embedding) 변환을 수행하며,
    embedding_dimension 에서의 사상은 feature마다 독립적으로 이루어진다.

    예를 들어,
    - 각 feature별로 embedding projection layer를 따로 적용하고자 할 때 사용.
    - categorical embedding 또는 continuous embedding 이후 feature-wise transformation으로 활용 가능.
    """
    def __init__(self, feature_dim, in_embedding, out_embedding, bias:bool=True):
        """
        Args:
            feature_dim (int):
                feature의 개수 (각 feature마다 별도의 linear transform을 학습).

            in_embedding (int):
                각 feature의 입력 임베딩 차원 (input embedding size).

            out_embedding (int):
                각 feature의 출력 임베딩 차원 (output embedding size).

            bias (bool, default=True):
                각 feature별로 bias term을 추가할지 여부.
                True이면 (feature_dim, out_embedding) 형태의 bias 파라미터를 학습.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.in_embedding = in_embedding
        self.out_embedding = out_embedding
    
        self.weight = nn.Parameter(torch.empty(feature_dim, in_embedding, out_embedding))
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)).item())
        
        if bias:
            self.bias = nn.Parameter(torch.empty(feature_dim, out_embedding))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(float(fan_in))).item() if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        embed_out = torch.einsum('bfe,fek->bfk', x, self.weight)
        if self.bias is not None:
            embed_out += self.bias
        return embed_out

    def __repr__(self):
        return f"EmbeddingLayer(feature_dim={self.feature_dim}, in_embedding={self.in_embedding}, out_embedding={self.out_embedding})"
# el = EmbeddingLinear(5,3,6)
# el( torch.rand(10,5,3) ).shape  # (10,5,6)


class ContinuousEmbeddingBlock(nn.Module):
    """
    ContinuousEmbeddingBlock
    ------------------------
    연속형 입력 feature를 여러 embedding 표현(선형형 + 사인형)으로 변환한 뒤,
    필요에 따라 feature 간 결합(linear mixing) 및 flatten까지 수행하는 블록 모듈.

    주요 구성:
        - 선형 임베딩 (independent embedding)
        - 사인 임베딩 (periodic embedding)
        - 입력값 자체의 self-concatenation (선택적)
        - 임베딩 차원 축소 또는 혼합 (EmbeddingLinear)
    """
    def __init__(self, input_dim, embed_dim=0, n_linear=1, n_sine=1, self_concat=True, flatten=False):
        """
        Args:
            input_dim (int):
                입력 feature의 차원 수. (feature_dim)
                예: 입력 x.shape = (batch_size, input_dim)

            embed_dim (int, default=0):
                최종 출력 embedding 차원.
                0보다 크면 내부적으로 EmbeddingLinear를 사용해 (n_linear + n_sine [+1]) → embed_dim 변환 수행.
                0이면 변환 없이 concat된 embedding 그대로 출력.

            n_linear (int, default=1):
                연속형 feature마다 independent embedding의 개수.
                (즉, feature별 독립적인 선형 embedding channel 수)

            n_sine (int, default=1):
                연속형 feature마다 sine embedding의 개수.
                주기적 특성을 가진 임베딩 channel 수.

            self_concat (bool, default=True):
                True → 원래 입력값 x를 임베딩 벡터 앞에 concat. (예: [linear_embed, sine_embed])
                False → 임베딩 결과만 사용. (예: [x, linear_embed, sine_embed])
                

            flatten (bool, default=False):
                True → feature_dim과 embed_dim을 flatten하여 (batch_size, -1) 형태로 변환.
                False → (batch_size, input_dim, embed_dim) 형태 유지.
        """
        super().__init__()
        self.flatten = flatten
        self.self_concat = self_concat
        

        self.ind_embedding = ContinuousEmbedding(input_dim, n_linear, independent=True)
        self.sin_embedding = ContinuousEmbedding(input_dim, n_sine, sine=True)

        self.embed_dim = n_linear + n_sine
        if self_concat:
            self.embed_dim += 1
        
        self.embedding_mixture = False
        if embed_dim > 0:
            self.embedding_linear = EmbeddingLinear(input_dim, self.embed_dim, embed_dim)
            self.embed_dim = embed_dim
            self.embedding_mixture = True
        
    def forward(self, x):
        x_shape = x.shape
        ind_x = self.ind_embedding(x)
        sin_x = self.sin_embedding(x)
        
        concat_tensors = [ind_x, sin_x]
        if self.self_concat:
            concat_tensors.insert(0, x.unsqueeze(-1))

        embed_output = torch.cat(concat_tensors, dim=-1)
        
        if self.embedding_mixture:
            embed_output = F.relu(embed_output)
            embed_output = self.embedding_linear(embed_output)
            
        if self.flatten:
            embed_output = embed_output.reshape(x_shape[0],-1)
        return embed_output

# eb = ContinuousEmbeddingBlock(input_dim=4)
# eb(torch.rand(5,4)).shape   # (5,4,3)

# eb = ContinuousEmbeddingBlock(input_dim=4, self_concat=False)
# eb(torch.rand(5,4)).shape   # (5,4,2)

# eb = ContinuousEmbeddingBlock(input_dim=4, embed_dim=10)
# eb(torch.rand(5,4)).shape   # (5,4,10)

# eb = ContinuousEmbeddingBlock(input_dim=4, embed_dim=10, flatten=True)
# eb(torch.rand(5,4)).shape   # (5,40)


 
 ####################################################################################################################################







# (Layer for PositionalEncoding)  ##################################################################################################
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding
    ------------------
    Transformer 모델에서 입력 토큰의 순서 정보를 인코딩하기 위한 **고정형(Non-learnable) 사인/코사인 기반 Positional Encoding** 모듈.

    원리:
        각 위치 pos와 차원 i에 대해 다음과 같이 정의됩니다:
            PE(pos, 2i)   = sin(pos / (10000^(2i / d_model)))
            PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))

    특징:
        - 학습되지 않는 deterministic positional encoding.
        - 입력 시퀀스에 위치 정보를 더해 Transformer의 순서 의존성을 부여함.
        - 논문 "Attention Is All You Need" (Vaswani et al., 2017)에서 제안된 방식.
    """
    def __init__(self, d_model, max_len=4096):
        """
        Args:
            d_model (int):
                모델의 hidden dimension 크기 (embedding 차원).
                예: Transformer 입력 벡터의 feature 수.

            max_len (int, default=4096):
                미리 계산할 최대 sequence 길이.
                입력 sequence 길이가 이보다 크면 오류 발생.

        Attributes:
            position_encoding (torch.Tensor):
                shape = (1, max_len, d_model)
                사인/코사인 값으로 미리 계산된 positional encoding 텐서.
        """
        super().__init__()
        position_encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        base_log = -torch.log(torch.tensor(10000.0)) / d_model
        div_even = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * base_log)  # len = ceil(d/2)
        div_odd  = torch.exp(torch.arange(1, d_model, 2, dtype=torch.float) * base_log)  # len = floor(d/2)

        position_encoding[:, 0::2] = torch.sin(pos * div_even)
        position_encoding[:, 1::2] = torch.cos(pos * div_odd)
        self.register_buffer('position_encoding', position_encoding.unsqueeze(0))  # (1, L, d_model)
        
    def forward(self, x):
        return x + self.position_encoding[..., :x.size(1)]

# pos_encoding = PositionalEncoding(5)
# pos_encoding(torch.rand(10,3,5)).shape  # (10, 3, 5)


class LearnablePositionalEncoding(nn.Module):
    """
    LearnablePositionalEncoding
    ---------------------------
    입력 시퀀스의 위치 정보를 학습 가능한 임베딩 벡터로 표현하는 Positional Encoding 모듈.

    원리:
        - 각 시퀀스 위치 pos에 대해 학습 가능한 embedding vector E[pos]를 부여함.
        - 모델이 task-specific positional pattern을 학습할 수 있도록 함.

    특징:
        - 학습 가능한 파라미터를 통해 positional pattern을 데이터 기반으로 최적화 가능.
        - 사인/코사인 기반 fixed encoding보다 유연하나, extrapolation 능력(미학습 길이에 대한 일반화)은 낮음.
    """
    def __init__(self, num_embedding, d_model):
        """
        Args:
        num_embedding (int):
            positional embedding을 학습할 최대 sequence 길이.
            
        d_model (int):
            모델의 hidden dimension 크기 (embedding 차원).

        Attributes:
            pos_embedding (nn.Embedding):
                shape = (max_len, d_model)
                각 위치 인덱스에 해당하는 학습 가능한 embedding 파라미터.
        """
        super().__init__()
        self.position_embedding = nn.Embedding(num_embedding, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_enc = self.position_embedding(positions)
        return x + pos_enc


# learn_pos_encoding = LearnablePositionalEncoding(5)
# learn_pos_encoding(torch.rand(10,3,5)).shape  # (10, 3, 5)

 ####################################################################################################################################









# (Layer for Normalization)  ##################################################################################################
class FeatureWiseEmbeddingNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape, 1))
        self.beta = nn.Parameter(torch.zeros(normalized_shape, 1))
    
    def forward(self, x):
        """
        x: (batch, feature, embedding_dim)
        feature별로 embedding_dim 축에 대해 normalization
        """
        mean = x.mean(dim=-1, keepdim=True)  # (B, F, 1)
        var = x.var(dim=-1, keepdim=True)    # (B, F, 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

 ####################################################################################################################################













# (Layer for Attention & MultiheadAttention)  ##################################################################################################

class ScaledDotProductAttention(nn.Module):
    """
    ScaledDotProductAttention
    -------------------------
    Transformer 및 Attention 계열 모델에서 사용하는 기본 어텐션 메커니즘.
    쿼리(Q)와 키(K)의 내적(dot-product)을 통해 유사도를 계산하고,
    이를 소프트맥스 확률로 정규화하여 값(V)에 대한 가중합(weighted sum)을 구함.

    개념적으로,
        - q, k, v가 모두 같은 값일 때(Self-Attention):
            → 각 feature가 다른 feature 중 **어떤 feature에 더 집중해야 하는지**를 학습적으로 판단함.
            즉, 입력 내 feature 간 상호 의존 관계(feature-to-feature dependency)를 반영하는 메커니즘.
        - q ≠ k, v일 때(Cross-Attention):
            → 쿼리 집합이 키-값 집합에서 **참조해야 할 정보 위치**를 학습적으로 선택함.

    수식:
        Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V

    특징:
        - head_dim이 커질수록 분산이 커지므로 sqrt(d_k)로 스케일링하여 안정화.
        - softmax로 가중치 분포 생성 후, V의 weighted sum을 통해 정보 집약.
        - dropout으로 어텐션 가중치의 regularization 지원.
        - padding mask, causal mask 등 다양한 형태의 마스크 입력 지원.

    Example:
        >>> x = torch.randn(2, 4, 8)  # (batch=2, feature=4, embed_dim=8)
        >>> attn = ScaledDotProductAttention()
        >>> out, weights = attn(x, x, x)
        >>> out.shape, weights.shape
        (torch.Size([2, 4, 8]), torch.Size([2, 4, 4]))

        - weights[b, i, j]: batch b에서 i번째 feature가 j번째 feature에 얼마나 집중하는지(attention strength)
    """
    def __init__(self, dropout=0.0):
        """
        Args:
            dropout (float, default=0.0):
                어텐션 가중치(attn_weights)에 적용할 드롭아웃 비율.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (B, num_heads, T_q, head_dim)
        k: (B, num_heads, T_k, head_dim)
        v: (B, num_heads, T_k, head_dim)
        attn_mask: (B, num_heads, T_q, T_k) 또는 broadcast 가능 형태
        """
        d_k = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

# a = torch.rand(10,5,3,4)

# sda = ScaledDotProductAttention()
# a1,a2 = sda(a,a,a)
# a1.shape, a2.shape      # (10,5,3,4), (10,5,3,3)





class MultiheadAttention(nn.Module):
    """
    MultiHeadAttention
    -----------------------
    입력 시퀀스(또는 feature 집합)에서 **여러 개의 '관점(head)'로 병렬 어텐션을 수행**하여,
    각 토큰/feature가 어떤 다른 위치에 주목해야 하는지 학습적으로 판단하고, 그 결과를 집계해 내보내는 모듈.

    직관:
        - 한 개의 Head는 "하나의 시각"으로 유사도를 보고 정보를 모읍니다.
        - 여러 Head는 서로 다른 시각(서브스페이스)을 병렬로 보며, **다양한 관계**를 동시에 포착합니다.
        - q=k=v(같은 텐서)면 **Self-Attention**: 입력 내부의 의존성(토큰/feature 간 관계)을 학습.
        - q≠k,v(다른 텐서)면 **Cross-Attention**: 쿼리가 외부 메모리(키/값)에서 정보를 검색.

    처리 흐름(요약):
        1) 입력을 선형사상으로 Q, K, V를 만듦 (필요 시 kdim/vdim 별도 지원)
        2) (num_heads)로 나누어 head_dim 단위로 분할
        3) 각 Head에서 Scaled Dot-Product Attention 수행
        4) 모든 Head의 출력을 concat → 최종 선형사상(out_proj)

    마스크:
        - key_padding_mask: 패딩 토큰 위치를 가리는 **True/1=mask-out** 형태. shape ~ (B, T_k)
        - attn_mask      : causal/제한 마스크 등. shape ~ (B or 1, H or 1, T_q, T_k)로 broadcasting 가능
        - 두 마스크를 OR로 결합해 attention score에 반영

    Shapes:
        - query: (B, T_q, E) 또는 (T_q, B, E)  [batch_first에 따라]
        - key  : (B, T_k, E_k) 또는 (T_k, B, E_k)  (E_k=kdim 또는 embed_dim)
        - value: (B, T_k, E_v) 또는 (T_k, B, E_v)  (E_v=vdim 또는 embed_dim)
        - output: query와 동일한 배치 포맷으로 (B, T_q, E) 또는 (T_q, B, E)
        - attn_weights:
            need_weights=True일 때 반환, 기본은 head 평균 (average_attn_weights=True)
            shape = (B, T_q, T_k) 또는 (B, H, T_q, T_k) (head 평균 여부에 따라)

    Notes:
        - Self-Attention: query=key=value일 때 내부 의존성 학습(각 토큰이 어떤 토큰을 보아야 하는가).
        - Cross-Attention: query ≠ key/value일 때 외부 메모리에서 정보 검색.
        - 마스크는 **True/1인 위치를 가림(mask-out)**. (padding/causal 등)
        - 수치 안정성: 내부적으로 scaled-dot-product(q·k^T / sqrt(d_k)) 사용.
        - 메모리/연산 복잡도: O(B * H * T_q * T_k).

    Example:
        >>> mha = MultiHeadAttention(embed_dim=256, num_heads=8, batch_first=True)
        >>> x = torch.randn(2, 16, 256)  # (B, T, E)
        >>> y, attn = mha(x, x, x, need_weights=True)  # self-attention
        >>> y.shape, attn.shape
        (torch.Size([2, 16, 256]), torch.Size([2, 16, 16]))
    """
    def __init__(self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
            qkv_projection = True,
            ind_qkv_projection=False,
            feature_dim = None
        ):
        """
        Args:
            embed_dim (int):
                입력/출력의 임베딩 차원. 전체 모델 차원.
            num_heads (int):
                멀티헤드 개수. `embed_dim % num_heads == 0` 이어야 하며, head_dim = embed_dim // num_heads.
            dropout (float, default=0.0):
                어텐션 가중치에 적용할 드롭아웃 비율(regularization).
            bias (bool, default=True):
                Q/K/V 및 out projection에 bias를 둘지 여부.
            add_bias_kv (bool, default=False):
                학습 가능한 bias 키/값 벡터를 K/V에 추가하여, 전역(anchor) 참조점을 제공.
            add_zero_attn (bool, default=False):
                0 벡터를 K/V의 마지막에 추가하여, 필요 시 "정보 없음" 선택지를 제공.
            kdim (int | None, default=None):
                K의 입력 차원(선형사상 전). None이면 embed_dim을 사용.
            vdim (int | None, default=None):
                V의 입력 차원(선형사상 전). None이면 embed_dim을 사용.
            batch_first (bool, default=False):
                입력이 (B, T, E)인지 (T, B, E)인지 지정. False면 (T, B, E)로 간주.
            device, dtype:
                파라미터 초기화 장치/자료형.
            qkv_projection (bool, default=True):
                True면 입력 query/key/value를 각각 학습 가능한 선형 변환으로 Q, K, V를 생성합니다.
                False면 입력을 그대로 Q, K, V로 사용하며, 외부에서 이미 계산된 Q/K/V를 전달하는 경우에 적합합니다.

        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim은 num_heads의 배수여야 합니다."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.qkv_projection = qkv_projection
        self.ind_qkv_projection = ind_qkv_projection
        
        if self.qkv_projection:
            if self.ind_qkv_projection:
                self.q_embed_linear = EmbeddingLinear(feature_dim, embed_dim, embed_dim, bias=bias)
                self.k_embed_linear = EmbeddingLinear(feature_dim, embed_dim, embed_dim, bias=bias)
                self.v_embed_linear = EmbeddingLinear(feature_dim, embed_dim, embed_dim, bias=bias)
                
            else:
                self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))
                self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, self.kdim, **factory_kwargs))
                self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, self.vdim, **factory_kwargs))

                if bias:
                    self.in_proj_bias_q = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
                    self.in_proj_bias_k = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
                    self.in_proj_bias_v = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
                else:
                    self.register_parameter('in_proj_bias_q', None)
                    self.register_parameter('in_proj_bias_k', None)
                    self.register_parameter('in_proj_bias_v', None)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
        else:
            self.bias_k = None
            self.bias_v = None

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Scaled Dot-Product Attention 모듈
        self.attn = ScaledDotProductAttention(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.qkv_projection and (self.ind_qkv_projection is False):
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

            if self.in_proj_bias_q is not None:
                nn.init.constant_(self.in_proj_bias_q, 0.)
                nn.init.constant_(self.in_proj_bias_k, 0.)
                nn.init.constant_(self.in_proj_bias_v, 0.)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
            Args:
                query, key, value (torch.Tensor):
                    - batch_first=True  → (B, T_*, E_*)
                    - batch_first=False → (T_*, B, E_*)
                    주의: key/value의 마지막 차원은 kdim/vdim에 따라 달라질 수 있으나,
                        내부 선형사상 후에는 embed_dim으로 정규화됩니다.

                key_padding_mask (torch.Tensor | None):
                    패딩 위치를 가릴 마스크. True/1 위치가 **가려짐**.
                    shape 예: (B, T_k) 또는 broadcast 가능 형태.
                    내부에서 (B_total, 1, 1, T_k)로 변형되어 attention score에 반영.

                attn_mask (torch.Tensor | None):
                    causal/제한 마스크(상삼각 등). True/1 위치가 **가려짐**.
                    shape 예: (1, 1, T_q, T_k) 또는 (B, H, T_q, T_k).
                    key_padding_mask와 OR로 결합.

                need_weights (bool, default=True):
                    어텐션 확률(가중치) 반환 여부.

                average_attn_weights (bool, default=True):
                    True면 Head 평균을 반환 → shape (B, T_q, T_k)
                    False면 Head별 가중치 유지 → shape (B, H, T_q, T_k)
                    
                is_causal (bool, default=False):
                    True로 설정하면 **자동으로 causal mask(미래 토큰 차단)**를 생성하여 적용합니다.
                    causal mask는 현재 시점 이후의 모든 토큰을 보지 못하게 하는 상삼각 형태의 마스크이며,
                    주로 **Decoder** 또는 **GPT 계열 모델**에서 사용됩니다.
                    이 옵션을 활성화하면, `attn_mask`에 별도로 causal mask를 전달하지 않아도
                    내부에서 `(T_q, T_k)` 크기의 상삼각 마스크를 생성하여 attention score에 반영합니다.
                    key_padding_mask 및 attn_mask와 OR 연산으로 결합됩니다.

            Returns:
                output (torch.Tensor):
                    query 포맷과 동일한 배치 포맷의 어텐션 결과.
                attn_weights (torch.Tensor | None):
                    need_weights=True일 때 반환. 위 설명에 따른 shape.

            Notes:
                - 메모리/연산량은 T_q와 T_k의 곱에 선형 비례.
                - key_padding_mask/attn_mask는 bool 타입 권장(True=mask-out).
                - add_bias_kv/add_zero_attn은 "항상 참조 가능한 위치"를 추가해 안정성/표현력 향상에 기여할 수 있음.
        """

        if not self.batch_first:
            query = query.transpose(0, -3)
            key = key.transpose(0, -3)
            value = value.transpose(0, -3)

        batch_shape = query.shape[:-2]
        B_total = int(torch.prod(torch.tensor(batch_shape)))
        T_q = query.size(-2)
        T_k = key.size(-2)

        query = query.reshape(B_total, T_q, self.embed_dim)
        key = key.reshape(B_total, T_k, self.kdim)
        value = value.reshape(B_total, T_k, self.vdim)
        
        if self.qkv_projection:
            if self.ind_qkv_projection:
                query = self.q_embed_linear(query)
                key = self.k_embed_linear(key)
                value = self.v_embed_linear(query)
            else:
                query = F.linear(query, self.q_proj_weight, self.in_proj_bias_q)
                key = F.linear(key, self.k_proj_weight, self.in_proj_bias_k)
                value = F.linear(value, self.v_proj_weight, self.in_proj_bias_v)

        if self.bias_k is not None and self.bias_v is not None:
            key = torch.cat([key, self.bias_k.expand(B_total, -1, -1)], dim=1)
            value = torch.cat([value, self.bias_v.expand(B_total, -1, -1)], dim=1)
            T_k = key.size(1)

        if self.add_zero_attn:
            key = torch.cat([key, torch.zeros(B_total, 1, self.embed_dim, device=key.device, dtype=key.dtype)], dim=1)
            value = torch.cat([value, torch.zeros(B_total, 1, self.embed_dim, device=value.device, dtype=value.dtype)], dim=1)
            T_k = key.size(1)

        query = query.view(B_total, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B_total, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B_total, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # key_padding_mask 적용
        if key_padding_mask is not None:
            mask = key_padding_mask.reshape(B_total, 1, 1, T_k)
            attn_mask_combined = mask
        else:
            attn_mask_combined = None

        # attn_mask 적용
        if attn_mask is not None:
            if attn_mask_combined is None:
                attn_mask_combined = attn_mask
            else:
                attn_mask_combined = attn_mask_combined | attn_mask
        
        if is_causal:
            # is_causal 적용
            causal_mask = torch.triu(torch.ones(T_q, T_k, dtype=torch.bool, device=query.device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)
            if attn_mask_combined is None:
                attn_mask_combined = causal_mask
            else:
                attn_mask_combined = attn_mask_combined | causal_mask

        # Scaled Dot-Product Attention 호출
        attn_output, attn_weights = self.attn(query, key, value, attn_mask=attn_mask_combined)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B_total, T_q, self.embed_dim)
        output = self.out_proj(attn_output)

        output = output.view(*batch_shape, T_q, self.embed_dim)
        attn_weights = attn_weights.view(*batch_shape, self.num_heads, T_q, T_k)

        if not self.batch_first:
            output = output.transpose(0, -3)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=-3)
            return output, attn_weights
        else:
            return output, None

# a = torch.rand(10,5,3,4)

# mha = nn.MultiheadAttention(4,2, batch_first=True)
# mha = MultiHeadAttentionLayer(4,2, batch_first=True)
# a3, a4 = mha(a,a,a)
# a3.shape, a4.shape



####################################################################################################################################






# (Layer of TransformerEncoder) ###################################################################################################################################
class PreLN_TransformerEncoderLayer(nn.Module):
    """
    PreLN_TransformerEncoderLayer
    -----------------------------
    **Pre-LayerNorm** 구조의 Transformer Encoder Layer.
    각 토큰/타임스텝(또는 feature 위치)이 **Self-Attention**으로 서로의 정보를 참고하고,
    이어서 **Position-wise Feed-Forward Network(FFN)**로 비선형 변환을 수행한 뒤
    잔차 연결(residual)로 안정적으로 학습되도록 설계된 블록입니다.

    직관:
        - LayerNorm을 각 서브레이어(MHA, FFN) **입구에서** 적용(=Pre-LN)하여
          깊은 네트워크에서도 gradient 흐름이 더 안정적입니다.
        - Self-Attention은 "이 토큰이 어떤 다른 토큰(과거/현재/미래)에 주목해야 하는가"를 학습적으로 결정합니다.
        - FFN은 토큰별로 독립적인 비선형 확장을 통해 표현력을 높입니다.

    Shapes:
        - 입력 src: (B, T, E) if batch_first else (T, B, E)
        - 출력 src: 입력과 동일한 포맷/shape

    Notes:
        - `is_causal=True`면 **미래 토큰을 자동 차단**(causal mask)하여 언어모델링 등 순방향 과제에 적합.
        - `src_mask`와 `src_key_padding_mask`로 임의의 마스킹/패딩 무시를 동시 지원.
        - 본 구현은 FFN에 Pre-LN을 포함(LN → Linear → ReLU → Dropout → Linear)하여
          MHA와 FFN 모두 **서브레이어 입구에서 정규화**되는 전형적 Pre-LN 패턴을 따릅니다.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        """
        Args:
            d_model (int):
                모델의 hidden dimension (토큰 임베딩 차원).
            nhead (int):
                Multi-Head Attention의 헤드 개수.
            dim_feedforward (int):
                FFN 내부 확장 차원(보통 d_model의 2~4배).
            dropout (float, default=0.1):
                MHA 가중치 및 FFN 내부 드롭아웃 비율.
            batch_first (bool, default=True):
                True면 입력/출력 텐서가 (B, T, E) 포맷, False면 (T, B, E)로 처리.
        """
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(d_model)    # layer_norm1
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff_layer = nn.Sequential(
            nn.LayerNorm(d_model),      # layer_norm2
            nn.Linear(d_model, dim_feedforward),     # FF_linear1
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),     # FF_linear2
            
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            src_key_padding_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Args:
            src (Tensor):
                입력 시퀀스. shape = (B, T, E) if batch_first else (T, B, E)
            src_mask (Tensor | None):
                (T, T) 또는 (B, T, T) 형태의 마스크. True/1인 위치를 가립니다(mask-out).
                causal 외의 임의 attention 억제에 사용.
            is_causal (bool, default=False):
                True면 미래 정보 차단(upper-triangular mask 자동 적용).
                언어모델링/오토레그레시브 설정에서 사용.
            src_key_padding_mask (Tensor | None):
                패딩 토큰 위치를 가릴 마스크. shape = (B, T), True/1 위치가 mask-out.

        Returns:
            Tensor:
                인코더 레이어를 통과한 출력. 입력과 동일한 포맷(shape 유지).

        Flow:
            1) MHA 블록
               - src_norm = LN(src)
               - attn_output = MHA(src_norm, src_norm, src_norm, mask들…)
               - src = src + Dropout(attn_output)
            2) FFN 블록
               - src = src + FFN(LN(src))   # (필요시 FFN 출력에 dropout2 적용 가능)

        Intuition:
            - Self-Attention: 각 토큰이 참조해야 할 다른 토큰(과거/현재/미래)을 확률적으로 선택.
            - Causal 설정: 미래 토큰을 보지 못하게 하여 정보 누출을 방지.
            - Pre-LN: 서브레이어 입구에서 정규화하여, 깊은 네트워크의 학습 안정성 향상.
        """
        # Pre-LN before MHA
        src_norm = self.layer_norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask,
                                        is_causal=is_causal)    # is_causal : 미래정보차단여부 (src_mask를 안넣어도 자동으로 차단해줌)
        src = src + self.dropout1(attn_output)

        # Pre-LN before FFN
        src = src + self.ff_layer(src)
        return src




class FeatureWiseTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, feature_dim, dim_feedforward=2048, dropout=0.1, 
                batch_first=True, qkv_projection=True, ind_qkv_projection=True):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(d_model)      # layer_norm1
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=True, 
                                    qkv_projection=qkv_projection, ind_qkv_projection=True, feature_dim=feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff_layer = nn.Sequential(
            nn.LayerNorm(d_model),      # layer_norm2
            EmbeddingLinear(feature_dim, d_model, dim_feedforward),     # FF_linear1
            nn.ReLU(),
            nn.Dropout(dropout),
            EmbeddingLinear(feature_dim, dim_feedforward, d_model),     # FF_linear2
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        src_key_padding_mask: Optional[torch.Tensor] = None
        ):
        
        # Pre-LN before MHA
        src_norm = self.layer_norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask,
                                        is_causal=is_causal)    # is_causal : 미래정보차단여부 (src_mask를 안넣어도 자동으로 차단해줌)
        src = src + self.dropout1(attn_output)

        # Pre-LN before FFN
        src = src + self.ff_layer(src)
        return src

####################################################################################################################################








# (Layer of AttentionPooling) ###################################################################################################################################

class AttentionPooling(nn.Module):
    """
    AttentionPoolingLayer
    ---------------------
    시퀀스(feature sequence)를 **학습 가능한 Attention 기반 Pooling** 방식으로 하나의 벡터로 요약하는 레이어.

    직관:
        - 평균(mean pooling)이나 최대값(max pooling)은 모든 위치를 동일하게 취급하지만,
          Attention Pooling은 **중요한 위치에 더 큰 가중치**를 부여합니다.
        - 학습 가능한 Query 벡터(query parameter)가 전체 입력 시퀀스의 각 토큰(feature)에 대한 중요도를 평가하고,
          그 확률 분포(attention weights)를 기반으로 가중합을 수행하여 전체 시퀀스의 대표 벡터를 생성합니다.

    수식 요약:
        1) Score 계산: s_t = x_t · q
        2) Softmax 정규화: a_t = softmax(s_t)
        3) (옵션) Learnable Threshold(τ) 적용:
           a'_t = ReLU(a_t - τ) / Σ_t ReLU(a_t - τ)
           → 작은 중요도 항목을 제거(sparsity 유도)
        4) 출력: pooled = Σ_t a'_t * x_t

    Shapes:
        - 입력 x: (B, T, d_model)
            B = batch size, T = sequence length, d_model = feature dimension
        - mask: (B, T) or broadcastable (True/1 = keep, False/0 = mask out)
        - 출력:
            pooled: (B, d_model)
            attn_weights: (B, T)

    동작 방식:
        1) query(학습 파라미터)와 x 간 내적을 통해 각 시점의 중요도(attention score) 계산
        2) softmax로 확률 분포화
        3) (옵션) learnable threshold로 희소성(sparsity) 조절
        4) attention 가중합(weighted sum)으로 전체 시퀀스를 하나의 벡터로 요약

    특징:
        - 입력 길이 T가 달라도, 항상 (B, d_model) 크기의 고정된 출력을 생성.
        - learnable_threshold=True 설정 시, 불필요한 토큰의 attention을 억제하는 “Soft Sparse Pooling” 효과.
        - 일반적인 attention-pooling 모듈(예: Sentence Embedding, Graph Readout, Feature Aggregation 등)에 직접 사용 가능.

    Example:
        >>> x = torch.randn(2, 5, 128)  # (batch=2, seq_len=5, embed_dim=128)
        >>> pool = AttentionPoolingLayer(128, learnable_threshold=True)
        >>> pooled, attn = pool(x)
        >>> pooled.shape, attn.shape
        (torch.Size([2, 128]), torch.Size([2, 5]))

    """
    def __init__(self, d_model, learnable_threshold=False, eps=1e-8):
        """
        Args:
            d_model (int):
                입력 feature 차원. (각 시점/토큰의 embedding 차원)
            learnable_threshold (bool, default=False):
                True일 경우, softmax 결과에서 일정 threshold 이하를 0으로 만들어
                sparse attention을 학습하는 learnable threshold τ를 도입합니다.
                τ는 sigmoid로 squash되어 (0,1) 범위 내에서 동작합니다.
            eps (float, default=1e-8):
                0으로 나누는 것을 방지하기 위한 안정화 상수.
        """
        super().__init__()
        self.learnable_threshold = learnable_threshold
        self.eps = eps
        
        self.query = nn.Parameter(torch.randn(d_model)/d_model)  # 학습 가능한 Query (d_model, ) : 어떤 방식으로 요약할까?, 무엇이 중요한 시점인지를 학습하기 위함
        
        if self.learnable_threshold:
            self.threshold = nn.Parameter(torch.randn(1)/d_model)

    def forward(self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """
        Args:
            x (Tensor):
                입력 텐서 of shape (B, T, d_model)
            mask (Tensor | None):
                어텐션 마스크. shape = (B, T) 또는 broadcast 가능.
                0 또는 False인 위치는 마스킹되어 attention score에 반영되지 않음.

        Returns:
            pooled (Tensor):
                attention 가중합으로 요약된 대표 벡터. shape = (B, d_model)
            attn_weights (Tensor):
                softmax로 정규화된 attention 분포. shape = (B, T)

        Flow:
            1) score 계산 → attn_scores = x @ query (B, T)
            2) mask 적용 (있다면 masked_fill)
            3) softmax → attn_weights
            4) (옵션) learnable threshold → 희소성 조정
            5) 가중합(Σ_t a_t * x_t) → pooled
        """
        # x: (B, T, d_model)
        
        # Attention score 계산
        attn_scores = torch.matmul(x, self.query)  # (B, T)
        
        # Attention Mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
        
        # Learnable Threshold : sparcity
        if self.learnable_threshold:
            tau = torch.sigmoid(self.threshold)
            attn_weights = torch.clamp_min(attn_weights - tau, 0.0)
            denom = attn_weights.sum(dim=-2, keepdim=True) + self.eps
            attn_weights = attn_weights / denom
        
        # Weighted Sum
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=-2)  # (B,T,d_modl) * (B,T,1) = (B,T,d_modl) → sum → (B, d_model)
        return pooled, attn_weights

####################################################################################################################################
























############################################################################################################################
# 【Layer for TimeSeries】  ##################################################################################################
############################################################################################################################




# (Layer for MaskedConv1d)  ##################################################################################################
class MaskedConv1d(nn.Conv1d):
    """
    A Conv1d variant that supports **mask-aware convolution** for irregular or padded time-series.

    특징:
        - 입력 x: (B, C, T)
        - 마스크 mask: (B, T) 또는 (B, 1, T)
        - Conv 수행 시 mask된 위치는 x에서 제거되며,
          실제 기여한 유효 타임스텝 개수로 kernel 윈도우 내 평균을 계산
        - padding/결측/불규칙 구간이 많은 시계열에서 안정적인 feature 추출 가능

    처리 과정:
        1) mask를 (B, 1, T)로 정규화
        2) x_masked = x * mask
        3) numerator = conv(x_masked)
        4) denom     = conv(mask, ones_kernel)
        5) out = numerator / (denom + eps)
        6) bias 추가 (if exists)

    Args:
        x (Tensor): (B, C, T)
        mask (Tensor, optional): (B, T) or (B, 1, T)

    Returns:
        Tensor: Mask-aware convolution 결과. (B, out_channels, T)

    Example:
        conv = MaskedConv1d(1, 64, kernel_size=5, padding=2)
        out = conv(x, mask=mask)
    """
    def forward(self, x, mask=None):
        # x: (B, C, T)
        # mask: (B, 1, T) or (B, T) or None
        
        if mask is None:
            return super().forward(x)

         # 1) mask를 (B, 1, T)로 정규화
        if mask.dim() == 2:            # (B, T)
            mask = mask.unsqueeze(1)   # (B, 1, T)
        elif mask.dim() == 3:
            # (B, 1, T)라고 가정, 아니면 에러 내거나 assert
            assert mask.size(1) == 1, "mask의 채널 차원은 1이어야 합니다."
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        mask = mask.to(x.dtype)
        
        # 2) 유효한 값만 사용
        x_masked = x * mask          # (B, C, T)
        numerator = F.conv1d(x_masked, self.weight, bias=None, stride=self.stride, 
                            padding=self.padding, dilation=self.dilation, groups=self.groups)

        # 3) 마스크도 conv해서 실제로 몇 개가 기여했는지 계산
        ones_kernel = torch.ones(1, 1, self.kernel_size[0], device=x.device, dtype=x.dtype)
        
        denom = F.conv1d(mask, ones_kernel, bias=None, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=1,)

        # 4) 평균
        out = numerator / (denom + 1e-8)

        # 4) bias 있으면 마지막에 더해줌
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        return out
 ####################################################################################################################################



