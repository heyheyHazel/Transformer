import torch
from torch import nn
import torch.nn.functional as F
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        '''
        Args:
            vocab_size: 词典大小
            d_model: 维度
        '''
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        ''' 
        输入形状: [batch_size, seq_len] - 整数索引序列
        输出形状: [batch_size, seq_len, d_model] - 词嵌入向量

        例如:
        输入: torch.tensor([[1, 2, 3], [4, 5, 0]])  # [2, 3]
        输出: torch.tensor([[[0.1, 0.2, ...], ...]])  # [2, 3, 512]
        '''
        # 细节：乘以根号d_model（参考论文）
        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    '''位置编码'''
    def __init__(self, d_model, max_len = 5000, dropout = 0.1):
        # 在参数数值大多固定的情况下时，可以直接在这里定义默认数值，后续调用的时候就不需要传入参数。
        # 有默认值的参数必须放在没有默认值的参数后面，否则会报错。
        ''' 
        Args:
        max_len: 序列的最大长度
        d_model: 模型维度
        dropout: 正则化系数
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout() # 正则化

        # 创建一个空的位置编码矩阵 形状是[max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 分子部分 位置索引 生成一个从0到max_len-1的一维张量，形状为[max_len,] unsqueeze(1)是再增加一维成为二维张量[max_len, 1],便于后续广播机制
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 分母部分 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 对偶数列用正弦函数 奇数列用余弦函数 用前面写好的空张量pe来选奇偶维度
        # pe形状为[行,列], pe[:,0::2]表示选择所有行，列从第0列开始到最后一列，步长为2，也就是所有偶数列，代入正弦函数计算
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # 增加一个批次维度，形状变成[1, max_len, d_model]以便于后续的广播
        pe = pe.unsqueeze(0)

        # 将pe注册为缓冲区，成为模型的一部分但不参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self,x):
        """
        Args:
            x: embedding的词向量, 形状为 [batch_size, seq_len, d_model]
        Return:
            加上位置编码后的张量, 形状同x
        """
        # pe形状为[1, max_len, d_model], 在相加时要去除多余的长度, 和seq_len保持一致即可
        # x.size(1)就是取x的第一维的形状 即seq_len
        # pe[:, :x.size(1)]表示 取所有的batch_size，取0到seq_len的行数和所有的d_model维度，最后一个切片可省略不写
        # 最终x的形状仍然为[batch_size, seq_len, d_model], 只是在embedding基础上多加了位置编码的信息
        x = x + self.pe[:, :x.size(1)]  
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    '''多头注意力机制'''
    def __init__(self, d_model, num_heads, dropout = 0.1):
        ''' 
        Args:
            d_model: 维度
            num_heads: 头的数量 原文中有8个头
            dropout: 正则化系数
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度 512/8 = 64

        # 通过线性变换，将输入的变量映射到Q、K、V三个矩阵中，每个线性变换都有不同的权重，即矩阵包含不同的信息
        # 输入维度和输出维度均为d_model, 形状为[batch_size, seq_len, d_model]
        self.w_q = nn.Linear(d_model, d_model)  # Q
        self.w_k = nn.Linear(d_model, d_model)  # K
        self.w_v = nn.Linear(d_model, d_model)  # V
        self.w_o = nn.Linear(d_model, d_model)  # 输出矩阵

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask = None):
        ''' 
        Args:
            query: [batch_size, q_seq_len, d_model]
            key: [batch_size, k_seq_len, d_model]
            value: [batch_size, v_seq_len, d_model]
            mask: [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        '''
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)
        
        # 首先传入query经过w_q的线性变化，view()用于重塑张量的结构，先把d_model拆成num_heads*d_k，再用transpose调换顺序
        # 如果保持形状为 [batch_size, seq_len, num_heads, d_k]，那么对于每个头，需要从第三个维度中提取数据，这会导致内存访问不连续
        # 而且无法直接使用矩阵乘法同时计算所有头的注意力，所以需要transpose调换顺序
        # 最终QKV的形状是: [batch_size, num_heads, q_seq_len, d_k]
        Q = self.w_q(query).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)

        ''' 
        注意力计算流程: 1. 注意力分数 2.应用掩码(可选) 3. 计算权重 4. 加权求和
        '''
        # 计算注意力分数
        # score的形状是[batch_size, num_heads, q_seq_len, k_seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 如果有mask的话 把一部分score替换成极小数
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重 在score最后一个维度上应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 注意力权重矩阵的形状是[batch_size, num_heads, q_seq_len, k_seq_len] 其中最后一列已经softmax归一化
        # V的形状是[batch_size, num_heads, k_seq_len, d_k]（注意k_seq_len 必须等于 v_seq_len）
        # 注意力权重和V的后两个维度相乘 得到[batch_size, num_heads, q_seq_len, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 重塑回原始形状: [batch_size, q_seq_len, d_model]
        # contiguous()确保内存连续存储
        context = context.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(context)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    '''前馈神经网络'''
    def __init__(self, d_model, d_ff, dropout = 0.1):
        ''' 
        Args:
            d_model: 维度
            d_ff: 前馈网络的隐藏层维度
            dropout: 正则化系数
        '''
        super(PositionwiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        ''' 
        Args:
            x: 输入张量, 形状为 [batch_size, seq_len, d_model]
        Return:
            输出张量, 形状同x
        '''
        return self.net(x)
    


class AddNorm(nn.Module):
    '''残差连接和层归一化'''
    def __init__(self, d_model, dropout = 0.1):
        ''' 
        Args:
            d_model: 维度
            dropout: 正则化系数
        '''
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)  # 层归一化 直接使用nn的内部实现
        self.dropout = nn.Dropout(dropout) # 正则化

    def forward(self, x, sublayer):
        ''' 
        Args:
            x: 输入张量, 形状为 [batch_size, seq_len, d_model]
            sublayer: 子层函数, 例如注意力机制或前馈网络的输出
        Return:
            输出张量, 形状同x
        '''
        # 残差连接 + 层归一化
        return self.norm(x + self.dropout(sublayer))


class EncoderLayer(nn.Module):
    '''编码器层'''
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        ''' 
        Args:
            d_model: 维度
            num_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            dropout: 正则化系数
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 多头自注意力机制
        self.add_norm1 = AddNorm(d_model, dropout)  # 残差连接和层归一化1
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈神经网络
        self.add_norm2 = AddNorm(d_model, dropout)  # 残差连接和层归一化2

    def forward(self, x, mask = None):
        ''' 
        Args:
            x: 输入张量, 形状为 [batch_size, seq_len, d_model]
            mask: 掩码张量, 形状为 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        Return:
            输出张量, 形状同x
        '''
        # 多头自注意力机制 + 残差连接和层归一化
        # MultiHeadAttention模块的输出output和weight，_表示忽略第二个输出，只保留output
        # QKV接收同一个输入x
        attn_output, _ = self.self_attn(x, x, x, mask)
        # Attention后连接AddNorm，AddNorm分别接收原始输入x和注意力机制的输出attn_output
        x = self.add_norm1(x, attn_output)

        # 前馈神经网络 + 残差连接和层归一化
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)

        return x
    

class DecoderLayer(nn.Module):
    '''解码器层'''
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        ''' 
        Args:
            d_model: 维度
            num_heads: 注意力头的数量
            d_ff: 前馈网络的隐藏层维度
            dropout: 正则化系数
        '''
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)
    
    def forward(self, x, encoder_output, src_mask = None, tgt_mask = None):
        ''' 
        Args:
            x: 输入张量， 形状为 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出张量, 形状为 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码张量, 形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 目标序列掩码张量, 形状为 [batch_size, 1, 1, tgt_seq_len] 或 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        '''
        # 掩码多头自注意力机制 + 残差连接和层归一化
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.add_norm1(x, attn_output)

        # 编码器-解码器交叉注意力机制 + 残差连接和层归一化
        # 交叉注意力机制中的KV来自于encoder计算输出，Q来自于decoder
        attn_output, _ = self.enc_dec_attn(x, encoder_output, encoder_output, src_mask)
        x = self.add_norm2(x, attn_output)

        # 前馈神经网络 + 残差连接和层归一化
        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)

        return x
    

class Encoder(nn.Module):
    '''编码器'''
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len = 5000, dropout = 0.1):
        super(Encoder, self).__init__()

        # 初始化时传入参数并存储
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask = None):
        ''' 
        Args:
            src: 源序列张量, 形状为 [batch_size, src_seq_len]
            src_mask: 源序列掩码张量, 形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
        Return:
            编码器输出张量, 形状为 [batch_size, src_seq_len, d_model]
        '''
        # 输入嵌入和位置编码
        x = self.embedding(src)  # [batch_size, src_seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, src_seq_len, d_model]
        x = self.dropout(x)

        # 通过每一层编码器层 
        # 只需要传入动态变化的输入和掩码
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    

class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len = 5000, dropout = 0.1):
        super(Decoder, self).__init__()

        # 初始化时传入参数并存储
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # 输出线性层
        self.output_linear = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, src_mask = None, tgt_mask = None):
        ''' 
        Args:
            tgt: 目标序列张量, 形状为 [batch_size, tgt_seq_len], 在训练阶段tgt是完整的目标序列, 推理阶段是encoder逐步生成的部分序列
            encoder_output: 编码器输出张量, 形状为 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码张量, 形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 目标序列掩码张量, 形状为 [batch_size, 1, 1, tgt_seq_len] 或 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        Return:
            解码器输出张量, 形状为 [batch_size, tgt_seq_len, d_model]
        '''
        # 输入嵌入和位置编码
        x = self.embedding(tgt)  # [batch_size, tgt_seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, tgt_seq_len, d_model]
        x = self.dropout(x)

        # 通过每一层解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # 输出线性层
        x = self.output_linear(x) # [batch_size, tgt_seq_len, vocab_size]
        
        return x


class Transformer(nn.Module):
    '''完整的Transformer模型'''
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model = 512, num_heads = 8, d_ff = 2048,
                 num_encoder_layers = 6, num_decoder_layers = 6, max_len = 5000, dropout = 0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers, max_len, dropout)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        ''' 
        Args:
            src: 源序列张量, 形状为 [batch_size, src_seq_len]
            tgt: 目标序列张量, 形状为 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码张量, 形状为 [batch_size, 1, 1, src_seq_len] 或 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 目标序列掩码张量, 形状为 [batch_size, 1, 1, tgt_seq_len] 或 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        Return:
            Transformer输出张量, 形状为 [batch_size, tgt_seq_len, vocab_size]
        '''
        # 编码器输出
        encoder_output = self.encoder(src, src_mask)

        # 解码器输出
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return decoder_output
    

# 测试代码
if __name__ == "__main__":
    # 设置随机种子以便重现
    torch.manual_seed(42)
        
    # 简化的测试函数
    def quick_test():
        src_vocab_size = 100
        tgt_vocab_size = 100
        d_model = 64
        num_encoder_layers = 6
        num_decoder_layers = 6
        num_heads = 4
        d_ff = 128
        max_len = 50
        
        print("创建模型...")
        # 传入Transformer参数
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len
        )
        
        print("创建测试数据...")
        batch_size = 2
        src_len = 10
        tgt_len = 8
        
        src = torch.randint(1, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
        
        print(f"源序列形状: {src.shape}")   # [batch_size, src_len]
        print(f"目标序列形状: {tgt.shape}") # [batch_size, tgt_len]
        
        # 前向传播测试
        print("运行前向传播...")
        try:
            output = model(src, tgt)
            print(f"前向传播成功!")
            print(f"输出形状: {output.shape}")
            print(f"预期形状: [{batch_size}, {tgt_len}, {tgt_vocab_size}]")
            
            # 检查形状是否正确
            if output.shape == (batch_size, tgt_len, tgt_vocab_size):
                print("形状匹配！基本功能测试通过！")
                return True
            else:
                print(f"形状不匹配！期望: {(batch_size, tgt_len, tgt_vocab_size)}，实际: {output.shape}")
                return False
                
        except Exception as e:
            print(f"前向传播失败: {e}")
            return False
    
    # 运行测试
    success = quick_test()
    
    if success:
        print("\n Transformer代码基本功能正常! ")
    else:
        print("\n 请检查代码实现")