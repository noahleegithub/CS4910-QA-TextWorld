import logging
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from layers import Embedding, MergeEmbeddings, EncoderBlock, CQAttention, AnswerPointer, masked_softmax, NoisyLinear

logger = logging.getLogger(__name__)

class Embedder(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

'''
batch = next(iter(train_iter))
input_seq = batch.English.transpose(0,1)
input_pad = EN_TEXT.vocab.stoi['<pad>']# creates mask with 0s wherever there is padding in the input
input_msk = (input_seq != input_pad).unsqueeze(1)

# create mask as beforetarget_seq = batch.French.transpose(0,1)
target_pad = FR_TEXT.vocab.stoi['<pad>']
target_msk = (target_seq != target_pad).unsqueeze(1)size = target_seq.size(1) # get seq_len for matrixnopeak_mask = np.triu(np.ones(1, size, size),
k=1).astype('uint8')
nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)target_msk = target_msk & nopeak_mask
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# build an encoder layer with one multi-head attention layer and one # feed-forward layerclass EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layerclass DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x# We can then build a convenient cloning function that can generate multiple layers:def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output# we don't perform softmax on the output as this will be handled 
# automatically by our loss function

class DQN(torch.nn.Module):

    def __init__(self, config, fasttext):
        super().__init__()
        self.config = config
        self.fasttext = fasttext
        self.W = np.array(f['embedding'])
        self.id2word = [str(b) for b in self.fasttext['words_flatten'][0].split(b'\n')]
        self.word2id = {word: idx for idx, word in enumerate(self.id2word)}

        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(fasttext['embedding']).float(),
            freeze=config.model.word_embedding_trainable)

        self.observation_encoder = None
        self.question_encoder = None

        self.obs_quest_attention = None

        self.action_decoder = None
    
        self.state_action_attention = None

    def _def_layers(self):

        self.encoders = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=7, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.context_question_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)

        self.context_question_attention_resizer = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)

        self.aggregators = torch.nn.ModuleList([EncoderBlock(conv_num=self.aggregation_conv_num, ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim,
                                                                n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.aggregation_layers)])

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear
        self.action_scorer_shared_linear = linear_function(self.block_hidden_dim, self.action_scorer_hidden_dim)

        if self.use_distributional:
            if self.dueling_networks:
                action_scorer_output_size = self.atoms
                action_scorer_advantage_output_size = self.word_vocab_size * self.atoms
            else:
                action_scorer_output_size = self.word_vocab_size * self.atoms
        else:
            if self.dueling_networks:
                action_scorer_output_size = 1
                action_scorer_advantage_output_size = self.word_vocab_size
            else:
                action_scorer_output_size = self.word_vocab_size

        action_scorers = []
        for _ in range(self.generate_length):
            action_scorers.append(linear_function(self.action_scorer_hidden_dim, action_scorer_output_size))
        self.action_scorers = torch.nn.ModuleList(action_scorers)

        if self.dueling_networks:
            action_scorers_advantage = []
            for _ in range(self.generate_length):
                action_scorers_advantage.append(linear_function(self.action_scorer_hidden_dim, action_scorer_advantage_output_size))
            self.action_scorers_advantage = torch.nn.ModuleList(action_scorers_advantage)

        self.answer_pointer = AnswerPointer(block_hidden_dim=self.block_hidden_dim, noisy_net=self.noisy_net)

        if self.answer_type in ["2 way"]:
            self.question_answerer_output_1 = linear_function(self.block_hidden_dim, self.question_answerer_hidden_dim)
            self.question_answerer_output_2 = linear_function(self.question_answerer_hidden_dim, 2)

    def get_match_representations(self, doc_encodings, doc_mask, q_encodings, q_mask):
        # node encoding: batch x num_node x hid
        # node mask: batch x num_node
        X = self.context_question_attention(doc_encodings, q_encodings, doc_mask, q_mask)
        M0 = self.context_question_attention_resizer(X)
        M0 = F.dropout(M0, p=self.block_dropout, training=self.training)
        square_mask = torch.bmm(doc_mask.unsqueeze(-1), doc_mask.unsqueeze(1))  # batch x time x time
        for i in range(self.aggregation_layers):
             M0 = self.aggregators[i](M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        return M0

    def representation_generator(self, _input_words, _input_chars):
        embeddings, mask = self.word_embedding(_input_words)  # batch x time x emb
        char_embeddings, _ = self.char_embedding(_input_chars)  # batch x time x nchar x emb
        merged_embeddings = self.merge_embeddings(embeddings, char_embeddings, mask)  # batch x time x emb
        square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x time x time
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoders[i](merged_embeddings, mask, square_mask, i * (self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc

        return encoding_sequence, mask

    def action_scorer(self, state_representation_sequence, word_masks):
        
        state_representation, _ = torch.max(state_representation_sequence, 1)
        hidden = self.action_scorer_shared_linear(state_representation)  # batch x hid
        hidden = torch.relu(hidden)  # batch x hid
    
        action_ranks = []
        for i in range(self.generate_length):
            a_rank = self.action_scorers[i](hidden)  # batch x n_vocab, or batch x n_vocab*atoms
            if self.use_distributional:
                if self.dueling_networks:
                    a_rank_advantage = self.action_scorers_advantage[i](hidden)  # advantage stream
                    a_rank = a_rank.view(-1, 1, self.atoms)
                    a_rank_advantage = a_rank_advantage.view(-1, self.word_vocab_size, self.atoms)
                    a_rank_advantage = a_rank_advantage * word_masks[i].unsqueeze(-1)
                    q = a_rank + a_rank_advantage - a_rank_advantage.mean(1, keepdim=True)  # combine streams
                else:
                    q = a_rank.view(-1, self.word_vocab_size, self.atoms)  # batch x n_vocab x atoms
                q = masked_softmax(q, word_masks[i].unsqueeze(-1), axis=-1)  # batch x n_vocab x atoms
            else:
                if self.dueling_networks:
                    a_rank_advantage = self.action_scorers_advantage[i](hidden)  # advantage stream, batch x vocab
                    a_rank_advantage = a_rank_advantage * word_masks[i]
                    q = a_rank + a_rank_advantage - a_rank_advantage.mean(1, keepdim=True)  # combine streams  # batch x vocab
                else:
                    q = a_rank   #batch x vocab
                q = q * word_masks[i]
            action_ranks.append(q)
        return action_ranks

    def answer_question(self, matching_representation_sequence, doc_mask):
        square_mask = torch.bmm(doc_mask.unsqueeze(-1), doc_mask.unsqueeze(1))  # batch x time x time
        M0 = matching_representation_sequence
        M1 = M0
        for i in range(self.aggregation_layers):
             M0 = self.aggregators[i](M0, doc_mask, square_mask, i * (self.aggregation_conv_num + 2) + 1, self.aggregation_layers)
        M2 = M0
        pred = self.answer_pointer(M1, M2, doc_mask)  # batch x time
        # pred_distribution: batch x time
        pred_distribution = masked_softmax(pred, m=doc_mask, axis=-1)  # 
        if self.answer_type == "pointing":
            return pred_distribution

        z = torch.bmm(pred_distribution.view(pred_distribution.size(0), 1, pred_distribution.size(1)), M2)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        hidden = self.question_answerer_output_1(z)  # batch x hid
        hidden = torch.relu(hidden)  # batch x hid
        pred = self.question_answerer_output_2(hidden)  # batch x out
        pred = masked_softmax(pred, axis=-1)
        return pred

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_shared_linear.reset_noise()
            for i in range(len(self.action_scorers)):
                self.action_scorers[i].reset_noise()
            self.answer_pointer.zero_noise()
            if self.answer_type in ["2 way"]:
                self.question_answerer_output_1.zero_noise()
                self.question_answerer_output_2.zero_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.action_scorer_shared_linear.zero_noise()
            for i in range(len(self.action_scorers)):
                self.action_scorers[i].zero_noise()
            self.answer_pointer.zero_noise()
            if self.answer_type in ["2 way"]:
                self.question_answerer_output_1.zero_noise()
                self.question_answerer_output_2.zero_noise()
