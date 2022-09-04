import torch.nn as nn
import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch 
import numpy as np 



class SingleTweet_model(nn.Module):
    
    def __init__(self, emb_matrix: np.ndarray, cfg : dict, device) :
        super().__init__()

        self.cfg = cfg
        self.device = device 

        self.embedding_layer, self.word_embedding_dim = self.build_emb_layer(emb_matrix,cfg['pad_idx'], cfg['freeze_embedding'])

        self.lstm = nn.LSTM(self.word_embedding_dim, cfg['hidden_dim'], num_layers = cfg['num_layers'], batch_first = True, bidirectional = True) 
            
        self.dropout = nn.Dropout(cfg['dropout_p']) 

        self.compress = nn.Linear(cfg['hidden_dim']*2,cfg['hidden_dim'])

        self.classifier = nn.Linear(cfg['hidden_dim'],1)   
    
    def name(self):
        return 'SingleTweet_model'

    def build_emb_layer(self, weights_matrix: np.ndarray, pad_idx : int, freeze : bool):
    
        matrix = torch.from_numpy(weights_matrix).to(self.device)   #the embedding matrix 
        _ , embedding_dim = matrix.shape 

        emb_layer = nn.Embedding.from_pretrained(matrix, freeze=freeze, padding_idx = pad_idx)   #load pretrained weights in the layer and make it non-trainable (TODO: trainable ? )
        
        return emb_layer, embedding_dim
        

    def forward(self, batch_data):
    
        tweets = batch_data['tweets']           # [batch_size, num_tokens]
        tweet_lengths = batch_data['lengths']   # [batch_size]

        #embed each word in a sentence with a n-dim vector 
        word_emb_tweets = self.embedding_layer(tweets)  # word_emb_tweets = [batch_size, num_tokens, embedding_dim]

        #pass the embedded tokens throught lstm network 
        packed_embeddings = pack_padded_sequence(word_emb_tweets, tweet_lengths, batch_first=True, enforce_sorted=False) #tweet_lengths.cpu() TODO
        output, (hn,cn)  = self.lstm(packed_embeddings)   # hn = [2, batch_size, embedding_dim]
        
        #concat forward and backward output
        fwbw_hn = torch.cat((hn[-1,:,:],hn[-2,:,:]),dim=1)  # fwbw_hn = [batch_size, 2*embedding_dim]
        
        #compress the output 
        compressed_out = self.compress(fwbw_hn) # compressed_out = [batch_size, embedding_dim]

        #apply non linearity
        compressed_out = F.relu(compressed_out)

        #eventual dropout 
        if self.cfg['dropout']: out = self.dropout(compressed_out)

        #final classification 
        predictions = self.classifier(out) #predictions [batch_size, 1]

        return predictions