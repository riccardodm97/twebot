import gensim
import random 


def check_OOV_terms(embedding_model: gensim.models.keyedvectors.KeyedVectors, unique_words):
    """
        Given the embedding model and the unique words in the dataframe, determines the out-of-vocabulary words 
    """
    oov_words = []

    if embedding_model is None:
        print('WARNING: empty embeddings model')

    else: 
        for word in unique_words:
            try: 
                embedding_model[word]
            except:
                oov_words.append(word) 
        
        print("Total number of unique words in dataset:",len(unique_words))
        print("Total OOV terms: {0} which is ({1:.2f}%)".format(len(oov_words), (float(len(oov_words)) / len(unique_words))*100))
        print("Some OOV terms:",random.sample(oov_words,10))
    
    return oov_words
