import gensim
import gensim.downloader as gloader
from gensim.models import KeyedVectors

glove_model_cached_path = DATA_FOLDER / 'glove_vectors.txt'
glove_model_download_path = 'glove-twitter-200'

def load_glove_emb(force_download = False):   
    """
        Download the glove embedding model and returns it 
    """
    emb_model = None

    if os.path.exists(glove_model_cached_path) and not force_download: 
        print('found cached glove vectors in data folder, retrieving the file...')
        emb_model = KeyedVectors.load_word2vec_format(glove_model_cached_path, binary=True)
        print('vectors loaded')

    else:
        print('downloading glove embeddings...')        
        emb_model = gloader.load(glove_model_download_path)

        print('saving glove embeddings to file')  
        emb_model.save_word2vec_format(glove_model_cached_path, binary=True)
        
    return emb_model

force_download = False      # to download glove model even if the vectors model has been already stored. Mainly for testing purposes