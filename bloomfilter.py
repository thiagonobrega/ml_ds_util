import xxhash
import nltk
import bitarray
import numpy as np

from ds_util import generate_ngrams


def create_bf(word,bf_len,k,bf_representation='binary'):
    """
        For a given string (representing a record), this method creates a new bloom filter.

        word : string to be encoded
        bf_len : bloom filter length
        k : number of hash function
        
        bf_representation : the re
            - binary (default) : the regular bf, a bitarray with 0 and 1's
            - pos1 : this represatation return a a integer list with the positions of ones in the filter

        It is important to mention that:
            - pos1 representation might generate filter of diferent lenght, requiring a regularization
    """

    # gerando os ngrams
    ngrams = generate_ngrams(word)
    bf = []

    #filtro vazio
    if bf_representation == 'binary':
        bf = bitarray.bitarray(bf_len)
        bf.setall(0)

        for n_gram in ngrams:
            for seed in range(k):
                pos=xxhash.xxh64(n_gram, seed=seed).intdigest() % bf_len
                bf[pos] = 1
        # end binary bf
    elif bf_representation == 'pos1':
        for n_gram in ngrams:
            for seed in range(k):
                pos=xxhash.xxh64(n_gram, seed=seed).intdigest() % bf_len
                bf.append(pos)
        
        bf = np.unique(np.asarray(bf,dtype=np.uint16))
        # end pos1 bf

    return bf

def jaccard(filter1, filter2):
    """
        Calculates the jaccard index between 2 bloom filters

        filter1 : bitarray.bitarray
        filter2 : bitarray.bitarray

        return : number between 0 and 1
    """

    # Brasileiro, i've coded this part using this website [1]
    # Please check this, chunk of code. It may contain error :p
    # 1 -http://blog.demofox.org/2015/02/08/estimating-set-membership-with-a-bloom-filter/


    inter = (filter1 & filter2).count(True)
    union = (filter1 | filter2).count(True)

    if union == 0:
        return 0

    return inter / union


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

#####
def completar_bf(bf,tamanho_max):
  '''
    Completa o filtro zeros ao final
  '''
  try:
    z = np.zeros(tamanho_max-len(bf))
  except ValueError:
    print(len(bf))
    print(tamanho_max)
  return np.concatenate((bf,z))