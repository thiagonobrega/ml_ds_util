import pandas as pd
import nltk

def generate_ngrams(word,ngrams=2,pad=False):
    """
    Generate (plain ngrams) of a string (word)
    """
    if pad:
        grams =  nltk.ngrams(word, ngrams,pad_left=True, pad_right=True,left_pad_symbol='$',right_pad_symbol='$')
    else:
        grams = nltk.ngrams(word, ngrams)
    saida = []
    for gram in grams:
        saida.append(''.join(gram))
  
    return saida

def pre_process_raw(df,atts,ngrams=2):
    """
        Pre-processing raw data: generate the bi-grams to be anonymized
        - remove spaces

        return: the preprocessed dataset and maximum number (maior) of ngram in the dataset
    """
    ldf = []
    maior = 0
    for row in df.iterrows():
        id = row[1][0]
        qid = ''
        for i in range(1,len(atts)):
            try:
                for w in row[1][i].split(): #removendo espacos
                    qid += w
            except AttributeError:
                qid += str(row[1][i])
        
        #
        ngrams = generate_ngrams(qid,ngrams=ngrams,pad=False)
        if len(ngrams) >  maior:
            maior = len(ngrams)

        ldf.append([id,ngrams])

    return ldf, maior

def ler_e_extrair_sample(df1,df2,sample_size,percentual_nao_duplicada=0.1,
                         sample_from_same=False,return_raw_sample=False,
                         atts=['voter_id','first_name','last_name','gender','street_address']):
    """
        sample and extract the from one dataset
    """
  
    ss1 = df1.sample(sample_size)
    l1 = int(sample_size*percentual_nao_duplicada)
    l2 = sample_size - l1

    ss2 = []
    if sample_from_same:
        ss2 = pd.concat([df1.sample(l2), ss1.sample(l1)])
    else:
        ss2 = pd.concat([df2.sample(l2), ss1.sample(l1)])
  
    #seleciona attributos
    ss1 = ss1[atts]
    ss2 = ss2[atts]

    sp1 , s1m = pre_process_raw(ss1,atts)
    sp2 , s2m = pre_process_raw(ss2,atts)

    max_ngram = max(s1m,s2m)

    if return_raw_sample:
        return sp1,sp2,max_ngram,ss1,ss2
    else:
        return sp1,sp2,max_ngram