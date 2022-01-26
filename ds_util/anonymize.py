import pandas as pd
import nltk


def get_amostra_anonimizada(sp1,sp2,bf_len,k,amostra=True):
  ps1 = anonimize_dataset_pos(sp1,bf_len,k)
  ps2 = anonimize_dataset_pos(sp2,bf_len,k)

  s1 = anonimize_dataset(sp1,bf_len,k)
  s2 = anonimize_dataset(sp2,bf_len,k)

  rdf = ranquear_dados(s1,s2,ps1,ps2)
  tr = rdf
  
  if amostra:
    tr = pd.DataFrame(retirar_amostra(rdf))

    
  
  tr.columns = ['id1','id2','bf1','bf2','sim','label']

  return tr

def anonimize_dataset(ldf,bf_len,k):

  saida =[]
  for row in ldf:
    id = row[0]
    bf = create_bf(row[1],bf_len,k)
    saida.append([id,bf])
  return saida

def anonimizar_dataset_mult(df,l,k):
  '''
    converte os dataset para serem processados
  '''
  a = np.asarray(df[['id1', 'raw1']])
  b = np.asarray(df[['id2', 'raw2']])

  a = anonimize_dataset_pos(a,l,k)
  b = anonimize_dataset_pos(b,l,k)

  z = pd.concat([pd.DataFrame(a),pd.DataFrame(b),df[['sim','label']]],axis=1)
  z.columns = ['id1','bf1','id2','bf2','sim','label']
  z = z[['id1','id2','bf1','bf2','sim','label']] #ordena
  return z


def anonimize_dataset_pos(ldf,bf_len,k):

  saida =[]
  for row in ldf:
    id = row[0]
    bf = create_bf_pos(row[1],bf_len,k)
    saida.append([id,bf])
  return saida

def get_maior(d1):
  '''
    Metodo utilizado para o filtro posicional
    retorna o tamanho do maior filtro
  '''
  len_max = 0
  for i in range(0,len(d1)):
    if len(d1[i]) > len_max:
      len_max = len(d1[i])
    # if len(d1[i][1]) > len_max:
    #   len_max = len(d1[i][1])
  
  return len_max