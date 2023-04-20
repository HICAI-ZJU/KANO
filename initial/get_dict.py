from re import S
from rdflib import Graph, URIRef
import pdb
from rdkit import Chem
import itertools
import numpy as np
import pandas as pd
import gensim
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import pickle


onto_path = 'elementkgontology.embeddings.txt'
ontoemb = KeyedVectors.load_word2vec_format(onto_path, binary=False)


with open('../chemprop/data/funcgroup.txt', "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]

# get functional group -> embedding dict
print("getting fg2emb dict ...")
fg2emb = {}
for fg in name:
    fg_name = "http://www.semanticweb.org/ElementKG#"+ fg
    ele_emb = ontoemb[fg_name]
    fg2emb[fg] = ele_emb
pickle.dump(fg2emb, open('fg2emb.pkl','wb'))

# get all element symbol
def get_atom_symbol(atomic_number):
    return Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), atomic_number)

element_symbols = [get_atom_symbol(i) for i in range(1,109)]

eletype_list = [i for i in range(108)]

# get element -> embedding dict
print("getting ele2emb dict ...")
ele2emb = {}
for eletype in eletype_list:
    s = "http://www.semanticweb.org/ElementKG#"+ element_symbols[eletype]
    ele_emb = ontoemb[s]

    ele2emb[eletype] = ele_emb

pickle.dump(ele2emb, open('ele2emb.pkl','wb'))

# get property -> embedding matrix
objectproperty = []
with open('objectproperty.txt') as f:
    for line in f.readlines():
        temp = line.split()
        objectproperty.extend(temp)

new_objectproperty = []
pro2emb = []
for p in range(len(objectproperty)):
    p_name = "http://www.semanticweb.org/ElementKG#" + objectproperty[p]
    if p_name in ontoemb:
        new_objectproperty.append(objectproperty[p])
        pro_emb = ontoemb[p_name]

        pro2emb.append(pro_emb)
pro_emb = np.concatenate(pro2emb, axis=0)
# dimensionality reduction
pca = PCA(n_components=14)
pro2emb = pca.fit_transform(pro2emb)

objectproperty = new_objectproperty

# get property -> embedding dict
print("getting property2emb dict ...")
property2emb = {}
for p in range(len(objectproperty)):
    p_name = URIRef(f"http://www.semanticweb.org/ElementKG#" + objectproperty[p])
    pro_emb = pro2emb[p]
    property2emb[p_name] = pro_emb
    
# pickle.dump(property2emb, open('property2emb.pkl','wb'))
# property2emb = pickle.load(open('property2emb.pkl','rb'))

g = Graph()
g.parse("../KGembedding/elementkg.owl", format="xml")

print("getting rel2emb dict ...")
rel2emb = {}
for i in range(len(element_symbols)-1):
    for j in range(1, len(element_symbols)):
        qr = "select ?relation where { <http://www.semanticweb.org/ElementKG#"+element_symbols[i]+"> ?relation <http://www.semanticweb.org/ElementKG#"+element_symbols[j]+">}"
        relations = g.query(qr)
        relations = list(relations)
        relations = [property2emb[rel[0]] for rel in relations]
        if relations:
            relation = np.mean(relations, axis=0)
            rel2emb[(i,j)] = relation
            rel2emb[(j,i)] = relation
pickle.dump(rel2emb, open('rel2emb.pkl','wb'))    

print("finish!")
               
                   
    