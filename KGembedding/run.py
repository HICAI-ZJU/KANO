from owl2vec_star import owl2vec_star


gensim_model = owl2vec_star.extract_owl2vec_model("elementkg.owl", "default.cfg", True, True, True)
output_folder="./../initial/elementkg"
# #Txt format
gensim_model.wv.save_word2vec_format(output_folder+"ontology.embeddings.txt", binary=False)