"""Console script for owl2vec_star."""
import configparser
import os
import sys

import click
from owl2vec_star import owl2vec_star


import nltk
nltk.download('punkt')


@click.group()
def main():
    pass




@main.command()
@click.option("--ontology_file", type=click.Path(exists=True), default=None, help="The input ontology for embedding")
@click.option("--embedding_dir", type=click.Path(exists=True), default=None, help="The output embedding directory")
@click.option("--config_file", type=click.Path(exists=True), default='default.cfg', help="Configuration file")
@click.option("--URI_Doc", help="Using URI document", is_flag=True)
@click.option("--Lit_Doc", help="Using literal document", is_flag=True)
@click.option("--Mix_Doc", help="Using mixture document", is_flag=True)
def standalone(ontology_file, embedding_dir, config_file, uri_doc, lit_doc, mix_doc):
    config = configparser.ConfigParser()
    config.read(click.format_filename(config_file))

    if ontology_file:
        config['BASIC']['ontology_file'] = click.format_filename(ontology_file)

    if embedding_dir:
        config['BASIC']['embedding_dir'] = click.format_filename(embedding_dir)

    if uri_doc:
        config['DOCUMENT']['URI_Doc'] = 'yes'
    if lit_doc:
        config['DOCUMENT']['Lit_Doc'] = 'yes'
    if mix_doc:
        config['DOCUMENT']['Mix_Doc'] = 'yes'
    if 'cache_dir' not in config['DOCUMENT']:
        config['DOCUMENT']['cache_dir'] = './cache'

    if not os.path.exists(config['DOCUMENT']['cache_dir']):
        os.mkdir(config['DOCUMENT']['cache_dir'])

    if 'embedding_dir' not in config['BASIC']:
        config['BASIC']['embedding_dir'] = os.path.join(config['DOCUMENT']['cache_dir'], 'output/')

    if not os.path.exists(config['BASIC']['embedding_dir']):
        os.mkdir(config['BASIC']['embedding_dir'])
        
 
    #Call to OWL2Vec*    
    model_ = owl2vec_star.__perform_ontology_embedding(config)


    #Gensim format
    model_.save(config['BASIC']['embedding_dir']+"ontology.embeddings")
    #Txt format
    model_.wv.save_word2vec_format(config['BASIC']['embedding_dir']+"ontology.embeddings.txt", binary=False)
    
    print('Model saved. Done!')
    
    

    return 0


@main.command()
@click.option("--ontology_dir", type=click.Path(exists=True), default=None, help="The directory of input ontologies for embedding")
@click.option("--embedding_dir", type=click.Path(exists=True), default=None, help="The output embedding directory")
@click.option("--config_file", type=click.Path(exists=True), default='default_multi.cfg', help="Configuration file")
@click.option("--URI_Doc", help="Using URI document", is_flag=True)
@click.option("--Lit_Doc", help="Using literal document", is_flag=True)
@click.option("--Mix_Doc", help="Using mixture document", is_flag=True)
def standalone_multi(ontology_dir, embedding_dir, config_file, uri_doc, lit_doc, mix_doc):
    # read and combine configurations
    # overwrite the parameters in the configuration file by the command parameters
    config = configparser.ConfigParser()
    config.read(click.format_filename(config_file))

    if ontology_dir:
        config['BASIC']['ontology_dir'] = click.format_filename(ontology_dir)

    if embedding_dir:
        config['BASIC']['embedding_dir'] = click.format_filename(embedding_dir)

    if uri_doc:
        config['DOCUMENT']['URI_Doc'] = 'yes'
    if lit_doc:
        config['DOCUMENT']['Lit_Doc'] = 'yes'
    if mix_doc:
        config['DOCUMENT']['Mix_Doc'] = 'yes'
    if 'cache_dir' not in config['DOCUMENT']:
        config['DOCUMENT']['cache_dir'] = './cache'

    if not os.path.exists(config['DOCUMENT']['cache_dir']):
        os.mkdir(config['DOCUMENT']['cache_dir'])

    if 'embedding_dir' not in config['BASIC']:
        config['BASIC']['embedding_dir'] = os.path.join(config['DOCUMENT']['cache_dir'], 'output/')
        
    if not os.path.exists(config['BASIC']['embedding_dir']):
        os.mkdir(config['BASIC']['embedding_dir'])
        
    
    
    #Call to OWL2Vec*
    model_ = owl2vec_star.__perform_joint_ontology_embedding(config)
    
    

    #Gensim format
    model_.save(config['BASIC']['embedding_dir']+"ontology.embeddings")
    #Txt format
    model_.wv.save_word2vec_format(config['BASIC']['embedding_dir']+"ontology.embeddings.txt", binary=False)
    

    print('Model saved. Done!')



if __name__ == "__main__":
    print("ciao")
    sys.exit(main())  # pragma: no cover
