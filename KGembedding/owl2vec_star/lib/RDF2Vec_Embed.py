import rdflib
import sys
import numpy as np

from owl2vec_star.rdf2vec.embed import RDF2VecTransformer
from owl2vec_star.rdf2vec.graph import KnowledgeGraph, Vertex
from owl2vec_star.rdf2vec.walkers.random import RandomWalker
from owl2vec_star.rdf2vec.walkers.weisfeiler_lehman import WeisfeilerLehmanWalker


def construct_kg_walker(onto_file, walker_type, walk_depth):
    g = rdflib.Graph()
    if onto_file.endswith('ttl') or onto_file.endswith('TTL'):
        g.parse(onto_file, format='turtle')
    else:
        g.parse(onto_file)
    kg = KnowledgeGraph()
    for (s, p, o) in g:
        s_v, o_v = Vertex(str(s)), Vertex(str(o))
        p_v = Vertex(str(p), predicate=True, _from=s_v, _to=o_v)
        kg.add_vertex(s_v)
        kg.add_vertex(p_v)
        kg.add_vertex(o_v)
        kg.add_edge(s_v, p_v)
        kg.add_edge(p_v, o_v)

    if walker_type.lower() == 'random':
        walker = RandomWalker(depth=walk_depth, walks_per_graph=float('inf'))
    elif walker_type.lower() == 'wl':
        walker = WeisfeilerLehmanWalker(depth=walk_depth, walks_per_graph=float('inf'))
    else:
        print('walker %s not implemented' % walker_type)
        sys.exit()

    return kg, walker


def get_rdf2vec_embed(onto_file, walker_type, walk_depth, embed_size, classes):
    kg, walker = construct_kg_walker(onto_file=onto_file, walker_type=walker_type, walk_depth=walk_depth)
    transformer = RDF2VecTransformer(walkers=[walker], vector_size=embed_size)
    instances = [rdflib.URIRef(c) for c in classes]
    walk_embeddings = transformer.fit_transform(graph=kg, instances=instances)
    return np.array(walk_embeddings)


def get_rdf2vec_walks(onto_file, walker_type, walk_depth, classes):
    kg, walker = construct_kg_walker(onto_file=onto_file, walker_type=walker_type, walk_depth=walk_depth)
    instances = [rdflib.URIRef(c) for c in classes]
    walks_ = list(walker.extract(graph=kg, instances=instances))
    return walks_
