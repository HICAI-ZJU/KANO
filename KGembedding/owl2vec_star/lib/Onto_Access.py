'''
Created on 2 Jan 2019

@author: ejimenez-ruiz
'''
from owlready2 import *
import rdflib
from rdflib.plugins.sparql import prepareQuery
import logging
from enum import Enum


class Reasoner(Enum):
    HERMIT=0 #Not really adding the right set of entailments
    PELLET=1 #Slow for large ontologies
    STRUCTURAL=2  #Basic domain/range propagation
    NONE=3 #No reasoning


class OntologyAccess(object):
    '''
    classdocs
    '''



    def __init__(self, urionto):

        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

        self.urionto = urionto
        #List from owlready2
        #onto_path.append(pathontos) #For local ontologies



    def getOntologyIRI(self):
        return self.urionto



    def loadOntology(self, reasoner=Reasoner.NONE, memory_java='10240'):

        #self.world = World()


        #Method from owlready
        self.onto = get_ontology(self.urionto).load()
        #self.onto = self.world.get_ontology(self.urionto).load()
        #self.onto.load()

        #self.classifiedOnto = get_ontology(self.urionto + '_classified')
        owlready2.reasoning.JAVA_MEMORY=memory_java
        owlready2.set_log_level(9)

        if reasoner==Reasoner.PELLET:

            try:
                with self.onto:  #it does add inferences to ontology

                    # Is this wrt data assertions? Check if necessary
                    # infer_property_values = True, infer_data_property_values = True
                    logging.info("Classifying ontology with Pellet...")
                    sync_reasoner_pellet() #it does add inferences to ontology

                    unsat = len(list(self.onto.inconsistent_classes()))
                    logging.info("Ontology successfully classified.")
                    if unsat > 0:
                        logging.warning("There are " + str(unsat) + " unsatisfiabiable classes.")
            except:
                logging.info("Classifying with Pellet failed.")

        elif reasoner==Reasoner.HERMIT:

                try:
                    with self.onto:  #it does add inferences to ontology

                        logging.info("Classifying ontology with HermiT...")
                        sync_reasoner() #HermiT doe snot work very well....

                        unsat = len(list(self.onto.inconsistent_classes()))
                        logging.info("Ontology successfully classified.")
                        if unsat > 0:
                            logging.warning("There are " + str(unsat) + " unsatisfiabiable classes.")

                except:

                    logging.info("Classifying with HermiT failed.")

        ##End Classification
        ####

        #report problem with unsat (Nothing not declared....)
        #print(list(self.onto.inconsistent_classes()))

        self.graph = default_world.as_rdflib_graph()
        logging.info("There are {} triples in the ontology".format(len(self.graph)))
        #self.graph = self.world.as_rdflib_graph()




    def getOntology(self):
        return self.onto

    #def getInferences(self):
    #    return self.inferences


    #Does not seem to be a better way (or working way) according to the documentation...
    def getClassByURI(self, uri):

        for cls in list(self.getOntology().classes()):
            if (cls.iri==uri):
                return cls

        return None


    def getClassByName(self, name):

        for cls in list(self.getOntology().classes()):
            if (cls.name.lower()==name.lower()):
                return cls

        return None



    def getEntityByURI(self, uri):

        for cls in list(self.getOntology().classes()):
            if (cls.iri==uri):
                return cls

        for prop in list(self.getOntology().properties()):
            if (prop.iri==uri):
                return prop

        return None


    def getEntityByName(self, name):

        for cls in list(self.getOntology().classes()):
            if (cls.name.lower()==name.lower()):
                return cls

        for prop in list(self.getOntology().properties()):
            if (prop.name.lower()==name.lower()):
                return prop


        return None



    def getClassObjectsContainingName(self, name):

        classes = []

        for cls in list(self.getOntology().classes()):
            if (name.lower() in cls.name.lower()):
                classes.append(cls)

        return classes


    def getClassIRIsContainingName(self, name):

        classes = []

        for cls in list(self.getOntology().classes()):
            if (name.lower() in cls.name.lower()):
                classes.append(cls.iri)

        return classes


    def getAncestorsURIsMinusClass(self,cls):
        ancestors_str = self.getAncestorsURIs(cls)

        ancestors_str.remove(cls.iri)

        return ancestors_str



    def getAncestorsURIs(self,cls):
        ancestors_str = set()

        for anc_cls in cls.ancestors():
            ancestors_str.add(anc_cls.iri)

        return ancestors_str


    def getDescendantURIs(self,cls):
        descendants_str = set()

        for desc_cls in cls.descendants():
            descendants_str.add(desc_cls.iri)

        return descendants_str


    def getDescendantNames(self,cls):
        descendants_str = set()

        for desc_cls in cls.descendants():
            descendants_str.add(desc_cls.name)

        return descendants_str



    def getDescendantNamesForClassName(self, cls_name):

        cls = self.getClassByName(cls_name)

        descendants_str = set()

        for desc_cls in cls.descendants():
            descendants_str.add(desc_cls.name)

        return descendants_str



    def isSubClassOf(self, sub_cls1, sup_cls2):

        if sup_cls2 in sub_cls1.ancestors():
            return True
        return False


    def isSuperClassOf(self, sup_cls1, sub_cls2):

        if sup_cls1 in sub_cls2.ancestors():
            return True
        return False



    def getDomainURIs(self, prop):

        domain_uris = set()

        for cls in prop.domain:
            #for c in cls.Classes:
            #    print(c)
            try:
                domain_uris.add(cls.iri)
            except AttributeError:
                pass

        return domain_uris


    def getDatatypeRangeNames(self, prop):

        range_uris = set()

        for cls in prop.range:
            range_uris.add(cls.name)  #datatypes are returned without uri

        return range_uris


    #Only for object properties
    def getRangeURIs(self, prop):

        range_uris = set()

        for cls in prop.range:

            try:
                range_uris.add(cls.iri)
            except AttributeError:
                pass

        return range_uris


    def geInverses(self, prop):

        inv_uris = set()

        for p in prop.inverse:
            inv_uris.add(p.iri)

        return inv_uris


    def getClasses(self):
        return self.getOntology().classes()

    def getDataProperties(self):
        return self.getOntology().data_properties()

    def getObjectProperties(self):
        return self.getOntology().object_properties()

    def getIndividuals(self):
        return self.getOntology().individuals()


    #Ontology graph representation (from RDFlib). Not to confuse with projection graph
    def getGraph(self):
        return self.graph



    def queryGraph(self, query):

        #query_owlready()

        #results = self.graph.query("""SELECT ?s ?p ?o WHERE { ?s ?p ?o . }""")
        results = self.graph.query(query)


        return list(results)


        #print(r)
        #for r in results:
        #    print(r.labels)
        #    print(r[0])




class DBpediaOntology(OntologyAccess):

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__(self.getOntologyIRI())


    def getOntologyIRI(self):
        return "http://www.cs.ox.ac.uk/isg/ontologies/dbpedia.owl"


    def getAncestorsURIs(self,cls):
        ancestors_str = set()

        for anc_cls in cls.ancestors():
            ancestors_str.add(anc_cls.iri)

        agent = "http://dbpedia.org/ontology/Agent"
        if agent in ancestors_str:
            ancestors_str.remove(agent)

        return ancestors_str


class SchemaOrgOntology(OntologyAccess):

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__(self.getOntologyIRI())


    def getOntologyIRI(self):
        return "http://www.cs.ox.ac.uk/isg/ontologies/schema.org.owl"




if __name__ == '__main__':

    uri_onto="http://www.cs.ox.ac.uk/isg/ontologies/dbpedia.owl"
    #uri_onto="http://www.cs.ox.ac.uk/isg/ontologies/schema.org.owl"


    #onto_access = DBpediaOntology()
    #onto_access = SchemaOrgOntology()
    onto_access = OntologyAccess(uri_onto)
    onto_access.loadOntology(True) #Classify True



    query = """SELECT ?s ?p ?o WHERE { ?s ?p ?o . }"""

    results = onto_access.queryGraph(query)

    for r in results:
        print(r)






