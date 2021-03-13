# -*- coding: utf-8 -*-
"""

Part of: Thesis Project - Learning interlingual document representations 
(Marc Lenz, 2021)
---

This File contains helper function used for the Thesis. 

"""

from collections import defaultdict
from gensim import corpora


def extract_docs_from_jrq_xml(tree, 
                              src_language = "english",
                              trg_language = "french",
                              align="sections"):
    """
    
    Takes an XML element tree of JRQ-Arquis Corpus and creates a list of 
    aligned paragraphs. 
    
    Use like that: 
        1. get aligned JRQ-Arquis corpus data (XML)
        2. load via import xml.etree.ElementTree as ET, tree = ET.parse(<file>)
        3. use this function on tree
    
    parameters - align = "sections" or "documents" describes if you want to obtain
    a list of aligned sections or aligned documents
    language

    """
    root = tree.getroot()
    
    documents = []
    language_keys = {"s1": src_language, "s2": trg_language}
    
    #Get aligned sections
    if align == "sections":
        for elem in root.iter("linkGrp"):
            sections = []
            for link in elem.iter("link"):
                #Get aligned sections
    
                    section = dict()      
                    for e in link.iter():
                        if e.tag in language_keys.keys():
                            language_key = language_keys[e.tag]
                            section_content = e.text
                            section[language_key] = section_content
                    sections.append(section)
            if len(sections) > 0:
                    documents.append(sections)
                
    #Get aligned documents
    if align == "documents":
        for elem in root.iter("linkGrp"):
            doc = {src_language: "", trg_language: ""}
            for link in elem.iter("link"):
                #Get aligned sections   
                    for e in link.iter():
                        if e.tag in language_keys.keys():
                            language_key = language_keys[e.tag]
                            section_content = e.text
                            if section_content != None:
                                doc[language_key] = doc[language_key] + str(section_content)             
            if len(doc) > 0:
                    documents.append(doc)
                

    return documents

def create_corpus(texts):
    """
    
    Function to create a corpus file out of a list of texts to prepare
    data for model training using Gensim

    """
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return dictionary, corpus
