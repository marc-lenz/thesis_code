# -*- coding: utf-8 -*-
"""

Part of: Thesis Project - Learning interlingual document representations 
(Marc Lenz, 2021)
---

This File contains helper function used for the Thesis. 

"""

from collections import defaultdict
from gensim import corpora
import csv

def extract_docs_from_jrq_xml(tree, 
                              src_language = "english",
                              trg_language = "french",
                              align="sections",
                              doc_limit = 1000):
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
    doc_count = 0
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
                    doc_count += 1
                    if doc_count >= doc_limit:
                        return documents
                
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
                    doc_count += 1
                    if doc_count >= doc_limit:
                        return documents
                

    return documents


def filter_docs(l1_docs, l2_docs, min_len=1, max_len=10000):
    filtered_l1_docs = []
    filtered_l2_docs = []
    for k in range(len(l1_docs)):
        l1_doc = l1_docs[k]
        l2_doc = l2_docs[k]
        word_num_l1_doc = len(l1_doc)
        word_num_l2_doc = len(l2_doc)
        min_condition = word_num_l1_doc >= min_len and word_num_l2_doc >= min_len
        max_condition = word_num_l1_doc <= max_len and word_num_l2_doc <= max_len 
        if min_condition and max_condition:
            filtered_l1_docs.append(l1_doc)
            filtered_l2_docs.append(l2_doc)
    return filtered_l1_docs, filtered_l2_docs


def save_docs(l1_docs, l2_docs, l1_destination, l2_destination):
    # open a file, where you ant to store the data
    with open(l1_destination, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for doc in l1_docs:
            spamwriter.writerow(doc)
    
    with open(l2_destination, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for doc in l2_docs:
            spamwriter.writerow(doc)

def read_docs(file_name1, file_name2):
    fd = []
    with open(file_name1, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            fd.append(row)
    
    ed = []
    with open(file_name2, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            ed.append(row)
    return fd, ed


def create_corpus(texts, filter_extremes=True):
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
    if filter_extremes == True:
        dictionary.filter_extremes(no_below=5, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return dictionary, corpus
