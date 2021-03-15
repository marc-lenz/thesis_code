# -*- coding: utf-8 -*-
"""
General Code, reading and preprocessing
"""
#import classes and functions written for this code
from Preprocessor import Preprocessor
from Utils import extract_docs_from_jrq_xml, filter_docs, save_docs, read_docs
#import general used packages
import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm

# parse an xml file by name
tree = ET.parse('Data/alignedCorpus-en-fr.xml')
documents = extract_docs_from_jrq_xml(tree, align="documents", doc_limit=5000)

# Make sure no empty documents are passed, filter those
filtered_docs = []
for doc in documents:
    if len(doc["english"]) > 0 and len(doc["french"]) > 0:
        filtered_docs.append(doc)
        
#preprocessing, text -> tokenized and stemmed/lemmatized words       
preprocessor_english = Preprocessor(language="en")
preprocessor_french = Preprocessor(language="fr")
print("Preprocessing of Documents, language1")
french_docs_preprocessed = [preprocessor_french.preprocess(doc["french"]) for doc in tqdm(filtered_docs)]
print("Preprocessing of Documents, language2")
english_docs_preprocessed = [preprocessor_english.preprocess(doc["english"]) for doc in tqdm(filtered_docs)]

#Now filter extreme short and long docs
filtered_en_docs, filtered_fr_docs = filter_docs(english_docs_preprocessed,
                                                 french_docs_preprocessed, 
                                                 min_len=10, max_len=10000)

save_docs(filtered_en_docs, filtered_fr_docs, 'Data/jrq_aligned_5000_fr.csv', 'Data/jrq_aligned_5000_en.csv')          
