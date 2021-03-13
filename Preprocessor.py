# -*- coding: utf-8 -*-
"""
Part of: Thesis Project - Learning interlingual document representations 
(Marc Lenz, 2021)
---
Class containing different preprocessor functions for NLP
"""
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import FrenchStemmer

class Preprocessor:
    def __init__(self, language = "en"):
        self.language = language
        self.set_settings(language)
        
    def set_settings(self, language):
        #Language Specifics
        if language == "en":
            self.tokenize = lambda x: word_tokenize(x, language="english")
            self.stem_word = WordNetLemmatizer().lemmatize
        if language == "fr":
            self.tokenize = lambda x: word_tokenize(x, language="french")
            self.stem_word = FrenchStemmer().stem
            
    def preprocess(self, text):
        filtered_text = re.sub(r"[^a-zA-Z0-9]+", ' ', text.lower())
        tokens = self.tokenize(filtered_text)
        stemmed_tokens = [self.stem_word(token) for token in tokens]
        return stemmed_tokens
        
        
    
    