import os
import numpy as np
import re
import string

def readfiles(dir):
    pwd = os.getcwd()
    os.chdir(dir)

    files = os.listdir('.')
    files_text = []
    count = 0
    limit = 0
    for i in files:
        count += 1
        if(count == limit):
            break
        try:
            f = open(i, 'r')
            files_text.append(f.read())
        except:
            print("Could not read %s." % i)
            continue
        finally:
            f.close()
        
    os.chdir(pwd)

    return files_text

_regex = re.compile('^[{0}]+|[{0}]+$'.format(string.punctuation))
_stopwords_file = open('stopwords', 'r')
_stopwords = set(_stopwords_file.read().split())
_stopwords_file.close()

def extract_words(s):
    s = s.lower()
    wordlist = [w for w in _regex.sub(' ', s).split() if w not in _stopwords]

    return wordlist


class BagOfWordsFeatureExtractor(object):
    def __init__(self, min_freq = 20):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.min_freq = min_freq

    def preprocess(self, documents):
        word_freq = {}
        for doc in documents:
            for w in extract_words(doc):
                if w not in word_freq:
                    word_freq[w] = 1
                else:
                    word_freq[w] += 1

        remove_words = set()
        for w in word_freq.keys():
            if word_freq[w] < self.min_freq:
                remove_words.add(w)
        for w in remove_words:
            del word_freq[w]
        for idx, word in enumerate(word_freq.keys()):
            self.idx_to_word[idx] = word
            self.word_to_idx[word] = idx

    def extract(self, documents):
        features = np.zeros((len(documents), len(self.idx_to_word)))
        if len(self.word_to_idx) == 0 or len(self.idx_to_word) == 0:
            raise Exception("Dictionary not initialised.")

        for idx, doc in enumerate(documents):
            words = extract_words(doc)
            for w in words:
                if w in self.word_to_idx:
                    features[idx][self.word_to_idx[w]] += 1
            features[idx] /= len(words)

        return features
