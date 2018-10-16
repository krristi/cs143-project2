#!/usr/bin/env python

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse
import json
import sys
import bz2

__author__ = ""
__email__ = ""

# Some useful data.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.
def notEmpty(word):
    if(word == ''):
        return False
    else:
        return True

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """
    # YOUR CODE GOES BELOW:

    #1.Replace new lines and tab characters with a single space.
    clean = text.replace("\t"," ")
    clean = clean.replace("\n"," ")
    clean = clean.replace("\\n"," ")
    clean = clean.replace("\\t"," ")

    #2.Remove URLs.
    clean = re.sub("\[([^\]]*)\]\([^\)]*\)","\g<1>", clean)
    clean = re.sub("\[{0,1}([a-zA-Z]*)\]{0,1}[\s]*((?:https\:\/\/|http\:\/\/|www\.)[^\s]*)","\g<1>", clean)
    #3.Split text on a single space, remove empty tokens
    clean = clean.split(" ")
    #boo = list(filter(notEmpty,clean))
    while True:
        try:
            clean.remove('')
        except:
            break

    #4.Separate all external punctuation such as periods, commas, etc. into their own tokens
    # . , ? ! ` ' " : ; - ... -- ---
    #                   \u2026 \u2010-2015
    noPre = []
    for word in clean:
        foo = re.sub("^([^a-zA-Z0-9_#$%']+)([a-zA-Z0-9_#$%']+)",
                    "\g<1> \g<2>",
                    word)
        foo = foo.split(" ")
        for sep in foo:
            noPre.append(sep)

    noPost = []
    for word in noPre:
        foo = re.sub("([a-zA-Z0-9_#$%']+)([^a-zA-Z0-9_#$%']+)$",
                    "\g<1> \g<2>",
                    word)
        foo = foo.split(" ")
        for sep in foo:
            noPost.append(sep)
        


    #5.Remove all punctuation except punctuation that ends a phrase or sentence.
    #,:;!?.
    onlyEnd = []
    for word in noPost:
        is_punc = re.search("^[^a-zA-Z0-9_#$%']+$", word)
        if is_punc:
            word = re.sub("[^.!?,;:]+",'',word)
            word = list(word)
            onlyEnd.extend(word)
        else:
            onlyEnd.append(word)

    while True:
        try:
            onlyEnd.remove('')
        except:
            break

    #6.Convert all text to lowercase.
    done = []
    for word in onlyEnd:
        done.append(word.lower())

    parsed_text = ""
    for word in done:
        parsed_text += (" "+word)

    #trigrams and such
    phrases = []
    phrases.append([])
    i_phrase = 0
    for word in done:
        is_punc = re.search("^[^a-zA-Z0-9_#$%']+$", word)
        if is_punc:
            phrases.append([])
            i_phrase += 1
        else:
            phrases[i_phrase].append(word)

    while True:
        try:
            phrases.remove([])
        except:
            break

    unigrams = ""
    bigrams = ""
    trigrams = ""
    for phrase in phrases:
        for word in phrase:
            unigrams += (" "+word)
        for i in range(len(phrase)-1):
            bigrams += (" "+phrase[i]+"_"+phrase[i+1])
        for i in range(len(phrase)-2):
            trigrams +=(" "+phrase[i]+"_"+phrase[i+1]+"_"+phrase[i+2])

    parsed_text = parsed_text.lstrip(" ")
    unigrams = unigrams.lstrip(" ")
    bigrams = bigrams.lstrip(" ")
    trigrams = trigrams.lstrip(" ")

    
    #print(parsed_text)
    #print(unigrams)
    #print(bigrams)
    #print(trigrams)
    

    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.
    try:
        f_name = sys.argv[1]
    except:
        print("Usage: python3 cleantext.py <filename>")
        sys.exit()

    if f_name.endswith(".json.bz2"):
        is_json = True
        try:
            f = bz2.BZ2File(f_name, "r")
        except:
            print("Could not read file:", f_name)
            sys.exit()
    else:
        is_json = False
        try:
            f = open(f_name, 'r')
        except:
            print("Could not read file:", f_name)
            sys.exit()

    f_lines = f.readlines()
    
    for line in f_lines:
        if is_json:
            curr_line = line.decode()
        else:
            curr_line = line
        loaded = json.loads(curr_line)
        processed = sanitize(loaded['body'])
        print(processed)





