import nltk
from nltk.book import *
from nltk.tokenize import word_tokenize
import numpy
from statistics import mode
from spellchecker import SpellChecker

spell = SpellChecker()

txt = text1[1:100000] + text2[1:100000] + text3[1:100000] + \
      text4[1:100000] + text5[1:100000] + text6[1:100000] + \
      text7[1:100000] + text8[1:100000] + text9[1:100000]
# collecting data from nltk book
count = 0  # counter
common = []  # all words after our input
word = []  # 3 common words after our input
sentence = []  # our sentence list
# find those words that may be misspelled
while 1:
    a = str(input(">"))  # input from user
    if " " in a:
        t = word_tokenize(a)
        a = t[len(t) - 1]
        misspelled = spell.unknown([a])
        for X in misspelled:
            print(spell.candidates(X))
            list0 = spell.candidates(X)
            cand = str(input("Choose"))
            if cand in list0:
                a = cand
    else:
        misspelled = spell.unknown([a])
        for X in misspelled:
            print(spell.candidates(X))
            list0 = spell.candidates(X)
            cand = str(input("Choose"))
            if cand in list0:
                a = cand
        t = word_tokenize(a)  # speared words from if we use a sentence
        txt += t  # we add new words to olur data
    for i in range(len(txt)):  # search in all data texts
        if a.lower() == txt[i].lower():  # control point to match words
            count += 1  # counter goes to upper bound
            if count > 3:  # counter upper bound
                count = 0  # counter reset
                break
            else:  # if we haven't reached to bound yet
                common.append(txt[i + 1])  # we add next word that is for input
                word.append(mode(common))  # mode function is for most common words
                common.remove(mode(common))
                # when we found the most common word we should delete it to find second common word

            txt.append(a)
            continue
        else:
            txt.append(a)
            continue
    print(word)  # Our Most common three words
    choose = str(input("Choose your word: "))  # to choose word press key between {1-3}
    if choose == "1":
        print(word[0])
        sentence.append(word[0])
        word.clear()
    elif choose == "2":
        print(word[1])
        sentence.append(word[1])
        word.clear()
    elif choose == "3":
        print(word[2])
        sentence.append(word[2])
        word.clear()
    else:
        print(sentence)
        sentence.clear()
