# so somehow build different sentences where the locus of
# ungrammaticality is a single word orrr compare two whole sentences, no more masking..
# would allow to observe the changes that transpire with the adjective and nouns (in the case of M to F)

# if a sentence was 'der netter Autor' -> 'die nettE AutorIN' but for that I do need an adjectvice so adj-noun agreement can be osberved,more specifically it makes 
#plausible for the two sentences to co-exist' 

#Open questions: 
    # is the obeserved change only for the article? 
    # is the observerd change article + adjective inflection? 

from nltk import grammar, parse
from nltk.parse.generate import generate
import pandas as pd
import spacy


nlp = spacy.load("de_core_news_sm")

# This doesnt work...
# from DERBI.derbi import DERBI
# derbi = DERBI(nlp)

NounPhrase_MF = grammar.CFG.fromstring("""
% start S 
S -> NP_DEF '.' | NP_INDEF '.'
NP_DEF -> ART_DEF_m ADJ_DEF_m NN_m | ART_DEF_f ADJ_DEF_f NN_f 
NP_INDEF -> ART_INDEF_m ADJ_INDEF_m NN_m | ART_INDEF_f ADJ_DEF_f NN_f 
ART_DEF_m -> 'Der' | 'Dieser'
ADJ_DEF_m -> 'dunkle' | 'helle' | 'runde' 
ART_INDEF_m -> 'Ein' | 'Mein'
ADJ_INDEF_m -> 'dunkler' | 'heller' | 'runder' 
ART_DEF_f -> 'Die' | 'Diese'
ADJ_DEF_f -> 'dunkle' | 'helle' | 'schwarze'
ART_INDEF_f -> 'Eine' | 'Meine'
NN_m -> 'Raum' | 'Ton' | 'Fleck' | 'Klang'
NN_f -> 'Wolke'| 'Farbe' | 'Hose' | 'Jacke' 
""")

counters = {
    "Case=Nom|Definite=Def|Gender=Masc|Number=Sing|PronType=Art": 'Die',
    "Case=Nom|Definite=Def|Gender=Fem|Number=Sing|PronType=Art" : 'Der',
    "Case=Nom|Definite=Ind|Gender=Masc|Number=Sing|PronType=Art" : 'Eine',
    "Case=Nom|Definite=Ind|Gender=Fem|Number=Sing|PronType=Art" : 'Ein',
    "Case=Nom|Gender=Masc|Number=Sing|PronType=Dem": 'Diese',
    "Case=Nom|Gender=Fem|Number=Sing|PronType=Dem": "Dieser",
    "Case=Nom|Gender=Fem|Number=Sing|Poss=Yes|PronType=Prs": "Mein",
    "Case=Nom|Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs": "Meine"
}


def create_simple_sents(): 
    result = []
    for sentence in generate(NounPhrase_MF):
        candidate = {}
        masked_sentence = sentence[:]
        counter_sentence = sentence[:]
        masked_sentence[0] = '[ARTICLE]'
        og_sentence = (' ').join(sentence)
  
        candidate["text"] = og_sentence
        candidate["masked"] = (' ').join(masked_sentence)
    
        doc = nlp(og_sentence)

        if doc[0] and counters.get(str(doc[0].morph)):
            label = doc[0].morph.get('Gender')
            pron_type = doc[0].morph.get('PronType')
            counter_sentence[0] = counters[str(doc[0].morph)]
        
        else:
            continue

        candidate = {
            "text": og_sentence,
            "masked": (' ').join(masked_sentence),
            "cf": (' ').join(counter_sentence),
            "label": label,
            "pron_type": pron_type
         }

        result.append(candidate)


    cfg_result = pd.DataFrame(result)

    cfg_result.to_csv("NP_cfg.csv")


