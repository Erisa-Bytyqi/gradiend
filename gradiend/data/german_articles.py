import logging
import random
import re
from operator import index

import pandas as pd
from datasets import load_dataset
from statsmodels.tsa.statespace.tools import set_mode
from tqdm.auto import tqdm
import json
import spacy


logger = logging.getLogger(__name__)
nlp = spacy.load("de_core_news_sm")
nlp.add_pipe("sentencizer")

def load_german_wiki_data(dataset_name,local, split=None, chunk_size=None, skip_size=None, num_repeat=None, total_lines=None, dataset_part=None, output_dir= None):
    logger.warning("Loading the wiki data")
    german_texts = []
    if local:
        german_texts = load_dataset("json", data_files=dataset_name)
    else:

        wiki_dataset = load_dataset(dataset_name, dataset_part, streaming=True, split=split, trust_remote_code=True)
        assert num_repeat * (skip_size + chunk_size) < total_lines

        iterator = iter(wiki_dataset)

        for _ in tqdm(range(num_repeat)):
            german_texts.extend([next(iterator)["text"] for _ in range(chunk_size)])
            # skip some lines
            [next(iterator) for _ in range(skip_size)]

        # save the wiki texts
        save_file_w_json(german_texts, output_dir)

    logger.warning("Texts loaded")
    return german_texts

#TODO: refactor this so it work with setences only.
def filter_wiki_texts(inputs, article, case, pos,  output_dir, mask, num_samples = 20000):
    """
    :param inputs: the dataset containing the texts.
    :param article: one of the german articles e.g. der.
    :param case: the case of the article, e.g. NOM, should correspond to the spaCy case notation.
    :param pos:  the part of speech of the article, e.g. DET, should correspond tho the spaCy pos notation.
    :param output_dir: path to where the filtered sentences should be saved.
    :param num_samples the number of samples to keep from the filtered sentences.
    """

    article = article.lower() # ensure that the article is in lowercase
    article_regex = rf"\b{article}\b"
    regex_pattern = re.compile(article_regex, re.IGNORECASE)

    logger.warning('creating wiki sentences')
    wiki_sentences = create_wiki_sentences(inputs)
    filtered_sentences = []

    progress = tqdm(total=len(wiki_sentences), desc="Filtering sentences")
    step = 0

    for sent in wiki_sentences:
        step += 1
        if step % 1000 == 0:
            progress.update(1000)

        if regex_pattern.match(sent.lower()):
            doc = nlp(sent)
            article_tokens = [token for token in doc if token.text.lower() == article and token.pos_ == pos]
            if len(article_tokens) < 5 and all(token.morph.get("Case") == [case] for token in article_tokens):
                filtered_sentences.append(sent)

    logger.warning(f"Filtered {len(filtered_sentences)} sentences for article: {article}.")
    if (len(filtered_sentences) > num_samples):
        filtered_sentences = random.sample(filtered_sentences, num_samples)

    create_filtered_set(filtered_sentences, article.upper() ,mask, output_dir)

    # finally add a NER, any sentences with more than 3 names -> removed, check how often this is the case


def filter_article(sentences, article, cases, pos, mask, output_dir, gender, dataset_label,number,num_samples = 10000):
    article = article.lower()  # ensure that the article is in lowercase
    article_regex = rf"\b{article}\b"
    regex_pattern = re.compile(article_regex, re.IGNORECASE)

    filtered_sentences = []

    progress = tqdm(total=len(sentences), desc="Filtering sentences")
    step = 0

    for sent in sentences:
        step += 1
        if step % 1000 == 0:
            progress.update(1000)

        if len(sent) < 50 or len(sent) > 500:
            continue

        if regex_pattern.search(sent.lower()):
            doc = nlp(sent)
            if len(doc.ents) > 3 :continue
            else:
                article_tokens = [token for token in doc if token.text.lower() == article and token.pos_ == pos]
                if 0 < len(article_tokens) < 5:
                    if all(token.morph.get("Case") in ([case] for case in cases) for token in article_tokens) and all(token.morph.get('Gender') == gender for token in article_tokens) and all(token.morph.get('Number') == number for token in article_tokens):
                        filtered_sentences.append(sent)

    logger.warning(f"Filtered {len(filtered_sentences)} sentences for article: {article}.")
    if len(filtered_sentences) > num_samples:
        random.seed(42)
        filtered_sentences = random.sample(filtered_sentences, num_samples)

    create_filtered_set(filtered_sentences, article.upper() ,mask, output_dir, dataset_label)




def create_wiki_sentences(wiki_texts):
    wiki_sentences = []

    for entry in tqdm(wiki_texts['train']):
        sentences = [sent.text.strip() for sent in nlp(entry.get("text")).sents]
        wiki_sentences.extend(sentences)

    #TODO: change to info -> config has to be changed so it actually prints 
    logger.warning(f"{len(wiki_sentences)} sentences were created.")
    return wiki_sentences



def save_file_w_json(input, output_dir):
    with open(output_dir, "a", encoding="utf-8" ) as f:
        for entry in input:
            json.dump({"text": entry}, f, ensure_ascii=False)
            f.write("\n")


def create_filtered_set(inputs,label, mask, output_dir, dataset_label):
    article = label.lower()  # ensure that the article is in lowercase
    article_regex = rf"\b{re.escape(article)}\b"
    regex_pattern = re.compile(article_regex, re.IGNORECASE)


    columns = ['id', 'text', 'masked', 'label', 'token_count', 'dataset_label']
    row_id = 0
    rows = []
    for sent in inputs:
        sent_dict = {}
        row_id += 1

        token_count = len(regex_pattern.findall(sent))
        masked_sentence = re.sub(article_regex, mask, sent, flags=re.IGNORECASE)

        print(masked_sentence)
        sent_dict.update({
            'id': row_id,
            'text': sent.replace("\n"," "),
            'masked': masked_sentence.replace("\n"," "),
            'label': label,
            'token_count': token_count,
            'dataset_label': dataset_label
        })
        rows.append(sent_dict)

    filtered_df = pd.DataFrame(rows, columns=columns)
    filtered_df.to_csv(output_dir, index=False)


def filter_ner(sentence):
    sentence_nlp = nlp(sentence)
    if len(sentence_nlp.ents) > 3:
        return False
    else:
        return True


#import threading

if __name__ == '__main__':
    wiki_sents_path = ""
    sentences = load_dataset("json", data_files=wiki_sents_path)["train"]["text"]
    #TODO change the 'masks' to match the case
    thread = filter_article(sentences, 'dem', ['Dat'], 'DET', '[DEM_ARTICLE]', "gradiend/data/der_die_das/splits/DN/filtered_DN.csv", gender=['Neut'], dataset_label ='DN', number=['Sing'])
    
    # die_thread = threading.Thread(target=filter_article, args=(sentences, 'die', ['Nom', 'Akk'], 'DET', '[DIE_ARTICLE]', "./der_die_das/filtered_die.csv"))
    # das_thread = threading.Thread(target=filter_article, args=(sentences, 'das', ['Nom', 'Akk'], 'DET', '[DAS_ARTICLE]', "./der_die_das/filtered_das.csv"))
    # die_thread.start()
    # das_thread.start()
    #combine_datasets("gradiend/data/der_die_das/filtered_der.csv", "gradiend/data/der_die_das/filtered_die.csv")