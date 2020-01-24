from flask import Flask, jsonify, request
from gensim.summarization.summarizer import summarize

import spacy
import nltk
import json
import configparser

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)


def summary_extraction(data):
    url = data['source_url']
    process_id = data['process_id']
    transcript_path = data['diarized_data']['response']['results']
    transcript_text = []
    for i in range(len(transcript_path)):
        try:
            transcript_text.append(transcript_path[i]['alternatives'][0]['transcript'])
        except:
            print([i, url])
    joined_transcript_text = " ".join(transcript_text)
    summary = summarize(joined_transcript_text)
    # remove newlines
    summary = summary.replace('\n', '')
    return url, joined_transcript_text, summary, process_id


def map_summaries(summary_list):
    mapped_summaries = []
    for summary_data in summary_list:
        mapped_summaries.append({
            'process_id':summary_data[3],
            'source_url': summary_data[0],
            'text': summary_data[1],
            'summary': summary_data[2]})
    return mapped_summaries


def consolidated_summary(json_file_data):
    data_ = json_file_data
    ls = []
    for i in range(len(data_['data'])):
        try:
            ls.append(summary_extraction(data_['data'][i]))
        except:
            print(i)
    # send back the json object for summary
    mapped_json_response = map_summaries(ls)
    return mapped_json_response


@app.route('/summarize', methods=['POST'])
def summarize_text():
    json_data = request.get_json()
    results_ls = {}
    results_ls['data'] = consolidated_summary(json_data)
    return results_ls


# functions for extractor
def get_text_from_json_new(data_dict_ls):
    text_ls = []
    for data_dict in data_dict_ls:
        for key, value in data_dict.items():
            if key == "alternatives":
                text_ls.append(value[0]["transcript"])
    text = '.'.join(text_ls)
    return text


def create_info_dataframe(json_file):
    rec_ls = []
    for i in range(len(json_file["data"])):
        try:
            rec_columns = ["title", "text"]
            rec_df = pd.DataFrame(index=[0], columns=rec_columns)
            rec_df = rec_df.fillna(0)
            text = ""
            data_dict_ls = json_file["data"][i]["diarized_data"]["response"]["results"]
            text = get_text_from_json_new(data_dict_ls)
            word = json_file["data"][i]["source_url"]
            rec_df["title"] = word
            rec_df["text"] = text
            rec_ls.append(rec_df)
        except NameError:
            print(NameError)
            pass
    big_rec_df = pd.concat(rec_ls).reset_index(drop=True)
    return big_rec_df


def clean_text(text): #cleantext
    """
    Custom function to clean enriched text

    :param text(str): Enriched ytext from Ekstep Content.
    :returns: Cleaned text.
    """
    replace_char = [
        "[",
        "]",
        "u'",
        "None",
        "Thank you",
        "-",
        "(",
        ")",
        "#",
        "Done",
        ">",
        "<",
        "-",
        "|",
        "/",
        "\"",
        "Hint",
        "\n",
        "'"]
    for l in replace_char:
        text = text.replace(l, "")
    text = re.sub(' +', ' ', text)
    return text


def tokenize(text, tokenizer=nltk.word_tokenize): #clean_text_tokens
    """
    A custom preprocessor to tokenise and clean a text.
    Used in Content enrichment pipeline.

    Process:

    * tokenise string using nltk word_tokenize()

    * Remove stopwords

    * Remove punctuations in words

    * Remove digits and whitespaces

    * Convert all words to lowercase

    * Remove words of length

    * Remove nan or empty string

    :param text(str): The string to be tokenised.
    :returns: List of cleaned tokenised words.
    """
    tokens = tokenizer(text)
    tokens = [token for token in tokens if token.lower() not in stopwords]
    # tokens = [token for token in tokens if token not in pun_list]
    tokens = [re.sub(r'[0-9\.\W_]', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]
    tokens = [token for token in tokens if token]
    return tokens


def stem_lem(keyword_list, DELIMITTER):
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    keyword_list = [item for item in keyword_list]
    keyword_list = [i.split(DELIMITTER) for i in keyword_list]
    lemma_ls_1 = [[wordnet_lemmatizer.lemmatize(
        item, pos="n") for item in words] for words in keyword_list]
    lemma_ls_2 = [[wordnet_lemmatizer.lemmatize(
        item, pos="v") for item in words] for words in lemma_ls_1]
    lemma_ls_3 = [[wordnet_lemmatizer.lemmatize(
        item, pos="a") for item in words] for words in lemma_ls_2]
    lemma_ls_4 = [[wordnet_lemmatizer.lemmatize(
        item, pos="r") for item in words] for words in lemma_ls_3]
    stemm_ls = [[stemmer.stem(item) for item in words] for words in lemma_ls_4]
    return [DELIMITTER.join(i) for i in stemm_ls]


def get_combined_summary(json_file_data):
    combined_summary = []
    for data_obj in json_file_data['data']:
        combined_summary.append(data_obj['transcript']['combined_transcript'])
    combined_summary = ''.join(combined_summary)
    return combined_summary


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def get_keywords(tfidf_transformer, cv, agg_corpus_df, index, feature_names, no_of_keys):
    tf_idf_vector = tfidf_transformer.transform(cv.transform([agg_corpus_df["agg_text"][index]]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, no_of_keys)
    return keywords


def print_results(tfidf_transformer, cv, agg_corpus_df, feature_names, no_of_keys):
    df_ls = []
    for i in range(len(agg_corpus_df)):
        keyword_df_columns = ["topic", "text", "keywords_generated"]
        keyword_df = pd.DataFrame(index=[0], columns=keyword_df_columns)
        keyword_df = keyword_df.fillna(0)
        keyword_df["topic"] = agg_corpus_df["topic"].iloc[i]
        keyword_df['text'] = agg_corpus_df["agg_text"].iloc[i]
        keyword_df["keywords_generated"] = ", ".join(
            k for k, v in get_keywords(tfidf_transformer, cv, agg_corpus_df, i, feature_names, no_of_keys).items())
        df_ls.append(keyword_df)
    result_df = pd.concat(df_ls).reset_index(drop=True)
    return result_df


def phrase_detection(text, keywords_list, no_of_sen):
    main_ls = []
    for key in keywords_list:
        sent_ls = []
        for sent in nltk.sent_tokenize(text):
            if key in sent:
                sent_ls.append(sent)
        #no_of_sen is the number of sentences to be displayed for a keyword, configurable
        main_ls.append({key: sent_ls[:no_of_sen]})
    return main_ls


@app.route('/extractor', methods=['POST'])
def extract_keywords_and_phrases():
    df_ls = []
    agg_ls = []

    json_data = json.loads(request.data)
    try:
        df = create_info_dataframe(json_data)
        topic = 'topic_name'
        df["topic"] = topic
        df['combined_summary'] = get_combined_summary(json_data)
        df_ls.append(df)
    except NameError:
        print(NameError)
        pass
    corpus_df = pd.concat(df_ls).reset_index(drop=True)
    corpus_df['text'] = corpus_df['text'].apply(lambda x: x.lower())
    for key, val in corpus_df.groupby(['topic']).groups.items():
        agg_columns = ["topic", "agg_text"]
        agg_df = pd.DataFrame(index=[0], columns=agg_columns)
        agg_df = agg_df.fillna(0)
        text = " "
        agg_df["topic"] = key
        for i in val:
            text += corpus_df["text"].iloc[i]
        agg_df["agg_text"] = text
        agg_ls.append(agg_df)
    agg_full_df = pd.concat(agg_ls).reset_index(drop=True)

    stop = stopwords.words('english')
    docs = agg_full_df["agg_text"]  # list of all text
    cv = CountVectorizer(max_df=13, stop_words=stop)
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names()
    doc = agg_full_df["agg_text"].iloc[0]

    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n, here it is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    result_df = print_results(tfidf_transformer, cv, agg_full_df, feature_names, 10)
    phrase_output = phrase_detection(result_df["text"].iloc[0], result_df["keywords_generated"].iloc[0].split(", "), 2)
    return jsonify({'keywords': keywords, 'phrases': phrase_output})


app.run(debug=True)
