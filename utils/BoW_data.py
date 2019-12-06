import numpy as np
import pandas as pd

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as cv
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

import gensim


def load_dataframe(train_or_test='train'):
    """Loads dataframe with resnet features vectors (as a list in one column), descriptions (as a list in one column), and tags"""

    df = pd.DataFrame(get_resnet_features(train_or_test=train_or_test))

    df['descriptions'] = get_descriptions(df, train_or_test=train_or_test)
    df['tags'] = get_tags(df, train_or_test=train_or_test)
    df['word_list'] = get_word_list(df, train_or_test=train_or_test)
    classes, df['word_vector'] = get_word_vec(df, train_or_test=train_or_test)

    return classes, df


def get_resnet_features(train_or_test='train'):
    """Gets image_id -> 1000 dim feature vector"""

    fp = f'data/features_{train_or_test}/features_resnet1000_{train_or_test}.csv'
    lines = []

    with open(fp) as f:
        line = f.readline()
        while line:
            line = line.split(',')
            lines.append({"image_file": line[0], "resnet_vector": np.array(
                [np.double(x.strip()) for x in line[1:]])})
            line = f.readline()

    return lines


def get_resnet_intermediate_features(train_or_test='train'):
    """Gets image_id -> 2048 dim intermediate feature vector"""

    fp = f'data/features_{train_or_test}/features_resnet1000intermediate_{train_or_test}.csv'
    lines = []

    with open(fp) as f:
        line = f.readline()
        while line:
            line = line.split(',')
            lines.append({"image_file": line[0], "resnet_vector": np.array(
                [np.double(x.strip()) for x in line[1:]])})
            line = f.readline()

    return lines


"""
Use get_descriptions() to add new column to dataframe with list of descriptions provided for image
"""


def open_description(fp):
    with open(fp) as f:
        descriptions = [x.strip() for x in f.readlines()]

    return descriptions


def get_descriptions(df, train_or_test='train', imfile_column='image_file'):
    """Descriptions are independant lists of line by line description"""
    return df[imfile_column].apply(
        lambda x: open_description(
            f'data/descriptions_{train_or_test}/' + x.split('/')[-1].replace('jpg', 'txt'))
    )


"""
Use get_tags() to add a new column to dataframe with tags as one whole phrase - e.g. sports:tennis
"""


def extract_tags_to_list(fp):

    with open(fp) as f:
        tags = f.readlines()

    return [tag.strip() for tag in tags]


def get_tags(df, train_or_test='train', imfile_column='image_file'):
    """Returns pandas series of image tags"""

    return df[imfile_column].apply(
        lambda x: extract_tags_to_list(
            f'data/tags_{train_or_test}/' +
            x.split('/')[-1].replace('jpg', 'txt')
        )
    )


"""
Use get_tags_split() to add two new columns to dataframe with tags split by hierarchy - e.g. [sports, tennis]
"""


def extract_tags_to_list_split(fp):

    with open(fp) as f:
        tags = f.readlines()

    tag_pairs = [tag.strip().split(':') for tag in tags]

    higher_cat = [x[0] for x in tag_pairs]
    lower_cat = [x[0] for x in tag_pairs]

    return higher_cat, lower_cat


def get_tags_split(df, train_or_test='train', imfile_column='image_file'):
    """Returns two pandas series of image tags, higher and lower category"""

    return zip(*df[imfile_column].map(
        lambda x: extract_tags_to_list_split(
            f'data/tags_{train_or_test}/' +
            x.split('/')[-1].replace('jpg', 'txt')
        )
    ))


def to_word_list(s):
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there',
                  'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
                  'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
                  'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who',
                  'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below',
                  'are', 'we', 'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their',
                  'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all',
                  'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
                  'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
                  'what', 'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he',
                  'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if',
                  'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
                        'further', 'was', 'here', 'than'}
    lower = s[0].lower()
    tokens = tokenizer.tokenize(lower)
    tokens = [w for w in tokens if not w in stop_words]
    return [ps.stem(t) for t in tokens]


def get_word_list(df, train_or_test='train'):
    return df['descriptions'].apply(to_word_list)


def get_word_vec(df, train_or_test='train'):
    mlb = MultiLabelBinarizer()
    wordVecs = mlb.fit_transform(df['word_list'])
    wordVecs = [[e] for e in wordVecs]

    return mlb.classes_, pd.DataFrame(wordVecs)
