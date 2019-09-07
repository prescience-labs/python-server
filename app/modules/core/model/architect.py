import math
from os import PathLike
from pathlib import Path
from typing import Union
import csv
import json
from enum import Enum
from json import JSONEncoder
import os
from os import path, remove, makedirs
import subprocess
from collections import Counter
import sys
import argparse
import gzip
import io
import posixpath
import re
import zipfile
from typing import List, Tuple
import spacy

import requests
from tqdm import tqdm

################################## Data Types ##################################

class LexiconElement(object):
    def __init__(self, term: list, score: str or float = None, polarity: str = None,
                 is_acquired: str = None, position: str = None):
        self.term = term
        self.polarity = polarity
        try:
            self.score = float(score)
        except TypeError:
            self.score = 0
        self.position = position
        if is_acquired == "N":
            self.is_acquired = False
        elif is_acquired == "Y":
            self.is_acquired = True
        else:
            self.is_acquired = None

    def __lt__(self, other):
        return self.term[0] < other.term[0]

    def __le__(self, other):
        return self.term[0] <= other.term[0]

    def __eq__(self, other):
        return self.term[0] == other.term[0]

    def __ne__(self, other):
        return self.term[0] != other.term[0]

    def __gt__(self, other):
        return self.term[0] > other.term[0]

    def __ge__(self, other):
        return self.term[0] >= other.term[0]


class TermType(Enum):
    OPINION = 'OP'
    ASPECT = 'AS'
    NEGATION = 'NEG'
    INTENSIFIER = 'INT'


class Polarity(Enum):
    POS = 'POS'
    NEG = 'NEG'
    UNK = 'UNK'


class Term(object):
    def __init__(self, text: str, kind: TermType, polarity: Polarity, score: float, start: int,
                 length: int):
        self._text = text
        self._type = kind
        self._polarity = polarity
        self._score = score
        self._start = start
        self._len = length

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def text(self):
        return self._text

    @property
    def type(self):
        return self._type

    @property
    def polarity(self):
        return self._polarity

    @property
    def score(self):
        return self._score

    @property
    def start(self):
        return self._start

    @property
    def len(self):
        return self._len

    @text.setter
    def text(self, val):
        self._text = val

    @score.setter
    def score(self, val):
        self._score = val

    @polarity.setter
    def polarity(self, val):
        self._polarity = val

    def __str__(self):
        return "text: " + self._text + " type: " + str(self._type) + " pol: " + \
               str(self._polarity) + " score: " + str(self._score) + " start: " + \
               str(self._start) + " len: " + \
               str(self._len)


class SentimentDoc(object):
    def __init__(self, doc_text: str = None, sentences: list = None):
        if sentences is None:
            sentences = []
        self._doc_text = doc_text
        self._sentences = sentences

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def doc_text(self):
        return self._doc_text

    @doc_text.setter
    def doc_text(self, val):
        self._doc_text = val

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, val):
        self._sentences = val

    @staticmethod
    def decoder(obj):
        """
        :param obj: object to be decoded
        :return: decoded Sentence object
        """
        # SentimentDoc
        if '_doc_text' in obj and '_sentences' in obj:
            return SentimentDoc(obj['_doc_text'], obj['_sentences'])

        # SentimentSentence
        if all((attr in obj for attr in ('_start', '_end', '_events'))):
            return SentimentSentence(obj['_start'], obj['_end'], obj['_events'])

        # Term
        if all(attr in obj for attr in
               ('_text', '_type', '_score', '_polarity', '_start', '_len')):
            return Term(obj['_text'], TermType[obj['_type']],
                        Polarity[obj['_polarity']], obj['_score'], obj['_start'],
                        obj['_len'])
        return obj

    def __repr__(self):
        return self.pretty_json()

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self.sentences.__iter__()

    def __len__(self):
        return len(self.sentences)

    def json(self):
        """
        Return json representations of the object
        Returns:
            :obj:`json`: json representations of the object
        """
        return json.dumps(self, cls=SentimentDocEncoder)

    def pretty_json(self):
        """
        Return pretty json representations of the object
        Returns:
            :obj:`json`: pretty json representations of the object
        """
        return json.dumps(self, cls=SentimentDocEncoder, indent=4)


class SentimentSentence(object):
    def __init__(self, start: int, end: int, events: list):
        self._start = start
        self._end = end
        self._events = events

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, val):
        self._start = val

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, val):
        self._end = val

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, val):
        self._events = val


################################## BISTParser ##################################

# DONE # from nlp_architect.common.core_nlp_doc import CoreNLPDoc
# DONE # from nlp_architect.data.conll import ConllEntry
# DONE # from nlp_architect.models.bist_parser import BISTModel
# DONE # from nlp_architect.models.bist import utils (.vocab .write_conll .run_eval)
# DONE # from nlp_architect.models.bist.utils import get_options_dict
# DONE # from nlp_architect.models.bist.eval.conllu.conll17_ud_eval import run_conllu_eval
# DONE # from nlp_architect.utils.io import validate, validate_existing_filepath
# DONE # from nlp_architect import LIBRARY_OUT
# DONE # from nlp_architect.utils.io import download_unlicensed_file, uncompress_file
# DONE # from nlp_architect.utils.io import validate
# from nlp_architect.utils.text import SpacyInstance


class Vocabulary:
    """
    A vocabulary that maps words to ints (storing a vocabulary)
    """

    def __init__(self, start=0):

        self._vocab = {}
        self._rev_vocab = {}
        self.next = start

    def add(self, word):
        """
        Add word to vocabulary
        Args:
            word (str): word to add
        Returns:
            int: id of added word
        """
        if word not in self._vocab.keys():
            self._vocab[word] = self.next
            self._rev_vocab[self.next] = word
            self.next += 1
        return self._vocab.get(word)

    def word_id(self, word):
        """
        Get the word_id of given word
        Args:
            word (str): word from vocabulary
        Returns:
            int: int id of word
        """
        return self._vocab.get(word, None)

    def __getitem__(self, item):
        """
        Get the word_id of given word (same as `word_id`)
        """
        return self.word_id(item)

    def __len__(self):
        return len(self._vocab)

    def __iter__(self):
        for word in self.vocab.keys():
            yield word

    @property
    def max(self):
        return self.next

    def id_to_word(self, wid):
        """
        Word-id to word (string)
        Args:
            wid (int): word id
        Returns:
            str: string of given word id
        """
        return self._rev_vocab.get(wid)

    @property
    def vocab(self):
        """
        dict: get the dict object of the vocabulary
        """
        return self._vocab

    def add_vocab_offset(self, offset):
        """
        Adds an offset to the ints of the vocabulary
        Args:
            offset (int): an int offset
        """
        new_vocab = {}
        for k, v in self.vocab.items():
            new_vocab[k] = v + offset
        self.next += offset
        self._vocab = new_vocab
        self._rev_vocab = {v: k for k, v in new_vocab.items()}

    def reverse_vocab(self):
        """
        Return the vocabulary as a reversed dict object
        Returns:
            dict: reversed vocabulary object
        """
        return self._rev_vocab


def try_to_load_spacy(model_name):
    try:
        spacy.load(model_name)
        return True
    except OSError:
        return False


class SpacyInstance:
    """
    Spacy pipeline wrapper which prompts user for model download authorization.
    Args:
        model (str, optional): spacy model name (default: english small model)
        disable (list of string, optional): pipeline annotators to disable
            (default: [])
        display_prompt (bool, optional): flag to display/skip license prompt
    """

    def __init__(self, model='en', disable=None, display_prompt=True):
        if disable is None:
            disable = []
        try:
            self._parser = spacy.load(model, disable=disable)
        except OSError:
            url = 'https://spacy.io/models'
            if display_prompt and license_prompt('Spacy {} model'.format(model), url) is False:
                sys.exit(0)
            spacy_download(model)
            self._parser = spacy.load(model, disable=disable)

    @property
    def parser(self):
        """return Spacy's instance parser"""
        return self._parser

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a sentence into tokens
        Args:
            text (str): text to tokenize
        Returns:
            list: a list of str tokens of input
        """
        # pylint: disable=not-callable

        return [t.text for t in self.parser(text)]


stemmer = EnglishStemmer()
lemmatizer = WordNetLemmatizer()
spacy_lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
p = re.compile(r'[ \-,;.@&_]')


class Stopwords(object):
    """
    Stop words list class.
    """
    stop_words = []

    @staticmethod
    def get_words():
        if not Stopwords.stop_words:
            sw_path = path.join(path.dirname(path.realpath(__file__)),
                                'resources',
                                'stopwords.txt')
            with open(sw_path) as fp:
                stop_words = []
                for w in fp:
                    stop_words.append(w.strip().lower())
            Stopwords.stop_words = stop_words
        return Stopwords.stop_words


def simple_normalizer(text):
    """
    Simple text normalizer. Runs each token of a phrase thru wordnet lemmatizer
    and a stemmer.
    """
    if not str(text).isupper() or \
            not str(text).endswith('S') or \
            not len(text.split()) == 1:
        tokens = list(filter(lambda x: len(x) != 0, p.split(text.strip())))
        text = ' '.join([stemmer.stem(lemmatizer.lemmatize(t))
                         for t in tokens])
    return text


def spacy_normalizer(text, lemma=None):
    """
    Simple text normalizer using spacy lemmatizer. Runs each token of a phrase
    thru a lemmatizer and a stemmer.
    Arguments:
        text(string): the text to normalize.
        lemma(string): lemma of the given text. in this case only stemmer will
        run.
    """
    if not str(text).isupper() or \
            not str(text).endswith('S') or \
            not len(text.split()) == 1:
        tokens = list(filter(lambda x: len(x) != 0, p.split(text.strip())))
        if lemma:
            lemma = lemma.split(' ')
            text = ' '.join([stemmer.stem(l)
                             for l in lemma])
        else:
            text = ' '.join([stemmer.stem(spacy_lemmatizer(t, u'NOUN')[0])
                             for t in tokens])
    return text


def read_sequential_tagging_file(file_path, ignore_line_patterns=None):
    """
    Read a tab separated sequential tagging file.
    Returns a list of list of tuple of tags (sentences, words)
    Args:
        file_path (str): input file path
        ignore_line_patterns (list, optional): list of string patterns to ignore
    Returns:
        list of list of tuples
    """
    if ignore_line_patterns:
        assert isinstance(ignore_line_patterns, list), 'ignore_line_patterns must be a list'

    def _split_into_sentences(file_lines):
        sentences = []
        s = []
        for line in file_lines:
            if len(line) == 0:
                sentences.append(s)
                s = []
                continue
            s.append(line)
        if len(s) > 0:
            sentences.append(s)
        return sentences

    with open(file_path, encoding='utf-8') as fp:
        data = fp.readlines()
        data = [d.strip() for d in data]
        if ignore_line_patterns:
            for s in ignore_line_patterns:
                data = [d for d in data if s not in d]
        data = [tuple(d.split()) for d in data]
    return _split_into_sentences(data)


def word_vector_generator(data, lower=False, start=0):
    """
    Word vector generator util.
    Transforms a list of sentences into numpy int vectors and returns the
    constructed vocabulary
    Arguments:
        data (list): list of list of strings
        lower (bool, optional): transform strings into lower case
        start (int, optional): vocabulary index start integer
    Returns:
        2D numpy array and Vocabulary of the detected words
    """
    vocab = Vocabulary(start)
    data_vec = []
    for sentence in data:
        sentence_vec = []
        for w in sentence:
            word = w
            if lower:
                word = word.lower()
            wid = vocab[word]
            if wid is None:
                wid = vocab.add(word)
            sentence_vec.append(wid)
        data_vec.append(sentence_vec)
    return data_vec, vocab


def character_vector_generator(data, start=0):
    """
    Character word vector generator util.
    Transforms a list of sentences into numpy int vectors of the characters
    of the words of the sentence, and returns the constructed vocabulary
    Arguments:
        data (list): list of list of strings
        start (int, optional): vocabulary index start integer
    Returns:
        np.array: a 2D numpy array
        Vocabulary: constructed vocabulary
    """
    vocab = Vocabulary(start)
    data_vec = []
    for sentence in data:
        sentence_vec = []
        for w in sentence:
            word_vec = []
            for char in w:
                cid = vocab[char]
                if cid is None:
                    cid = vocab.add(char)
                word_vec.append(cid)
            sentence_vec.append(word_vec)
        data_vec.append(sentence_vec)
    return data_vec, vocab


def extract_nps(annotation_list, text=None):
    """
    Extract Noun Phrases from given text tokens and phrase annotations.
    Returns a list of tuples with start/end indexes.
    Args:
        annotation_list (list): a list of annotation tags in str
        text (list, optional): a list of token texts in str
    Returns:
        list of start/end markers of noun phrases, if text is provided a list of noun phrase texts
    """
    np_starts = [i for i in range(len(annotation_list)) if annotation_list[i] == 'B-NP']
    np_markers = []
    for s in np_starts:
        i = 1
        while s + i < len(annotation_list) and annotation_list[s + i] == 'I-NP':
            i += 1
        np_markers.append((s, s + i))
    return_markers = np_markers
    if text:
        assert len(text) == len(annotation_list), 'annotations/text length mismatch'
        return_markers = [' '.join(text[s:e]) for s, e in np_markers]
    return return_markers


def bio_to_spans(text: List[str], tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert BIO tagged list of strings into span starts and ends
    Args:
        text: list of words
        tags: list of tags
    Returns:
        tuple: list of start, end and tag of detected spans
    """
    pointer = 0
    starts = []
    for i, t, in enumerate(tags):
        if t.startswith('B-'):
            starts.append((i, pointer))
        pointer += len(text[i]) + 1

    spans = []
    for s_i, s_char in starts:
        label_str = tags[s_i][2:]
        e = 0
        e_char = len(text[s_i + e])
        while len(tags) > s_i + e + 1 and tags[s_i + e + 1].startswith('I-'):
            e += 1
            e_char += 1 + len(text[s_i + e])
        spans.append((s_char, s_char + e_char, label_str))
    return spans


def _download_pretrained_model():
    """Downloads the pre-trained BIST model if non-existent."""
    if not path.isfile(SpacyBISTParser.dir / 'bist.model'):
        print('Downloading pre-trained BIST model...')
        zip_path = SpacyBISTParser.dir / 'bist-pretrained.zip'
        makedirs(SpacyBISTParser.dir, exist_ok=True)
        download_unlicensed_file(
            'https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/dep_parse/',
            'bist-pretrained.zip', zip_path)
        print('Unzipping...')
        uncompress_file(zip_path, outpath=str(SpacyBISTParser.dir))
        remove(zip_path)
        print('Done.')


def vocab(conll_path):
    # pylint: disable=missing-docstring
    words_count = Counter()
    pos_count = Counter()
    rel_count = Counter()

    for sentence in read_conll(conll_path):
        words_count.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
        pos_count.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
        rel_count.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return words_count, {w: i for i, w in enumerate(words_count.keys())}, list(
        pos_count.keys()), list(rel_count.keys())


def read_conll(path):
    """Yields CoNLL sentences read from CoNLL formatted file.."""
    with open(path, 'r') as conll_fp:
        root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_',
                          -1, 'rroot', '_', '_')
        tokens = [root]
        for line in conll_fp:
            stripped_line = line.strip()
            tok = stripped_line.split('\t')
            if not tok or line.strip() == '':
                if len(tokens) > 1:
                    yield tokens
                tokens = [root]
            else:
                if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                    # noinspection PyTypeChecker
                    tokens.append(stripped_line)
                else:
                    tokens.append(
                        ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3],
                                   tok[5],
                                   int(tok[6]) if tok[6] != '_' else -1,
                                   tok[7], tok[8], tok[9]))
        if len(tokens) > 1:
            yield tokens


def run_eval(gold, test):
    """Evaluates a set of predictions using the appropriate script."""
    if is_conllu(gold):
        run_conllu_eval(gold_file=gold, test_file=test)
    else:
        eval_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval', 'eval.pl')
        with open(test[:test.rindex('.')] + '_eval.txt', 'w') as out_file:
            subprocess.run(['perl', eval_script, '-g', gold, '-s', test], stdout=out_file)


def is_conllu(path):
    """Determines if the file is in CoNLL-U format."""
    return os.path.splitext(path.lower())[1] == '.conllu'


def get_options_dict(activation, lstm_dims, lstm_layers, pos_dims):
    """Generates dictionary with all parser options."""
    return {'activation': activation, 'lstm_dims': lstm_dims, 'lstm_layers': lstm_layers,
            'pembedding_dims': pos_dims, 'wembedding_dims': 100, 'rembedding_dims': 25,
            'hidden_units': 100, 'hidden2_units': 0, 'learning_rate': 0.1, 'blstmFlag': True,
            'labelsFlag': True, 'bibiFlag': True, 'costaugFlag': True, 'seed': 0, 'mem': 0}


# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = list(range(10))

WEIGHTS = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights.clas')

# UD Error is used when raising exceptions in this module
class UDError(Exception):
    pass


# Load given CoNLL-U file into internal representation
def load_conllu(file):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # Internal representation classes
    class UDRepresentation:
        # pylint: disable=too-few-public-methods
        def __init__(self):
            # Characters of all the tokens in the whole file.
            # Whitespace between tokens is not included.
            self.characters = []
            # List of UDSpan instances with start&end indices into `characters`
            self.tokens = []
            # List of UDWord instances.
            self.words = []
            # List of UDSpan instances with start&end indices into `characters`
            self.sentences = []

    class UDSpan:
        # pylint: disable=too-few-public-methods
        def __init__(self, start, end):
            self.start = start
            # Note that self.end marks the first position **after the end** of
            # span, so we can use characters[start:end] or range(start, end).
            self.end = end

    class UDWord:
        # pylint: disable=too-few-public-methods
        def __init__(self, span, columns, is_multiword):
            # Span of this word (or MWT, see below) within
            # ud_representation.characters.
            self.span = span
            # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
            self.columns = columns
            # is_multiword==True means that this word is part of a multi-word
            # token.
            # In that case, self.span marks the span of the whole multi-word
            # token.
            self.is_multiword = is_multiword
            # Reference to the UDWord instance representing the HEAD (or None
            # if root).
            self.parent = None
            # Let's ignore language-specific deprel subtypes.
            self.columns[DEPREL] = columns[DEPREL].split(':')[0]

    ud = UDRepresentation()

    # Load the CoNLL-U file
    index, sentence_start = 0, None
    while True:
        line = file.readline()
        if not line:
            break
        line = line.rstrip("\r\n")

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:
            # Add parent UDWord links and check there are no cycles
            def process_word(word):
                if word.parent == "remapping":
                    raise UDError("There is a cycle in a sentence")
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head > len(ud.words) - sentence_start:
                        raise UDError(
                            "HEAD '{}' points outside of the sentence".format(
                                word.columns[HEAD]))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent

            for word in ud.words[sentence_start:]:
                process_word(word)

            # Check there is a single root node
            if len([word for word in ud.words[sentence_start:] if
                    word.parent is None]) != 1:
                raise UDError("There are multiple roots in a sentence")

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line.split("\t")
        if len(columns) != 10:
            raise UDError(
                "The CoNLL-U line does not contain 10 tab-separated columns: "
                "'{}'".format(line))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM  so gold.characters == system.characters
        # even if one of them tokenizes the space.
        columns[FORM] = columns[FORM].replace(" ", "")
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            try:
                start, end = list(map(int, columns[ID].split("-")))
            except Exception:
                raise UDError("Cannot parse multi-word token ID '{}'".format(
                    columns[ID]))

            for _ in range(start, end + 1):
                word_line = file.readline().rstrip("\r\n")
                word_columns = word_line.split("\t")
                if len(word_columns) != 10:
                    raise UDError(
                        "The CoNLL-U line does not contain 10 tab-separated "
                        "columns: '{}'".format(word_line))
                ud.words.append(
                    UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except Exception:
                raise UDError("Cannot parse word ID '{}'".format(columns[ID]))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected"
                              " '{}'".format(columns[ID], columns[FORM],
                                             len(ud.words)
                                             - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except Exception:
                raise UDError("Cannot parse HEAD '{}'".format(columns[HEAD]))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")

    return ud


# Evaluate the gold and system treebanks (loaded using load_conllu).
def evaluate(gold_ud, system_ud, deprel_weights=None):
    # pylint: disable=too-many-locals
    class Score:
        # pylint: disable=too-few-public-methods
        def __init__(self, gold_total, system_total, correct,
                     aligned_total=None):
            self.precision = correct / system_total if system_total else 0.0
            self.recall = correct / gold_total if gold_total else 0.0
            self.f1 = 2 * correct / (system_total + gold_total) \
                if system_total + gold_total else 0.0
            self.aligned_accuracy = \
                correct / aligned_total if aligned_total else aligned_total

    class AlignmentWord:
        # pylint: disable=too-few-public-methods
        def __init__(self, gold_word, system_word):
            self.gold_word = gold_word
            self.system_word = system_word
            self.gold_parent = None
            self.system_parent_gold_aligned = None

    class Alignment:
        def __init__(self, gold_words, system_words):
            self.gold_words = gold_words
            self.system_words = system_words
            self.matched_words = []
            self.matched_words_map = {}

        def append_aligned_words(self, gold_word, system_word):
            self.matched_words.append(AlignmentWord(gold_word, system_word))
            self.matched_words_map[system_word] = gold_word

        def fill_parents(self):
            # We represent root parents in both gold and system data by '0'.
            # For gold data, we represent non-root parent by corresponding gold
            # word.
            # For system data, we represent non-root parent by either gold word
            # aligned
            # to parent system nodes, or by None if no gold words is aligned to
            # the parent.
            for words in self.matched_words:
                words.gold_parent = words.gold_word.parent if \
                    words.gold_word.parent is not None else 0
                words.system_parent_gold_aligned = self.matched_words_map.get(
                    words.system_word.parent, None) \
                    if words.system_word.parent is not None else 0

    def lower(text):
        if sys.version_info < (3, 0) and isinstance(text, str):
            return text.decode("utf-8").lower()
        return text.lower()

    def spans_score(gold_spans, system_spans):
        correct, gi, si = 0, 0, 0
        while gi < len(gold_spans) and si < len(system_spans):
            if system_spans[si].start < gold_spans[gi].start:
                si += 1
            elif gold_spans[gi].start < system_spans[si].start:
                gi += 1
            else:
                correct += gold_spans[gi].end == system_spans[si].end
                si += 1
                gi += 1

        return Score(len(gold_spans), len(system_spans), correct)

    def alignment_score(alignment, key_fn, weight_fn=lambda w: 1):
        gold, system, aligned, correct = 0, 0, 0, 0

        for word in alignment.gold_words:
            gold += weight_fn(word)

        for word in alignment.system_words:
            system += weight_fn(word)

        for words in alignment.matched_words:
            aligned += weight_fn(words.gold_word)

        if key_fn is None:
            # Return score for whole aligned words
            return Score(gold, system, aligned)

        for words in alignment.matched_words:
            if key_fn(words.gold_word, words.gold_parent) == key_fn(
                    words.system_word,
                    words.system_parent_gold_aligned):
                correct += weight_fn(words.gold_word)

        return Score(gold, system, correct, aligned)

    def beyond_end(words, i, multiword_span_end):
        if i >= len(words):
            return True
        if words[i].is_multiword:
            return words[i].span.start >= multiword_span_end
        return words[i].span.end > multiword_span_end

    def extend_end(word, multiword_span_end):
        if word.is_multiword and word.span.end > multiword_span_end:
            return word.span.end
        return multiword_span_end

    def find_multiword_span(gold_words, system_words, gi, si):
        # We know gold_words[gi].is_multiword or system_words[si].is_multiword.
        # Find the start of the multiword span (gs, ss), so the multiword span
        # is minimal.
        # Initialize multiword_span_end characters index.
        if gold_words[gi].is_multiword:
            multiword_span_end = gold_words[gi].span.end
            if not system_words[si].is_multiword and system_words[si].span.start < \
                    gold_words[gi].span.start:
                si += 1
        else:  # if system_words[si].is_multiword
            multiword_span_end = system_words[si].span.end
            if not gold_words[gi].is_multiword and gold_words[gi].span.start < \
                    system_words[si].span.start:
                gi += 1
        gs, ss = gi, si

        # Find the end of the multiword span
        # (so both gi and si are pointing to the word following the multiword
        # span end).
        while not beyond_end(gold_words, gi, multiword_span_end) or \
                not beyond_end(system_words, si, multiword_span_end):
            gold_start = gold_words[gi].span.start
            sys_start = system_words[si].span.start
            if gi < len(gold_words) and (si >= len(system_words) or gold_start <= sys_start):
                multiword_span_end = extend_end(gold_words[gi], multiword_span_end)
                gi += 1
            else:
                multiword_span_end = extend_end(system_words[si], multiword_span_end)
                si += 1
        return gs, ss, gi, si

    def compute_lcs(gold_words, system_words, gi, si, gs, ss):
        # pylint: disable=too-many-arguments
        lcs = [[0] * (si - ss) for _ in range(gi - gs)]
        for g in reversed(list(range(gi - gs))):
            for s in reversed(list(range(si - ss))):
                if lower(gold_words[gs + g].columns[FORM]) == lower(
                        system_words[ss + s].columns[FORM]):
                    lcs[g][s] = 1 + (lcs[g + 1][s + 1] if
                                     g + 1 < gi - gs and s + 1 < si - ss
                                     else 0)
                lcs[g][s] = max(lcs[g][s],
                                lcs[g + 1][s] if g + 1 < gi - gs else 0)
                lcs[g][s] = max(lcs[g][s],
                                lcs[g][s + 1] if s + 1 < si - ss else 0)
        return lcs

    def align_words(gold_words, system_words):
        alignment = Alignment(gold_words, system_words)

        gi, si = 0, 0
        while gi < len(gold_words) and si < len(system_words):
            if gold_words[gi].is_multiword or system_words[si].is_multiword:
                # A: Multi-word tokens => align via LCS within the whole
                # "multiword span".
                gs, ss, gi, si = find_multiword_span(gold_words, system_words,
                                                     gi, si)

                if si > ss and gi > gs:
                    lcs = compute_lcs(gold_words, system_words, gi, si, gs, ss)

                    # Store aligned words
                    s, g = 0, 0
                    while g < gi - gs and s < si - ss:
                        if lower(gold_words[gs + g].columns[FORM]) == lower(
                                system_words[ss + s].columns[FORM]):
                            alignment.append_aligned_words(gold_words[gs + g],
                                                           system_words[
                                                               ss + s])
                            g += 1
                            s += 1
                        elif lcs[g][s] == (
                                lcs[g + 1][s] if g + 1 < gi - gs else 0):
                            g += 1
                        else:
                            s += 1
            else:
                # B: No multi-word token => align according to spans.
                if (gold_words[gi].span.start, gold_words[gi].span.end) == (
                        system_words[si].span.start,
                        system_words[si].span.end):
                    alignment.append_aligned_words(gold_words[gi],
                                                   system_words[si])
                    gi += 1
                    si += 1
                elif gold_words[gi].span.start <= system_words[si].span.start:
                    gi += 1
                else:
                    si += 1

        alignment.fill_parents()

        return alignment

    # Check that underlying character sequences do match
    if gold_ud.characters != system_ud.characters:
        index = 0
        while gold_ud.characters[index] == system_ud.characters[index]:
            index += 1

        raise UDError(
            "The concatenation of tokens in gold file and in system file "
            "differ!\n"
            + "First 20 differing characters in gold file: '{}' and system file:"
            " '{}'".format(
                "".join(gold_ud.characters[index:index + 20]),
                "".join(system_ud.characters[index:index + 20])
            )
        )

    # Align words
    alignment = align_words(gold_ud.words, system_ud.words)

    # Compute the F1-scores
    result = {
        "Tokens": spans_score(gold_ud.tokens, system_ud.tokens),
        "Sentences": spans_score(gold_ud.sentences, system_ud.sentences),
        "Words": alignment_score(alignment, None),
        "UPOS": alignment_score(alignment, lambda w, parent: w.columns[UPOS]),
        "XPOS": alignment_score(alignment, lambda w, parent: w.columns[XPOS]),
        "Feats": alignment_score(alignment,
                                 lambda w, parent: w.columns[FEATS]),
        "AllTags": alignment_score(alignment, lambda w, parent: (
            w.columns[UPOS], w.columns[XPOS], w.columns[FEATS])),
        "Lemmas": alignment_score(alignment,
                                  lambda w, parent: w.columns[LEMMA]),
        "UAS": alignment_score(alignment, lambda w, parent: parent),
        "LAS": alignment_score(alignment,
                               lambda w, parent: (parent, w.columns[DEPREL])),
    }

    # Add WeightedLAS if weights are given
    if deprel_weights is not None:
        def weighted_las(word):
            return deprel_weights.get(word.columns[DEPREL], 1.0)

        result["WeightedLAS"] = alignment_score(alignment, lambda w, parent: (
            parent, w.columns[DEPREL]), weighted_las)

    return result


def load_deprel_weights(weights_file):
    if weights_file is None:
        return None

    deprel_weights = {}
    with open(weights_file) as f:
        for line in f:
            # Ignore comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            columns = line.rstrip("\r\n").split()
            if len(columns) != 2:
                raise ValueError(
                    "Expected two columns in the UD Relations weights file on line"
                    " '{}'".format(
                        line))

            deprel_weights[columns[0]] = float(columns[1])

    return deprel_weights


def load_conllu_file(path):
    with open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {})) \
            as _file:
        return load_conllu(_file)


def evaluate_wrapper(gold_file: str, system_file: str, weights_file: str):
    # Load CoNLL-U files
    gold_ud = load_conllu_file(gold_file)
    system_ud = load_conllu_file(system_file)

    # Load weights if requested
    deprel_weights = load_deprel_weights(weights_file)

    return evaluate(gold_ud, system_ud, deprel_weights)


def run_conllu_eval(gold_file, test_file, weights_file=WEIGHTS, verbose=True):
    # Use verbose if weights are supplied
    if weights_file is not None and not verbose:
        verbose = True

    # Evaluate
    evaluation = evaluate_wrapper(gold_file, test_file, weights_file)

    # Write the evaluation to file
    with open(test_file[:test_file.rindex('.')] + '_eval.txt', 'w') as out_file:
        if not verbose:
            out_file.write("LAS F1 Score: {:.2f}".format(100 * evaluation["LAS"].f1) + '\n')
        else:
            metrics = ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "Feats",
                       "AllTags", "Lemmas", "UAS", "LAS"]
            if weights_file is not None:
                metrics.append("WeightedLAS")

            out_file.write("Metrics    | Precision |    Recall |  F1 Score | AligndAcc" + '\n')
            out_file.write("-----------+-----------+-----------+-----------+-----------" + '\n')
            for metric in metrics:
                out_file.write("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                    metric,
                    100 * evaluation[metric].precision,
                    100 * evaluation[metric].recall,
                    100 * evaluation[metric].f1,
                    "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy)
                    if evaluation[metric].aligned_accuracy is not None else ""
                ) + '\n')


def download_unlicensed_file(url, sourcefile, destfile, totalsz=None):
    """
    Download the file specified by the given URL.
    Args:
        url (str): url to download from
        sourcefile (str): file to download from url
        destfile (str): save path
        totalsz (:obj:`int`, optional): total size of file
    """
    req = requests.get(posixpath.join(url, sourcefile),
                       stream=True)

    chunksz = 1024 ** 2
    if totalsz is None:
        if "Content-length" in req.headers:
            totalsz = int(req.headers["Content-length"])
            nchunks = totalsz // chunksz
        else:
            print("Unable to determine total file size.")
            nchunks = None
    else:
        nchunks = totalsz // chunksz

    print("Downloading file to: {}".format(destfile))
    with open(destfile, 'wb') as f:
        for data in tqdm(req.iter_content(chunksz), total=nchunks, unit="MB"):
            f.write(data)
    print("Download Complete")


def uncompress_file(filepath, outpath='.'):
    """
    Unzip a file to the same location of filepath
    uses decompressing algorithm by file extension
    Args:
        filepath (str): path to file
        outpath (str): path to extract to
    """
    if filepath.endswith('.zip'):
        z = zipfile.ZipFile(filepath, 'r')
        z.extractall(outpath)
        z.close()
    elif filepath.endswith('.gz'):
        if os.path.isdir(outpath):
            raise ValueError('output path for gzip must be a file')
        with gzip.open(filepath, 'rb') as fp:
            file_content = fp.read()
        with open(outpath, 'wb') as fp:
            fp.write(file_content)
    else:
        raise ValueError('Unsupported archive provided. Method supports only .zip/.gz files.')


################################## Inference  ##################################

INTENSIFIER_FACTOR = 0.3
VERB_POS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
NUMBER_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


class CoreNLPDoc(object):
    """Object for core-components (POS, Dependency Relations, etc).
    Attributes:
        _doc_text: the doc text
        _sentences: list of sentences, each word in a sentence is
            represented by a dictionary, structured as follows: {'start': (int), 'len': (int),
            'pos': (str), 'ner': (str), 'lemma': (str), 'gov': (int), 'rel': (str)}
    """
    def __init__(self, doc_text: str = '', sentences: list = None):
        if sentences is None:
            sentences = []
        self._doc_text = doc_text
        self._sentences = sentences

    @property
    def doc_text(self):
        return self._doc_text

    @doc_text.setter
    def doc_text(self, val):
        self._doc_text = val

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, val):
        self._sentences = val

    @staticmethod
    def decoder(obj):
        if '_doc_text' in obj and '_sentences' in obj:
            return CoreNLPDoc(obj['_doc_text'], obj['_sentences'])
        return obj

    def __repr__(self):
        return self.pretty_json()

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self.sentences.__iter__()

    def __len__(self):
        return len(self.sentences)

    def json(self):
        """Returns json representations of the object."""
        return json.dumps(self.__dict__)

    def pretty_json(self):
        """Returns pretty json representations of the object."""
        return json.dumps(self.__dict__, indent=4)

    def sent_text(self, i):
        parsed_sent = self.sentences[i]
        first_tok, last_tok = parsed_sent[0], parsed_sent[-1]
        return self.doc_text[first_tok['start']: last_tok['start'] + last_tok['len']]

    def sent_iter(self):
        for parsed_sent in self.sentences:
            first_tok, last_tok = parsed_sent[0], parsed_sent[-1]
            sent_text = self.doc_text[first_tok['start']: last_tok['start'] + last_tok['len']]
            yield sent_text, parsed_sent

    def brat_doc(self):
        """Returns doc adapted to BRAT expected input."""
        doc = {'text': '', 'entities': [], 'relations': []}
        tok_count = 0
        rel_count = 1
        for sentence in self.sentences:
            sentence_start = sentence[0]['start']
            sentence_end = sentence[-1]['start'] + sentence[-1]['len']
            doc['text'] = doc['text'] + '\n' + self.doc_text[sentence_start:sentence_end]
            token_offset = tok_count

            for token in sentence:
                start = token['start']
                end = start + token['len']
                doc['entities'].append(['T' + str(tok_count), token['pos'], [[start, end]]])

                if token['gov'] != -1 and token['rel'] != 'punct':
                    doc['relations'].append(
                        [rel_count, token['rel'], [['', 'T' + str(token_offset + token['gov'])],
                                                   ['', 'T' + str(tok_count)]]])
                    rel_count += 1
                tok_count += 1
        doc['text'] = doc['text'][1:]
        return doc

    def displacy_doc(self):
        """Return doc adapted to displacyENT expected input."""
        doc = []
        for sentence in self.sentences:
            sentence_doc = {'arcs': [], 'words': []}
            # Merge punctuation:
            merged_punct_sentence = merge_punctuation(sentence)
            fix_gov_indexes(merged_punct_sentence, sentence)
            for tok_index, token in enumerate(merged_punct_sentence):
                sentence_doc['words'].append({'text': token['text'], 'tag': token['pos']})
                dep_tok = tok_index
                gov_tok = token['gov']
                direction = 'left'
                arc_start = dep_tok
                arc_end = gov_tok
                if dep_tok > gov_tok:
                    direction = 'right'
                    arc_start = gov_tok
                    arc_end = dep_tok
                if token['gov'] != -1 and token['rel'] != 'punct':
                    sentence_doc['arcs'].append({'dir': direction, 'label': token['rel'],
                                                 'start': arc_start, 'end': arc_end})
            doc.append(sentence_doc)
        return doc


def merge_punct_tok(merged_punct_sentence, last_merged_punct_index, punct_text, is_traverse):
    # merge the text of the punct tok
    if is_traverse:
        merged_punct_sentence[last_merged_punct_index]["text"] = \
            punct_text + merged_punct_sentence[last_merged_punct_index]["text"]
    else:
        merged_punct_sentence[last_merged_punct_index]["text"] = \
            merged_punct_sentence[last_merged_punct_index]["text"] + punct_text


def find_correct_index(orig_gov, merged_punct_sentence):
    for tok_index, tok in enumerate(merged_punct_sentence):
        if tok["start"] == orig_gov["start"] and tok["len"] == orig_gov["len"] and tok["pos"] == \
                orig_gov["pos"] and tok["text"] == orig_gov["text"]:
            return tok_index
    return None


def fix_gov_indexes(merged_punct_sentence, sentence):
    for merged_token in merged_punct_sentence:
        tok_gov = merged_token['gov']
        if tok_gov == -1:  # gov is root
            merged_token['gov'] = -1
        else:
            orig_gov = sentence[tok_gov]
            correct_index = find_correct_index(orig_gov, merged_punct_sentence)
            merged_token['gov'] = correct_index


def merge_punctuation(sentence):
    merged_punct_sentence = []
    tmp_punct_text = None
    punct_text = None
    last_merged_punct_index = -1
    for tok_index, token in enumerate(sentence):
        if token['rel'] == 'punct':
            punct_text = token["text"]
            if tok_index < 1:  # this is the first tok - append to the next token
                tmp_punct_text = punct_text
            else:  # append to the previous token
                merge_punct_tok(merged_punct_sentence, last_merged_punct_index, punct_text,
                                False)
        else:
            merged_punct_sentence.append(token)
            last_merged_punct_index = last_merged_punct_index + 1
            if tmp_punct_text is not None:
                merge_punct_tok(merged_punct_sentence, last_merged_punct_index, punct_text,
                                True)
                tmp_punct_text = None
    return merged_punct_sentence


def validate(*args):
    """
    Validate all arguments are of correct type and in correct range.
    Args:
        *args (tuple of tuples): Each tuple represents an argument validation like so:
        Option 1 - With range check:
            (arg, class, min_val, max_val)
        Option 2 - Without range check:
            (arg, class)
        If class is a tuple of type objects check if arg is an instance of any of the types.
        To allow a None valued argument, include type(None) in class.
        To disable lower or upper bound check, set min_val or max_val to None, respectively.
        If arg has the len attribute (such as string), range checks are performed on its length.
    """
    for arg in args:
        arg_val = arg[0]
        arg_type = (arg[1],) if isinstance(arg[1], type) else arg[1]
        if not isinstance(arg_val, arg_type):
            raise TypeError('Expected type {}'.format(' or '.join([t.__name__ for t in arg_type])))
        if arg_val is not None and len(arg) >= 4:
            name = 'of ' + arg[4] if len(arg) == 5 else ''
            arg_min = arg[2]
            arg_max = arg[3]
            if hasattr(arg_val, '__len__'):
                val = 'Length'
                num = len(arg_val)
            else:
                val = 'Value'
                num = arg_val
            if arg_min is not None and num < arg_min:
                raise ValueError('{} {} must be greater or equal to {}'.format(val, name, arg_min))
            if arg_max is not None and num >= arg_max:
                raise ValueError('{} {} must be less than {}'.format(val, name, arg_max))


def validate_existing_filepath(arg):
    """Validates an input argument is a path string to an existing file."""
    validate((arg, str, 0, 255))
    if not os.path.isfile(arg):
        raise ValueError("{0} does not exist.".format(arg))
    return arg


def _spacy_pos_to_ptb(pos, text):
    """
    Converts a Spacy part-of-speech tag to a Penn Treebank part-of-speech tag.
    Args:
        pos (str): Spacy POS tag.
        text (str): The token text.
    Returns:
        ptb_tag (str): Standard PTB POS tag.
    """
    validate((pos, str, 0, 30), (text, str, 0, 1000))
    ptb_tag = pos
    if text in ['...', '—']:
        ptb_tag = ':'
    elif text == '*':
        ptb_tag = 'SYM'
    elif pos == 'AFX':
        ptb_tag = 'JJ'
    elif pos == 'ADD':
        ptb_tag = 'NN'
    elif text != pos and text in [',', '.', ":", '``', '-RRB-', '-LRB-']:
        ptb_tag = text
    elif pos in ['NFP', 'HYPH', 'XX']:
        ptb_tag = 'SYM'
    return ptb_tag


def normalize(word):
    return 'NUM' if NUMBER_REGEX.match(word) else word.lower()


class ConllEntry:
    def __init__(self, eid, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None,
                 deps=None, misc=None):
        self.id = eid
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

        self.vec = None
        self.lstms = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None,
                  self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def get_options_dict(activation, lstm_dims, lstm_layers, pos_dims):
    """Generates dictionary with all parser options."""
    return {'activation': activation, 'lstm_dims': lstm_dims, 'lstm_layers': lstm_layers,
            'pembedding_dims': pos_dims, 'wembedding_dims': 100, 'rembedding_dims': 25,
            'hidden_units': 100, 'hidden2_units': 0, 'learning_rate': 0.1, 'blstmFlag': True,
            'labelsFlag': True, 'bibiFlag': True, 'costaugFlag': True, 'seed': 0, 'mem': 0}


class SpacyInstance:
    """
    Spacy pipeline wrapper which prompts user for model download authorization.
    Args:
        model (str, optional): spacy model name (default: english small model)
        disable (list of string, optional): pipeline annotators to disable
            (default: [])
        display_prompt (bool, optional): flag to display/skip license prompt
    """

    def __init__(self, model='en', disable=None, display_prompt=True):
        self._parser = spacy.load(model, disable=disable)

    @property
    def parser(self):
        """return Spacy's instance parser"""
        return self._parser

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a sentence into tokens
        Args:
            text (str): text to tokenize
        Returns:
            list: a list of str tokens of input
        """
        # pylint: disable=not-callable

        return [t.text for t in self.parser(text)]


class BISTModel(object):
    """
    BIST parser model class.
    This class handles training, prediction, loading and saving of a BIST parser model.
    After the model is initialized, it accepts a CoNLL formatted dataset as input, and learns to
    output dependencies for new input.
    Args:
        activation (str, optional): Activation function to use.
        lstm_layers (int, optional): Number of LSTM layers to use.
        lstm_dims (int, optional): Number of LSTM dimensions to use.
        pos_dims (int, optional): Number of part-of-speech embedding dimensions to use.
    Attributes:
        model (MSTParserLSTM): The underlying LSTM model.
        params (tuple): Additional parameters and resources for the model.
        options (dict): User model options.
    """

    def __init__(self, activation='tanh', lstm_layers=2, lstm_dims=125, pos_dims=25):
        validate((activation, str), (lstm_layers, int, 0, None), (lstm_dims, int, 0, 1000),
                 (pos_dims, int, 0, 1000))
        self.options = get_options_dict(activation, lstm_dims, lstm_layers, pos_dims)
        self.params = None
        self.model = None

    def fit(self, dataset, epochs=10, dev=None):
        """
        Trains a BIST model on an annotated dataset in CoNLL file format.
        Args:
            dataset (str): Path to input dataset for training, formatted in CoNLL/U format.
            epochs (int, optional): Number of learning iterations.
            dev (str, optional): Path to development dataset for conducting evaluations.
        """
        if dev:
            dev = validate_existing_filepath(dev)
        dataset = validate_existing_filepath(dataset)
        validate((epochs, int, 0, None))

        print('\nRunning fit on ' + dataset + '...\n')
        words, w2i, pos, rels = vocab(dataset)
        self.params = words, w2i, pos, rels, self.options

        from nlp_architect.models.bist.mstlstm import MSTParserLSTM
        self.model = MSTParserLSTM(*self.params)

        for epoch in range(epochs):
            print('Starting epoch', epoch + 1)
            self.model.train(dataset)
            if dev:
                ext = dev.rindex('.')
                res_path = dev[:ext] + '_epoch_' + str(epoch + 1) + '_pred' + dev[ext:]
                write_conll(res_path, self.model.predict(dev))
                run_eval(dev, res_path)  # needed?

    def predict(self, dataset, evaluate=False):
        """
        Runs inference with the BIST model on a dataset in CoNLL file format.
        Args:
            dataset (str): Path to input CoNLL file.
            evaluate (bool, optional): Write prediction and evaluation files to dataset's folder.
        Returns:
            res (list of list of ConllEntry): The list of input sentences with predicted
            dependencies attached.
        """
        dataset = validate_existing_filepath(dataset)
        validate((evaluate, bool))

        print('\nRunning predict on ' + dataset + '...\n')
        res = list(self.model.predict(conll_path=dataset))
        # if evaluate:
        #     ext = dataset.rindex('.')
        #     pred_path = dataset[:ext] + '_pred' + dataset[ext:]
        #     write_conll(pred_path, res)
        #     run_eval(dataset, pred_path)
        return res

    def predict_conll(self, dataset):
        """
        Runs inference with the BIST model on a dataset in CoNLL object format.
        Args:
            dataset (list of list of ConllEntry): Input in the form of ConllEntry objects.
        Returns:
            res (list of list of ConllEntry): The list of input sentences with predicted
            dependencies attached.
        """
        res = None
        if hasattr(dataset, '__iter__'):
            res = list(self.model.predict(conll=dataset))
        return res

    def load(self, path):
        """Loads and initializes a BIST model from file."""
        with open(path.parent / 'params.json') as file:
            self.params = json.load(file)

        from nlp_architect.models.bist.mstlstm import MSTParserLSTM  # TODO
        self.model = MSTParserLSTM(*self.params)  # TODO
        self.model.model.populate(str(path))  # TODO

    def save(self, path):
        """Saves the BIST model to file."""
        print("Saving")
        with open(os.path.join(os.path.dirname(path), 'params.json'), 'w') as file:
            json.dump(self.params, file)
        self.model.model.save(path)


class SpacyBISTParser(object):
    """Main class which handles parsing with Spacy-BIST parser.
    Args:
        verbose (bool, optional): Controls output verbosity.
        spacy_model (str, optional): Spacy model to use
        (see https://spacy.io/api/top-level#spacy.load).
        bist_model (str, optional): Path to a .model file to load. Defaults pre-trained model'.
    """

    def __init__(self, verbose=False, spacy_model='en', bist_model='bist.model'):
        validate((verbose, bool), (spacy_model, str, 0, 1000),
                 (bist_model, (type(None), str), 0, 1000))

        self.verbose = verbose
        self.bist_parser = BISTModel()
        self.bist_parser.load(bist_model)  # TODO
        self.spacy_parser = SpacyInstance(spacy_model,
                                          disable=['ner', 'vectors', 'textcat']).parser

    def to_conll(self, doc_text):
        """Converts a document to CoNLL format with spacy POS tags.
        Args:
            doc_text (str): raw document text.
        Yields:
            list of ConllEntry: The next sentence in the document in CoNLL format.
        """
        validate((doc_text, str))
        for sentence in self.spacy_parser(doc_text).sents:
            sentence_conll = [ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_',
                                         -1, 'rroot', '_', '_')]
            i_tok = 0
            for tok in sentence:
                if self.verbose:
                    print(tok.text + '\t' + tok.tag_)

                if not tok.is_space:
                    pos = tok.tag_
                    text = tok.text

                    if text != '-' or pos != 'HYPH':
                        pos = _spacy_pos_to_ptb(pos, text)
                        token_conll = ConllEntry(i_tok + 1, text, tok.lemma_, pos, pos,
                                                 tok.ent_type_, -1, '_', '_', tok.idx)
                        sentence_conll.append(token_conll)
                        i_tok += 1

            if self.verbose:
                print('-----------------------\ninput conll form:')
                for entry in sentence_conll:
                    print(str(entry.id) + '\t' + entry.form + '\t' + entry.pos + '\t')
            yield sentence_conll

    def parse(self, doc_text, show_tok=True, show_doc=True):
        """Parse a raw text document.
        Args:
            doc_text (str)
            show_tok (bool, optional): Specifies whether to include token text in output.
            show_doc (bool, optional): Specifies whether to include document text in output.
        Returns:
            CoreNLPDoc: The annotated document.
        """
        validate((doc_text, str), (show_tok, bool), (show_doc, bool))
        doc_conll = self.to_conll(doc_text)
        parsed_doc = CoreNLPDoc()

        if show_doc:
            parsed_doc.doc_text = doc_text

        for sent_conll in self.bist_parser.predict_conll(doc_conll):
            parsed_sent = []
            conj_governors = {'and': set(), 'or': set()}

            for tok in sent_conll:
                gov_id = int(tok.pred_parent_id)
                rel = tok.pred_relation

                if tok.form != '*root*':
                    if tok.form.lower() == 'and':
                        conj_governors['and'].add(gov_id)
                    if tok.form.lower() == 'or':
                        conj_governors['or'].add(gov_id)

                    if rel == 'conj':
                        if gov_id in conj_governors['and']:
                            rel += '_and'
                        if gov_id in conj_governors['or']:
                            rel += '_or'

                    parsed_tok = {'start': tok.misc, 'len': len(tok.form),
                                  'pos': tok.pos, 'ner': tok.feats,
                                  'lemma': tok.lemma, 'gov': gov_id - 1,
                                  'rel': rel}

                    if show_tok:
                        parsed_tok['text'] = tok.form
                    parsed_sent.append(parsed_tok)
            if parsed_sent:
                parsed_doc.sentences.append(parsed_sent)
        return parsed_doc


def _read_lexicon_from_csv(lexicon_path: Union[str, PathLike]) -> dict:
    """Read a lexicon from a CSV file.
    Returns:
        Dictionary of LexiconElements, each LexiconElement presents a row.

    Used with built in lexicons like IntensifiersLex
    """
    lexicon = {}
    ABSA_ROOT = Path(path.realpath(__file__)).parent
    INFERENCE_LEXICONS = ABSA_ROOT / 'lexicons'
    with open(INFERENCE_LEXICONS / lexicon_path, newline='', encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in reader:
            try:
                lexicon[row[0]] = LexiconElement(term=row[0], score=row[1], polarity=None,
                                                 is_acquired=None, position=row[2])
            except Exception:
                lexicon[row[0]] = LexiconElement(term=row[0], score=row[1], polarity=None,
                                                 is_acquired=None, position=None)
    return lexicon


def _load_opinion_lex(file_name: Union[str, PathLike]) -> dict:
    """Read opinion lexicon from CSV file.
    Returns:
        Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    ABSA_ROOT = Path(path.realpath(__file__)).parent
    opinion_lex_name = ABSA_ROOT / 'lexicons' / file_name
    lexicon = {}
    with open(opinion_lex_name, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            term, score, polarity, is_acquired = row[0], row[1], row[2], row[3]
            score = float(score)
            # ignore terms with low score
            if score >= 0.5 and polarity in (Polarity.POS.value, Polarity.NEG.value):
                lexicon[term] = \
                    LexiconElement(term.lower(),
                                   score if polarity == Polarity.POS.value else -score, polarity,
                                   is_acquired)
    return lexicon


def _load_aspects_from_dict(aspect_lex: dict):
    """BPS version
    Returns: Dictionary?? of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = []
    for asp_head, variants in aspect_lex.items():
        lexicon.append(LexiconElement(variants))
    return lexicon


class SentimentInference(object):
    """Main class for sentiment inference execution.
    Attributes:
        opinion_lex: Opinion lexicon as outputted by TrainSentiment module.  # TODO: change
        aspect_lex: Aspect lexicon as outputted by TrainSentiment module.  # TODO: change
        intensifier_lex (dict): Pre-defined intensifier lexicon.
        negation_lex (dict): Pre-defined negation lexicon.
    """

    def __init__(self, aspect_lex: dict, opinion_lex: str, parse: bool = False):
        """Inits SentimentInference with given aspect and opinion lexicons."""
        self.aspect_lex = _load_aspects_from_dict(aspect_lex)  # custom aspects from auto-aspect
        self.opinion_lex = _load_opinion_lex(opinion_lex)  # have this as hardcoded like below?
        self.intensifier_lex = _read_lexicon_from_csv('IntensifiersLex.csv')
        self.negation_lex = _read_lexicon_from_csv('NegationSentLex.csv')
        self.parser = None  # needed?

    def run(self, doc: str = None, parsed_doc: CoreNLPDoc = None) -> SentimentDoc:
        """Run SentimentInference on a single document.
        Returns:
            The sentiment annotated document, which contains the detected events per sentence.
        """
        if not parsed_doc:
            if not self.parser:
                raise RuntimeError("Parser not initialized (try parse=True at init )")
            parsed_doc = self.parser.parse(doc)

        sentiment_doc = None
        for sentence in parsed_doc.sentences:
            events = []
            scores = []
            for aspect_row in self.aspect_lex:
                _, asp_events = self._extract_event(aspect_row, sentence)
                for asp_event in asp_events:
                    events.append(asp_event)
                    scores += [term.score for term in asp_event if term.type == TermType.ASPECT]

            if events:
                if not sentiment_doc:
                    sentiment_doc = SentimentDoc(parsed_doc.doc_text)
                sentiment_doc.sentences.append(
                    SentimentSentence(sentence[0]['start'],
                                      sentence[-1]['start'] + sentence[-1]['len'] - 1, events))
        return sentiment_doc

    def _extract_intensifier_terms(self, toks, sentiment_index, polarity, sentence):
        """Extract intensifier events from sentence."""
        count = 0
        terms = []
        for intens_i, intens in [(i, x) for i, x in enumerate(toks) if x in self.intensifier_lex]:
            if math.fabs(sentiment_index - intens_i) == 1:
                score = self.intensifier_lex[intens].score
                terms.append(Term(intens, TermType.INTENSIFIER, polarity, score,
                                  sentence[intens_i]['start'], sentence[intens_i]['len']))
                count += abs(score + float(INTENSIFIER_FACTOR))
        return count if count != 0 else 1, terms

    def _extract_neg_terms(self, toks: list, op_i: int, sentence: list) -> tuple:
        """Extract negation terms from sentence.
        Args:
            toks: Sentence text broken down to tokens (words).
            op_i: Index of opinion term in sentence.
            sentence: parsed sentence
        Returns:
            List of negation terms and its aggregated sign (positive or negative).
        """
        sign = 1
        terms = []
        gov_op_i = sentence[op_i]['gov']
        dep_op_indices = [sentence.index(x) for x in sentence if x['gov'] == op_i]
        for neg_i, negation in [(i, x) for i, x in enumerate(toks) if x in self.negation_lex]:
            position = self.negation_lex[negation].position
            dist = op_i - neg_i
            before = position == 'before' and (dist == 1 or neg_i in dep_op_indices)
            after = position == 'after' and (dist == -1 or neg_i == gov_op_i)
            both = position == 'both' and dist in (1, -1)
            if before or after or both:
                terms.append(Term(negation, TermType.NEGATION, Polarity.NEG,
                                  self.negation_lex[negation].score,
                                  sentence[toks.index(negation)]['start'],
                                  sentence[toks.index(negation)]['len']))
                sign *= self.negation_lex[negation].score
        return terms, sign

    def _extract_event(self, aspect_row: LexiconElement, parsed_sentence: list) -> tuple:
        """Extract opinion and aspect terms from sentence."""
        event = []
        sent_aspect_pair = None
        real_aspect_indices = _consolidate_aspects(aspect_row.term, parsed_sentence)  # TODO
        aspect_key = aspect_row.term[0]
        for aspect_index_range in real_aspect_indices:
            for word_index in aspect_index_range:
                sent_aspect_pair, event = \
                    self._detect_opinion_aspect_events(word_index, parsed_sentence, aspect_key,
                                                       aspect_index_range)
                if sent_aspect_pair:
                    break
        return sent_aspect_pair, event

    @staticmethod
    def _modify_for_multiple_word(cur_tkn, parsed_sentence, index_range):
        """Modify multiple-word aspect tkn length and start index.
        Args:
            index_range: The index range of the multi-word aspect.
        Returns:
            The modified aspect token.
        """
        if len(index_range) >= 2:
            cur_tkn["start"] = parsed_sentence[index_range[0]]["start"]
            cur_tkn["len"] = len(parsed_sentence[index_range[0]]["text"])
            for i in index_range[1:]:
                cur_tkn["len"] = int(cur_tkn["len"]) + len(
                    parsed_sentence[i]["text"]) + 1
        return cur_tkn

    def _detect_opinion_aspect_events(self, aspect_index, parsed_sent, aspect_key, index_range):
        """Extract opinion-aspect events from sentence.
        Args:
            aspect_index: index of aspect in sentence.
            parsed_sent: current sentence parse tree.
            aspect_key: main aspect term serves as key in aspect dict.
            index_range: The index range of the multi word aspect.
        Returns:
            List of aspect sentiment pair, and list of events extracted.
        """
        all_pairs, events = [], []
        sentence_text_list = [x["text"] for x in parsed_sent]
        sentence_text = ' '.join(sentence_text_list)
        for tok_i, tok in enumerate(parsed_sent):
            aspect_op_pair = []
            terms = []
            gov_i = tok['gov']
            gov = parsed_sent[gov_i]
            gov_text = gov['text']
            tok_text = tok['text']

            # 1st order rules
            # Is cur_tkn an aspect and gov an opinion?
            if tok_i == aspect_index:
                if gov_text.lower() in self.opinion_lex:
                    aspect_op_pair.append(
                        (self._modify_for_multiple_word(tok, parsed_sent, index_range), gov))

            # Is gov an aspect and cur_tkn an opinion?
            if gov_i == aspect_index and tok_text.lower() in self.opinion_lex:
                aspect_op_pair.append(
                    (self._modify_for_multiple_word(gov, parsed_sent, index_range), tok))

            # If not found, try 2nd order rules
            if not aspect_op_pair and tok_i == aspect_index:
                # 2nd order rule #1
                for op_t in parsed_sent:
                    if op_t['gov'] == gov_i and op_t['text'].lower() in self.opinion_lex:
                        aspect_op_pair.append(
                            (self._modify_for_multiple_word(tok, parsed_sent, index_range), op_t))

                # 2nd order rule #2
                gov_gov = parsed_sent[parsed_sent[gov_i]['gov']]
                if gov_gov['text'].lower() in self.opinion_lex:
                    aspect_op_pair.append(
                        (self._modify_for_multiple_word(tok, parsed_sent, index_range), gov_gov))

            # if aspect_tok found
            for aspect, opinion in aspect_op_pair:
                op_tok_i = parsed_sent.index(opinion)
                score = self.opinion_lex[opinion['text'].lower()].score
                neg_terms, sign = self._extract_neg_terms(sentence_text_list, op_tok_i,
                                                          parsed_sent)
                polarity = Polarity.POS if score * sign > 0 else Polarity.NEG
                intensifier_score, intensifier_terms = self._extract_intensifier_terms(
                    sentence_text_list, op_tok_i, polarity, parsed_sent)
                over_all_score = score * sign * intensifier_score
                terms.append(Term(aspect_key, TermType.ASPECT, polarity, over_all_score,
                                  aspect['start'], aspect['len']))
                terms.append(Term(opinion['text'], TermType.OPINION, polarity, over_all_score,
                                  opinion['start'], opinion['len']))
                if len(neg_terms) > 0:
                    terms = terms + neg_terms
                if len(intensifier_terms) > 0:
                    terms = terms + intensifier_terms
                all_pairs.append([aspect_key, opinion['text'], over_all_score, polarity,
                                  sentence_text])
                events.append(terms)
        return all_pairs, events


def inference(data: str, aspect_lex: dict, opinion_lex: str) -> dict:
    """BPS version of intel's SentimentSolution.run()
    """
    # initialize the inference engine object
    inference = SentimentInference(aspect_lex, opinion_lex, parse=False)

    # source data is raw text, need to parse
    parse = SpacyBISTParser(bist_model='/Users/Smith-Box/nlp-architect/cache/bist-pretrained/').parse

    # run inference on the data
    parsed_doc = parse(doc)  # but do this as preprocessing?

    # sentiment_doc = inference.run(parsed_doc=parsed_doc)
    sentiment_doc = {'aspects': aspect_lex,
                     'opinions': opinion_lex}

    return sentiment_doc
