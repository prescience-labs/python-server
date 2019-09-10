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

# import requests
# from tqdm import tqdm
import dynet


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
        # return json.dumps(self, cls=SentimentDocEncoder)
        jobj = json.dumps(self, cls=SentimentDocEncoder)
        return json.loads(jobj)

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


class SentimentDocEncoder(JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        try:
            if isinstance(o, Enum):
                return getattr(o, '_name_')
            if hasattr(o, '__dict__'):
                return vars(o)
            if hasattr(o, '__slots__'):
                ret = {slot: getattr(o, slot) for slot in o.__slots__}
                for cls in type(o).mro():
                    spr = super(cls, o)
                    if not hasattr(spr, '__slots__'):
                        break
                    for slot in spr.__slots__:
                        ret[slot] = getattr(o, slot)
                return ret
        except Exception as e:
            print(e)
        return None


################################## BISTParser ##################################


################################## Inference  ##################################


INTENSIFIER_FACTOR = 0.3
VERB_POS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
NUMBER_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")

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

    def __init__(self, model="en-core-web-sm", disable=None, display_prompt=True):
        # self._parser = spacy.load(model, disable=disable)
        self._parser = spacy.load("en_core_web_sm", disable=disable)

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
        # with open(path.parent / 'params.json') as file:
        with open('app/modules/core/model/params.json') as file:
            self.params = json.load(file)

        from .mstlstm import MSTParserLSTM
        self.model = MSTParserLSTM(*self.params)
        self.model.model.populate(str(path))

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

    def __init__(self, verbose=False, spacy_model="en-core-web-sm", bist_model='bist.model'):
        validate((verbose, bool), (spacy_model, str, 0, 1000),
                 (bist_model, (type(None), str), 0, 1000))

        self.verbose = verbose
        self.bist_parser = BISTModel()
        self.bist_parser.load(Path(bist_model))
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


def _sentence_contains_after(sentence, index, phrase):
    """Returns sentence contains phrase after given index."""
    for i in range(len(phrase)):
        if len(sentence) <= index + i or phrase[i].lower() not in \
                {sentence[index + i][field].lower() for field in ('text', 'lemma')}:
            return False
    return True


def _consolidate_aspects(aspect_row, sentence):
    """Returns consolidated indices of aspect terms in sentence.

    Args:
        aspect_row: List of aspect terms which belong to the same aspect-group.
    """
    indices = []
    aspect_phrases: list = \
        sorted([phrase.split(' ') for phrase in aspect_row], key=len, reverse=True)
    appeared = set()
    for tok_i in range(len(sentence)):
        for aspect_phrase in aspect_phrases:
            if _sentence_contains_after(sentence, tok_i, aspect_phrase):
                span = range(tok_i, tok_i + len(aspect_phrase))
                if not appeared & set(span):
                    appeared |= set(span)
                    indices.append(list(span))
    return indices


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
        return sentiment_doc.json()

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
        real_aspect_indices = _consolidate_aspects(aspect_row.term, parsed_sentence)
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
    parse = SpacyBISTParser(bist_model='app/modules/core/model/bist.model').parse

    # run inference on the data
    parsed_doc = parse(data)  # but do this as preprocessing?

    sentiment_doc = inference.run(parsed_doc=parsed_doc)
    # sentiment_doc = {'aspects': aspect_lex,
    #                  'opinions': opinion_lex}

    return sentiment_doc
