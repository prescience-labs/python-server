"""
"""
import csv
import json
import re
from json import JSONEncoder
from enum import Enum
from pathlib import Path
from typing import Union
from os import PathLike


def _load_aspects_from_dict(aspect_lex: dict):
    """BPS version
    Returns: Dictionary?? of LexiconElements, each LexiconElement presents a row.
    """
    lexicon = []
    for asp_head, variants in aspect_lex.items():
        lexicon.append(LexiconElement(variants))
    return lexicon


def _load_opinion_lex(file_name: Union[str, PathLike]) -> dict:
    """Read opinion lexicon from CSV file.
    Returns:
        Dictionary of LexiconElements, each LexiconElement presents a row.
    """
    # ABSA_ROOT = Path(file_name.realpath(__file__)).parent
    ABSA_ROOT = Path('app/modules/core/model/')
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


def _read_lexicon_from_csv(lexicon_path: Union[str, PathLike]) -> dict:
    """Read a lexicon from a CSV file.
    Returns:
        Dictionary of LexiconElements, each LexiconElement presents a row.

    Used with built in lexicons like IntensifiersLex
    """
    lexicon = {}
    # ABSA_ROOT = Path(path.realpath(__file__)).parent
    ABSA_ROOT = Path('app/modules/core/model/')
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


def _sentence_contains_after(sentence, index, phrase):
    """Returns sentence contains phrase after given index."""
    for i in range(len(phrase)):
        if len(sentence) <= index + i or phrase[i].lower() not in \
                {sentence[index + i][field].lower() for field in ('text', 'lemma')}:
            return False
    return True


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


def get_options_dict(activation, lstm_dims, lstm_layers, pos_dims):
    """Generates dictionary with all parser options."""
    return {'activation': activation, 'lstm_dims': lstm_dims, 'lstm_layers': lstm_layers,
            'pembedding_dims': pos_dims, 'wembedding_dims': 100, 'rembedding_dims': 25,
            'hidden_units': 100, 'hidden2_units': 0, 'learning_rate': 0.1, 'blstmFlag': True,
            'labelsFlag': True, 'bibiFlag': True, 'costaugFlag': True, 'seed': 0, 'mem': 0}


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


def normalize(word):
    NUMBER_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
    return 'NUM' if NUMBER_REGEX.match(word) else word.lower()


def merge_punct_tok(merged_punct_sentence, last_merged_punct_index, punct_text, is_traverse):
    # merge the text of the punct tok
    if is_traverse:
        merged_punct_sentence[last_merged_punct_index]["text"] = \
            punct_text + merged_punct_sentence[last_merged_punct_index]["text"]
    else:
        merged_punct_sentence[last_merged_punct_index]["text"] = \
            merged_punct_sentence[last_merged_punct_index]["text"] + punct_text


def fix_gov_indexes(merged_punct_sentence, sentence):
    for merged_token in merged_punct_sentence:
        tok_gov = merged_token['gov']
        if tok_gov == -1:  # gov is root
            merged_token['gov'] = -1
        else:
            orig_gov = sentence[tok_gov]
            correct_index = find_correct_index(orig_gov, merged_punct_sentence)
            merged_token['gov'] = correct_index


def find_correct_index(orig_gov, merged_punct_sentence):
    for tok_index, tok in enumerate(merged_punct_sentence):
        if tok["start"] == orig_gov["start"] and tok["len"] == orig_gov["len"] and tok["pos"] == \
                orig_gov["pos"] and tok["text"] == orig_gov["text"]:
            return tok_index
    return None


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
