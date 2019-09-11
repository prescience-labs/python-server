"""
"""
import json
from enum import Enum


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


from .utils import normalize
from .utils import merge_punctuation
from .utils import fix_gov_indexes
