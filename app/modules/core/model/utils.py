"""
"""
# from .utils import vocab  # maybe no
# from .utils import write_conll  # maybe no
# from .utils import run_eval  # maybe no
import csv
import re
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


from .data_types import LexiconElement
from .data_types import ConllEntry
