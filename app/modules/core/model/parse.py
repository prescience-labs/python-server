"""Adaptation of spacy+BIST parser used in NLP Architect by Intel.
Preprocessing engine.
"""
import json
from pathlib import Path
from typing import List

import spacy

from .utils import validate
from .utils import validate_existing_filepath
from .utils import _spacy_pos_to_ptb
from .utils import get_options_dict
from .utils import ConllEntry
from .utils import CoreNLPDoc
from .mstlstm import MSTParserLSTM
# from .utils import vocab  # TODO: remove
# from .utils import write_conll  # TODO: remove
# from .utils import run_eval  # TODO: remove


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

        from .mstlstm import MSTParserLSTM
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
