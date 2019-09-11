"""A module for running leveraging a pretrained model to analyze a single
review.

TODO:
- use import parse, utils, data_types and then dot notation instead of from?
"""
import math
from pathlib import Path

from .parse import SpacyBISTParser
from .utils import _load_aspects_from_dict
from .utils import _load_opinion_lex
from .utils import _read_lexicon_from_csv
from .utils import _consolidate_aspects
from .utils import CoreNLPDoc
from .utils import SentimentDoc
from .utils import SentimentSentence
from .utils import LexiconElement
from .utils import Term
from .utils import TermType
from .utils import Polarity

INTENSIFIER_FACTOR = 0.3


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
    parse = SpacyBISTParser(bist_model='app/modules/core/model/bist/bist.model').parse
    parsed_doc = parse(data)  # but do this as preprocessing?

    # run inference on the data
    sentiment_doc = inference.run(parsed_doc=parsed_doc)

    return sentiment_doc
