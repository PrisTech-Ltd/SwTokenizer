# swahili_tokenizer.py
from lark import Lark, Tree, UnexpectedInput
from typing import Set, List
import logging
import yaml
from pathlib import Path
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

WORD_BOUNDARY_MARKER = "▁"

class MorphemeExtractor:
    def __init__(self, original_text: str):
        self.original_text = original_text
        self.tokens = []

    def extract(self, tree: Tree) -> List[str]:
        """Extract morphemes from the parse tree, preserving capitalization."""
        self._process_node(tree, start_of_word=True)
        return self.tokens

    def _process_node(self, node, start_of_word):
        """Recursively process nodes in the parse tree to extract morphemes."""
        if isinstance(node, Tree):
            for child in node.children:
                self._process_node(child, start_of_word)
                start_of_word = False  # Set False after the first morpheme in a word
        else:
            self._process_token(node.type, node.value, start_of_word)

    def _process_token(self, token_type: str, value: str, start_of_word: bool):
        """Process each token, preserving capitalization at the start of words."""
        # Apply capitalization if it's the start of a word and original text has it
        if start_of_word and self.original_text[0].isupper():
            value = value.capitalize()
        if start_of_word:
            value = WORD_BOUNDARY_MARKER + value
        self.tokens.append(value)



class SwahiliTokenizer:
    # Base grammar template with placeholders for dynamic vocabularies
    BASE_GRAMMAR = r"""
    start: word+

    ?word:  pronoun_form | verb_form | noun_form |  adjective_form | invariable | interrogative_quantifier

    // Verb forms
    verb_form: infinitive_verb | conjugated_verb | existential_verb  | relative_verb  | locative_form

    infinitive_verb: INFINITIVE_VERB_PREFIX INFINITIVE_VERB_NEGATION_INFIX? REFLEXIV_PREFIX? VERB_STEM RECIPROCAL_SUFFIX? FINAL_VOWEL
    conjugated_verb: SUBJECT_PREFIX? NEGATION_PREFIX? tense_modal_sequence? REFLEXIV_PREFIX? OBJECT_PREFIX? VERB_STEM RECIPROCAL_SUFFIX? FINAL_VOWEL    
    existential_verb: SUBJECT_PREFIX LOCATIVE_SUFFIX  
    relative_verb: SUBJECT_PREFIX TENSE_MARKER RELATIVE_MARKER VERB_STEM FINAL_VOWEL 
    locative_form: LOCATIVE_PREFIX VERB_STEM FINAL_VOWEL 
    
    INFINITIVE_VERB_PREFIX: "ku"
    INFINITIVE_VERB_NEGATION_INFIX: "to"
    NEGATION_PREFIX: "si"
    RELATIVE_MARKER: "o"

    tense_modal_sequence: TENSE_MARKER? MODAL_MARKER?  PERFECT_ASPECT?  LOCATIVE_MARKER? 

    // Interrogative and Quantifying Terms
    interrogative_quantifier: INTERROGATIVE | QUANTIFIER

    // Lexical rules for verbs    
    VERB_STEM: {verb_stems}
    SUBJECT_PREFIX: {subject_prefixes}
    TENSE_MARKER: {tense_markers}
    OBJECT_PREFIX: {object_prefixes}
    FINAL_VOWEL: {final_vowels}
    LOCATIVES: "po" | "ko" | "mo"
    LOCATIVE_MARKER:  LOCATIVES
    LOCATIVE_SUFFIX: LOCATIVES
    MODAL_MARKER: "ka" | "nge" | "ki"
    REFLEXIV_PREFIX: "ji"
    RECIPROCAL_SUFFIX: "an"
    PERFECT_ASPECT: "sha"

    // Noun forms
    noun_form: compound_noun | simple_noun | locative_noun | verb_noun
    
    // nouns
    simple_noun: NOUN_PREFIX? NOUN_STEM NOUN_SUFFIX?
    compound_noun: NOUN_PREFIX? NOUN_STEM_FIRST NOUN_STEM  NOUN_SUFFIX?
    locative_noun: LOCATIVE_PREFIX NOUN_STEM
    verb_noun: CLASS_NOUN_PREFIX VERB_STEM CLASS_NOUN_SUFFIX
    
    // Lexical rules for nouns
    NOUN_STEM: {noun_stems}
    NOUN_STEM_FIRST: {noun_stem_firsts}
    NOUN_PREFIX: {noun_prefixes}
    NOUN_SUFFIX: "ni"   
    LOCATIVE_PREFIX: "wa" | "pa" | "ma" | "kw"
    CLASS_NOUN_PREFIX: "cha" | "ki" | "vi" | "ma" | "m" | "mw" | "wa"
    CLASS_NOUN_SUFFIX: "a" | "o" | "izi" | "u"
    
    // Pronoun forms
    
    pronoun_form: PRONOUN_CLASS_MARKER PRONOUN_STEM | indefinite_pronoun | relative_pronoun
    PRONOUN_CLASS_MARKER: "w" | "mw"| "m" | "y"  | "vy" | "ch" | "z" | "p" | "l" | "k"
    POSSESSIVE_STEM: "angu" | "ake" | "ako" | "etu" | "enu" | "ao"
    RELATIONAL_STEM:  "enye"
    WHOLE_STEM:       "ote"
    PRONOUN_STEM: POSSESSIVE_STEM | RELATIONAL_STEM | WHOLE_STEM
    
    indefinite_pronoun: INDEFINITE_PREFIX INDEFINITE_PREFIX INDEFINITE_SUFFIX
    INDEFINITE_SUFFIX: "te"
    INDEFINITE_PREFIX:  {indefinite_prefixes}
    relative_pronoun: RELATIVE_ROOT RELATIVE_SUFFIX
    RELATIVE_ROOT: "amba"
    RELATIVE_SUFFIX: "o" | "ye" | "cho" | "vyo" | "lo" | "po" | "zo"
    

    // Adjective forms
    adjective_form: ADJ_PREFIX ADJ_STEM | QUANTIFIER_PREFIX QUANTIFIER_SUFFIX

    // Lexical rules for adjectives
    ADJ_STEM: {adj_stems}
    ADJ_PREFIX: {adj_prefixes}
    QUANTIFIER_PREFIX: "w" | "m" |  "ny" | "ch" | "mw"
    QUANTIFIER_SUFFIX: "ingi" | "engi" | "ingine"

    // Invariable words
    invariable: INVARIABLE
    INVARIABLE: {invariables}
    
    // Interrogative and quantifying terms
    INTERROGATIVE: {interrogatives}
    QUANTIFIER: {quantifiers}    

    // Ignore whitespace
    %import common.WS
    %ignore WS
"""

    MIN_TOKEN_LENGTH = 4

    def __init__(self, config_path: str = None):
        """
        Initialize tokenizer with vocabulary from config file

        Args:
            config_path: Path to YAML configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'swahili_rules.yaml'

        self.config_path = Path(config_path)
        self.load_config()
        self._rebuild_parser()
        self.success = Counter()
        self.failure = Counter()
        self.all_morphemes = set()

    def load_config(self):
        """Load vocabulary and morpheme configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Load vocabularies
            # Convert each vocabulary entry to a string to avoid type errors
            self._verb_stems = {str(item) for item in config['vocabularies']['verb_stems']}
            self._noun_stems = {str(item) for item in config['vocabularies']['noun_stems']}
            self._noun_stem_firsts = {str(item) for item in config['vocabularies']['noun_stem_firsts']}
            self._adj_stems = {str(item) for item in config['vocabularies']['adjective_stems']}

            # # Load morphemes
            self._morphemes = {key: [str(m) for m in morphemes]
                               for key, morphemes in config['morphemes'].items()}

            # Load interrogatives and quantifiers
            self._interrogatives = {str(item) for item in config['interrogatives']}
            self._quantifiers = {str(item) for item in config['quantifiers']}

        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {str(e)}")
            # Initialize with empty sets as fallback
            self._verb_stems = set()
            self._noun_stems = set()
            self._noun_stem_firsts = set()
            self._adj_stems = set()
            self._morphemes = {
                'subject_prefixes': [], 'tense_markers': [],
                'object_prefixes': [], 'noun_prefixes': [],
                'adjective_prefixes': [], 'indefinite_prefixes': [], 'final_vowels': [],
                'invariables': []
            }
            self._interrogatives = set()
            self._quantifiers = set()

    def _rebuild_parser(self):
        """Rebuild the Lark parser with current vocabularies"""

        def format_alternatives(items: list) -> str:
            """Format a list of items as Lark alternatives"""
            return " | ".join(f'"{item}"' for item in sorted(items)) or '"none"'

        # Format all grammar components
        grammar = self.BASE_GRAMMAR.format(
            verb_stems=format_alternatives(self._verb_stems),
            noun_stems=format_alternatives(self._noun_stems),
            noun_stem_firsts=format_alternatives(self._noun_stem_firsts),
            adj_stems=format_alternatives(self._adj_stems),
            subject_prefixes=format_alternatives(self._morphemes['subject_prefixes']),
            tense_markers=format_alternatives(self._morphemes['tense_markers']),
            object_prefixes=format_alternatives(self._morphemes['object_prefixes']),
            noun_prefixes=format_alternatives(self._morphemes['noun_prefixes']),
            indefinite_prefixes=format_alternatives(self._morphemes['indefinite_prefixes']),
            adj_prefixes=format_alternatives(self._morphemes['adjective_prefixes']),
            final_vowels=format_alternatives(self._morphemes['final_vowels']),
            invariables=format_alternatives(self._morphemes['invariables']),
            interrogatives=format_alternatives(self._interrogatives),
            quantifiers=format_alternatives(self._quantifiers)
        )

        self.parser = Lark(grammar, parser='earley', start='start', maybe_placeholders=False)

    def reload_config(self):
        """Reload configuration from file and rebuild parser"""
        self.load_config()
        self._rebuild_parser()

    def add_verb_stems(self, stems: Set[str]) -> None:
        """Add new verb stems to the vocabulary"""
        self._verb_stems.update(stems)
        self._rebuild_parser()

    def add_noun_stems(self, stems: Set[str]) -> None:
        """Add new noun stems to the vocabulary"""
        self._noun_stems.update(stems)
        self._rebuild_parser()

    def add_adj_stems(self, stems: Set[str]) -> None:
        """Add new adjective stems to the vocabulary"""
        self._adj_stems.update(stems)
        self._rebuild_parser()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a word into morphemes, handling unknown words gracefully.

        Args:
            text: Text to tokenize

        Returns:
            List of morphemes
        """
        try:
            tree = self.parser.parse(text.lower())
            morphemes = self._extract_morphemes(text, tree)
            logger.debug(f" {text} -> {morphemes}")
            for morpheme in morphemes:
                self.success[morpheme] += 1
            return morphemes
        except UnexpectedInput as e:
            #print(f"tokenization error: {e}")
            #logger.warning(f"Unable to parse '{text}': {str(e)}")
            logger.warning(f"Did not tokenize '{text}'")
            self.failure[text] += 1
            return [WORD_BOUNDARY_MARKER+text]  # Return the original word if parsing fails
        except Exception as e:
            logger.error(f"Unexpected error parsing '{text}': {str(e)}")
            return [WORD_BOUNDARY_MARKER+text]


    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text into morphemes, with Gemma2-style word boundary markers.
        Handling unknown words gracefully.

        Args:
            text: Text to tokenize
        Returns:
            List of morphemes with word boundary markers at word starts
        """

        all_tokens = []
        for word in text.split():
            morphemes = self._tokenize(word)
            all_tokens.extend(morphemes)
        return all_tokens

    def _extract_morphemes(self, text: str, tree: Tree) -> List[str]:
        extractor = MorphemeExtractor(text)
        return extractor.extract(tree)


if __name__ == "__main__":
    tokenizer = SwahiliTokenizer()
    print(tokenizer.tokenize("Unahitaji kuona kwangu, kaka. Nimejenga nyumba nzuri sana"))
    print(tokenizer.success, tokenizer.failure)