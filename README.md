# Rationale

Swahili is an agglitunative language that builds words up from semantically meaningful morphemes.  For example the word "kinachohitajika", meaning "the thing which is needed" or  "what is required" can be analyzed like this:

- "ki" is a subject prefix that refers to a Class 7 noun, typically "thing" or "object." This is third person, singular, that is "it" in English.
- "na" is a temporal verb marker that expresses present tense.
- "cho" is an object infix that adds specificity to the noun class in the relative clause.
- "hitaj" is the verb stem. It means "need" or "require."
- "ik" is a suffix that expresses reflexivity in the passive voice. It morphs the verb to mean "be needed" or "be required."
-  "a" is the Bantu verb ending vowel

Currently available Large Language Models don't tokenize text on such a semantic basis, so they are not aware of this grammatical structure. Therefore they cannot develop "understanding" for it. 
The model embeddings of the tokens won't reflect this semantic knowledge about Swahili. The word will be probably tokenized into a token sequence, for example, like "kinac", "ho", "hit", "aji", "ka".
Using a semantic tokenizer for the Swahili language should allow us building a more proficient Swahili-speaking LLM model because the predicted token sequences will take the semantic meaning of the morphemes "ki", "na", etc. into account.

# Installation 

`poetry install`

# Status

Swahili is a morphologically rich language with complex rules. Creating a perfect Swahili morphological analyzer would be very difficult, if it's possible at all, and actually it's not even necessary to achieve our goals. 
Using`coverage_report.py` on the Swahili portion of Wikipedia, currently more than half of the words were analyzed successfully. Unsuccessful cases will be handled by a traditional BPE tokenization method.
Coverage could be increased by adding even more noun and verb stems and adding even more grammar rules to handle less common use cases. Please study `tests\sw_morphs.yaml` to see what works now and feel free to extend it.


