# test_swahili_tokenizer.py
import pytest
import yaml
from swahili_tokenizer import SwahiliTokenizer

def load_test_cases():
    with open('sw_morphs.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['test_cases']

@pytest.fixture
def tokenizer():
    # Create tokenizer with default config
    return SwahiliTokenizer()

@pytest.mark.parametrize(
    "test_input,expected,description",
    [(case['input'], case['expected'], case['description'])
     for case in load_test_cases()],
    ids=lambda case: case[0]
)
def test_tokenizer(tokenizer, test_input, expected, description):
    result = tokenizer.tokenize(test_input)
    assert result == expected, f"Failed on '{test_input}' ({description})"

def test_config_reload(tokenizer):
    # Test that the tokenizer can reload its configuration
    tokenizer.reload_config()
    result = tokenizer.tokenize("kuona")
    assert result == ["▁ku", "on", "a"]

def test_missing_config():
    # Test that the tokenizer handles missing config gracefully
    tokenizer = SwahiliTokenizer('nonexistent.yaml')
    result = tokenizer.tokenize("xyz")
    assert result == ["▁xyz"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])