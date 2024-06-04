"""An example of a pytest-runnable tests that integrate with Langsmith."""

import random

import langsmith


def capitals(query):
    state_capitals = {
        'What is the capital of California?': 'Sacramento',
        'What is the capital of Vermont?': 'Montpelier',
        'What is the capital of Washington?': 'New Delhi',
    }
    return state_capitals[query]


def fuzzy_capitals(query):
    answer = capitals(query)
    return random.choice(["It's", "the capital is", "duh, "]) + answer

@langsmith.unit
def test_california_capital():
    """A test with a standard pytest assertion."""
    answer = capitals('What is the capital of California?')
    assert 'Sacramento' == answer


@langsmith.unit
def test_california_capital():
    """A similarity test. The calculated distance will be logged in LangChain"""
    answer = fuzzy_capitals('What is the capital of California?')
    langsmith.expect.embedding_distance(
        answer, 'Sacramento'
    ).to_be_less_than(0.5)


@langsmith.unit
def test_washington_capital():
    """A failing similarity test. The calculation will be logged in LangChain"""
    answer = fuzzy_capitals('What is the capital of Washington?')
    langsmith.expect.embedding_distance(
        answer, 'Seattle'
    ).to_be_less_than(0.5)

