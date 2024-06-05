"""Tests using G-Eval's custom metrics"""

from deepeval import assert_test
from deepeval import test_case
from deepeval import metrics

import langsmith


# Build a custom metric with G-Eval
correctness_metric = metrics.GEval(
    name='Correctness',
    criteria='Test whether the actual output is factually correct.',
    # alternative to using criteria - evaluation_steps
    # evaluation_steps=[
    #     "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
    #     "You should also heavily penalize omission of detail",
    #     "Vague language, or contradicting OPINIONS, are OK"
    # ],
    evaluation_params=[
        test_case.LLMTestCaseParams.INPUT,
        test_case.LLMTestCaseParams.ACTUAL_OUTPUT
    ],
)


@langsmith.unit
def test_answer_correct():
    """This test should pass"""
    test = test_case.LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris"
    )
    assert_test(test, [correctness_metric])


@langsmith.unit
def test_vague_answer():
    """This test should fail"""
    test = test_case.LLMTestCase(
        input="What if the capital of France?",
        actual_output="It is Paris, probably"
    )
    try:
        assert_test(test, [correctness_metric])
    finally:
        print(correctness_metric.score)
        print(correctness_metric.reason)

