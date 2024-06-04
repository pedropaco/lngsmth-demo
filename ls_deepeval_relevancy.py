"""Example tests using DeepEval relevancy"""

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval import metrics

import langsmith


@langsmith.unit
def test_answer_relevancy():
    """Good evaluation"""
    answer_relevancy_metric = metrics.AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost."
    )
    assert_test(test_case, [answer_relevancy_metric])


@langsmith.unit
def test_answer_irrelevant():
    """Poor evaluation"""
    answer_relevancy_metric = metrics.AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="I hate clowns!"
    )
    assert_test(test_case, [answer_relevancy_metric])
