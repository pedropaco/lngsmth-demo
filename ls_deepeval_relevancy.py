"""Example tests using DeepEval relevancy"""

from deepeval import assert_test
from deepeval import test_case
from deepeval import metrics

import langsmith


answer_relevancy_metric = metrics.AnswerRelevancyMetric(threshold=0.5)


@langsmith.unit
def test_answer_relevancy():
    """This test should pass"""
    test = test_case.LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost."
    )
    assert_test(test, [answer_relevancy_metric])


@langsmith.unit
def test_irrelevant_answer():
    """This test should fail"""
    answer_relevancy_metric = metrics.AnswerRelevancyMetric(threshold=0.5)
    test = test_case.LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="I hate clowns!"
    )
    try:
        assert_test(test, [answer_relevancy_metric])
    finally:
        print(answer_relevancy_metric.score)
        print(answer_relevancy_metric.reason)
