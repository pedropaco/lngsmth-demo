"""Sample evaluation.

This sample test uses LangSmith to evaluate the welcome_message() function.
"""

import langsmith
from langsmith import evaluation


SAMPLE_DATASET_NAME='Sample Dataset'
SAMPLE_DATASET_ID='2a689744-b2b1-4815-bd87-37e36af9df1f'


client = langsmith.Client()


def welcome_message(inputs_obj):
    """Message under test.

    Create input using the "inputs" list from the examples above.
    """
    return "Welcome " + inputs_obj['postfix']


def exact_match(run, example):
    """Match the result against the dataset output."""
    return {"score": run.outputs["output"] == example.outputs["output"]}


def create_dataset():
    """Dataset for test cases.

    This defines and persists a dataset in the langchain project, so multiple
    uses will cause "already exists" errors.
    """
    dataset = client.create_dataset(
        SAMPLE_DATASET_NAME, description="A sample dataset in LangSmith.")
    return dataset_name, dataset.id


def get_dataset():
    """Return the ID of an already-created dataset."""
    return SAMPLE_DATASET_NAME, SAMPLE_DATASET_ID


def main():
    dataset_name, dataset_id = get_dataset()
    # dataset_id = create_dataset()

    client.create_examples(
        inputs=[
            {"postfix": "to LangSmith"},
            {"postfix": "to Evaluations in LangSmith"},
        ],
        outputs=[
            {"output": "Welcome to LangSmith"},
            {"output": "Welcome to Evaluations in LangSmith"},
        ],
        dataset_id=dataset_id,
    )

    experiment_results = evaluation.evaluate(
        welcome_message,
        data=dataset_name,
        evaluators=[exact_match],
        experiment_prefix="sample-experiment",
        metadata={
          "version": "1.0.0",
          "revision_id": "beta"
        },
    )


if __name__ == '__main__':
    main()


