"""SageMaker pipeline skeleton for the Yelp sentiment MLOps project.

This file documents the planned AWS pipeline structure. It intentionally avoids
hardcoded AWS account IDs, bucket names, roles, or credentials.
"""


PIPELINE_STEPS = [
    {
        "name": "ValidateData",
        "service": "SageMaker Processing",
        "description": "Check Yelp review schema, missing text, and valid star ratings.",
    },
    {
        "name": "PreprocessData",
        "service": "SageMaker Processing",
        "description": "Create labels, clean text, and write train/validation/test/batch splits.",
    },
    {
        "name": "RunEDA",
        "service": "SageMaker Processing",
        "description": "Generate class balance, rating, review length, and missing data reports.",
    },
    {
        "name": "TrainModel",
        "service": "SageMaker Training",
        "description": "Train baseline or transformer-based sentiment classifier.",
    },
    {
        "name": "EvaluateModel",
        "service": "SageMaker Processing",
        "description": "Calculate accuracy, precision, recall, F1-score, and confusion matrix.",
    },
    {
        "name": "QualityGate",
        "service": "SageMaker Pipeline ConditionStep",
        "description": "Approve only if macro F1-score meets the configured threshold.",
    },
    {
        "name": "RegisterModel",
        "service": "SageMaker Model Registry",
        "description": "Register approved model artifact for deployment.",
    },
    {
        "name": "BatchInference",
        "service": "SageMaker Batch Transform",
        "description": "Score new Yelp reviews on a scheduled batch workflow.",
    },
    {
        "name": "MonitorModel",
        "service": "SageMaker Model Monitor and CloudWatch",
        "description": "Monitor prediction distribution, model quality, latency, and resource usage.",
    },
]


def describe_pipeline() -> None:
    """Print the planned AWS pipeline steps."""
    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"{index}. {step['name']} ({step['service']})")
        print(f"   {step['description']}")


if __name__ == "__main__":
    describe_pipeline()
