"""Run the local end-to-end sentiment MLOps pipeline."""

from src import evaluate, model_gate, monitor, prepare_data, train


def main() -> None:
    """Prepare data, train, evaluate, check quality gate, and monitor."""
    prepare_data.main()
    train.main()
    evaluate.main()
    model_gate.main()
    monitor.main()


if __name__ == "__main__":
    main()
