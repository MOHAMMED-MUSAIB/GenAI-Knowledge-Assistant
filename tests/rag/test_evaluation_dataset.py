
import json
from pathlib import Path


def test_evaluation_dataset_has_valid_cases() -> None:
    dataset_path = Path(__file__).with_name("evaluation_dataset.json")
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    assert len(cases) >= 3

    for case in cases:
        assert isinstance(case["question"], str)
        assert case["question"].strip()
        assert isinstance(case["expected_answer"], str)
        assert case["expected_answer"].strip()
        assert "expected_source" in case
        assert (
            case["expected_source"] is None
            or (
                isinstance(case["expected_source"], str)
                and case["expected_source"].strip()
            )
        )
        assert case["category"] in {
            "factual",
            "unsupported",
            "retrieval",
        }
