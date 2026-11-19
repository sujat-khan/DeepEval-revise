import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from judge import GroqJudge


from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import Golden
from deepeval import assert_test
from agent import my_agent
import pytest

judge= GroqJudge()

goldens = [Golden(input="Plan a three-day trip to Paris")]
@pytest.mark.parametrize("golden", goldens)
def test_agent(golden: Golden):
    my_agent(golden.input)
    assert_test(golden=golden, metrics=[TaskCompletionMetric(model=judge)])