import pytest
import os
import sys

# make sure pytest knows current dir
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(current_dir))

@pytest.fixture
def test_data_dir():
    file_path = os.path.realpath(__file__)
    return os.path.join(os.path.dirname(file_path), "test_data")
