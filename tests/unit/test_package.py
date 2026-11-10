import dli


def test_package_exposes_version() -> None:
    assert dli.__version__.startswith("0.1.0")
