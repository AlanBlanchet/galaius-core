import pytest
from pydantic import ValidationError

from interact_core import DeviceLoginStart, UserCode


@pytest.mark.parametrize("typed, code", [("bcdf-ghjk", "BCDF-GHJK"), (" BCDF GHJK ", "BCDF-GHJK"), ("bcdfghjk", "BCDF-GHJK")])
def test_typed_code_reads_as_the_shown_one(typed: str, code: str) -> None:
    assert UserCode.normalize(typed) == code


@pytest.mark.parametrize("typed", ["", "BCDF-GHJ", "BCDF-GHJKL", "ABCD-EFGH", "BCDF-GH1K"])
def test_anything_else_is_not_a_code(typed: str) -> None:
    with pytest.raises(ValueError):
        UserCode.normalize(typed)


def test_new_codes_use_the_alphabet_only() -> None:
    assert all(UserCode.PATTERN.fullmatch(UserCode.new()) for _ in range(200))


@pytest.mark.parametrize("hostname, shown", [("laptop.local", "laptop.local"), ("<img src=x>", "img-src-x"), ("   ", "computer"), ("x" * 90, "x" * 63), ("PC d'Alan", "PC-d-Alan")])
def test_computer_name_is_plain_text(hostname: str, shown: str) -> None:
    assert DeviceLoginStart(client_name=hostname, platform="linux", client_version="0.43.0").client_name == shown


def test_platform_and_version_are_closed_sets() -> None:
    with pytest.raises(ValidationError):
        DeviceLoginStart(client_name="pc", platform="plan9", client_version="1")
    with pytest.raises(ValidationError):
        DeviceLoginStart(client_name="pc", platform="linux", client_version="1 <b>")
