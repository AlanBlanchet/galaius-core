"""Bounds of the public contact form: every field fails alone, at its own limit."""

import pytest
from pydantic import ValidationError

from interact_core.contact import ContactSubmission

VALID = {"name": "Ada Lovelace", "email": "ada@example.com", "company": "Analytical Engines", "message": "Hello\nthere", "locale": "fr"}


def test_valid_submission_is_stripped_and_blank_company_is_none() -> None:
    value = ContactSubmission.model_validate({**VALID, "name": "  Ada  ", "company": "   "})
    assert (value.name, value.company, value.message) == ("Ada", None, "Hello\nthere")


@pytest.mark.parametrize(("field", "value"), [
    ("name", "   "), ("name", "x" * 121), ("name", "Ada\nLovelace"),
    ("email", "not-an-email"), ("email", "a@b"), ("email", f"{'a' * 250}@b.io"),
    ("company", "x" * 161),
    ("message", ""), ("message", "x" * 4001), ("message", "bell\x07"),
    ("locale", "de"),
    ("website", "http://spam.example"),
])
def test_each_bound_rejects_only_its_field(field: str, value: str) -> None:
    with pytest.raises(ValidationError) as error:
        ContactSubmission.model_validate({**VALID, field: value})
    assert {item["loc"][0] for item in error.value.errors()} == {field}


def test_limits_are_inclusive() -> None:
    ContactSubmission.model_validate({**VALID, "name": "x" * 120, "company": "x" * 160, "message": "x" * 4000, "email": f"{'a' * 244}@b.example"})
