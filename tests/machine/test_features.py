import pytest

from galaius_core import PcFeature, PcFeatures


@pytest.mark.parametrize(("said", "named", "scripts"), [
    ([], set(), set()),
    (["places", "screen_view"], {PcFeature.PLACES, PcFeature.SCREEN}, set()),
    # A newer runner's entry, a language no runner here knows, and a non-string are dropped, never asked about.
    (["places", "a_later_feature", "script:cobol", 7, None], {PcFeature.PLACES}, set()),
    (["script:python", "script:powershell", "script:"], set(), {"python", "powershell"}),
])
def test_a_hello_is_read_for_the_members_and_languages_it_names(said: list, named: set, scripts: set) -> None:
    features = PcFeatures.of(said)
    assert (features.named, features.scripts) == (named, scripts)
    assert all(feature in features for feature in named) and PcFeature.TOKEN_SWAP not in features
    assert PcFeatures.of(features.said()) == PcFeatures.of(features) == features | [] == features
