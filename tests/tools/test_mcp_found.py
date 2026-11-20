"""What a search shows of an MCP server: the grammars a stranger's strings are held to, and what blocks a server."""

import pytest
from pydantic import ValidationError

from galaius_core.tools.mcp_found import McpNeed, McpPackage, McpServerFacts, exact_host, exact_version, https_host, npm_name, pypi_name


@pytest.mark.parametrize(("value", "kept"), [
    ("@scope/tool-x", True), ("plain", True),
    ("--registry=http://127.0.0.1", False), ("git+ssh://git@host/x.git", False), ("https://host/x.tgz", False), ("file:../x", False),
    ("name@^1", False), ("-g", False), ("Upper", False), ("a b", False), ("x" * 215, False),
])
def test_an_npm_name_is_a_name_never_an_option_or_a_spec(value, kept):
    assert (npm_name(value) is not None) is kept


@pytest.mark.parametrize(("value", "kept"), [("some_tool", True), ("a.b-c", True), ("--index-url=x", False), ("pkg[extra]", False), ("pkg==1", False), ("", False)])
def test_a_pypi_name_has_no_extra_specifier_or_option(value, kept):
    assert (pypi_name(value) is not None) is kept


@pytest.mark.parametrize(("value", "kept"), [("1.0.1", True), ("0.4.0-beta.1", True), ("^1.2", False), ("latest", False), ("1.x", False), (">=1", False), ("http://x/y", False)])
def test_a_version_is_one_exact_release_or_empty(value, kept):
    assert (exact_version(value) != "") is kept


@pytest.mark.parametrize(("value", "kept"), [
    ("api.example.org", True), ("Api.Example.org", True), ("a-b.example.co.uk", True),
    ("*", False), ("*.x.example.org", False), ("x.example.org:8443", False), ("10.0.0.1", False), ("192.168.1.10", False), ("[::1]", False),
    ("localhost", False), ("nas.local", False), ("router.lan", False), ("single", False), ("example.org.", False), ("a_b.example.org", False), ("user@x.example.org", False),
])
def test_an_exact_host_is_one_public_name(value, kept):
    assert (exact_host(value) is not None) is kept


@pytest.mark.parametrize("url", ["http://x.example.org/mcp", "https://user:pw@x.example.org/mcp", "https://x.example.org:8443/mcp", "https://10.1.1.1/mcp", "ftp://x.example.org"])
def test_a_hosted_address_is_https_on_an_exact_host(url):
    assert https_host(url) is None


def need(name="TOKEN", via="env", required=True):
    return McpNeed(name=name, via=via, secret=True, required=required)


@pytest.mark.parametrize(("parts", "route", "blocked"), [
    ({"package": McpPackage(kind="npm", identifier="@a/b", version="1.0.1")}, "package", ()),                                  # a data tool over stdio
    ({"remote": "https://tools.example.org/mcp"}, "remote", ()),                                                                # hosted, declares no key
    ({"package": McpPackage(kind="npm", identifier="a-b", version="2.0.0"), "needs": (need(),)}, "package", ("needs_input",)),  # needs a key: shown, not attachable
    ({"remote": "https://tools.example.org/mcp", "needs": (need("Authorization", "header"),)}, "remote", ("needs_input",)),
    ({"remote": "https://tools.example.org/mcp", "needs": (need("X-Region", "header", required=False),)}, "remote", ()),         # an optional key does not block
    ({"package": McpPackage(kind="cargo", identifier="some-crate", version="1.0.0")}, None, ("kind_not_fetched",)),
    ({"package": McpPackage(kind="oci", identifier="ghcr.io/x/y:1.0")}, None, ("kind_not_fetched",)),
    ({"package": McpPackage(kind="npm", identifier="web-tool", stdio=False)}, None, ("not_stdio",)),
    ({}, None, ("no_route",)),
    ({"remote": "https://api.anthropic.com/mcp"}, None, ("host_refused",)),                                                     # a model host is never a server's
    ({"remote": "https://galaius.ai/x"}, None, ("host_refused",)),
    ({"remote": "https://127.0.0.1/mcp"}, None, ("host_refused",)),
    ({"package": McpPackage(kind="npm", identifier="a-b"), "shipped": True}, "package", ("program_shipped",)),
    ({"package": McpPackage(kind="npm", identifier="a-b"), "unread": True}, "package", ("record_unreadable",)),
    ({"unpackaged": True}, None, ("start_not_package",)),                                                                       # a script or a binary is never run
])
def test_what_blocks_a_server_is_decided_in_one_place(parts, route, blocked):
    facts = McpServerFacts.made("x/y", "registry", **parts)
    assert (facts.route, facts.blocked, facts.attachable) == (route, blocked, route is not None and not blocked)


def test_a_package_by_a_package_route_reaches_no_host_and_a_remote_reaches_its_own():
    assert McpServerFacts.made("a/b", "registry", package=McpPackage(kind="npm", identifier="a-b"), remote="https://t.example.org/mcp").hosts == ()
    assert McpServerFacts.made("a/b", "registry", remote="https://T.example.org/mcp").hosts == ("t.example.org",)


def test_the_listing_line_holds_closed_fields_only():
    line = McpServerFacts.made("a/b", "registry", package=McpPackage(kind="npm", identifier="a-b", version="1.0.0"), needs=(need("MY_KEY"),), vouched=True).said()
    assert "npm a-b 1.0.0" in line and "MY_KEY (secret env)" in line and "vouched by its registry" in line and "not attachable: needs a key" in line


@pytest.mark.parametrize("bad", [{"kind": "npm", "identifier": "--registry=x"}, {"kind": "pypi", "identifier": "a[b]"}, {"kind": "npm", "identifier": "a-b", "version": "^1"},
                                 {"kind": "oci", "identifier": "x y"}])
def test_a_package_a_stranger_names_wrongly_is_refused(bad):
    with pytest.raises(ValidationError):
        McpPackage(**bad)


@pytest.mark.parametrize(("name", "via"), [("bad name", "env"), ("1ABC", "env"), ("X Key", "header"), ("a=b", "env")])
def test_a_need_names_a_variable_or_header(name, via):
    with pytest.raises(ValidationError):
        McpNeed(name=name, via=via, secret=False, required=True)
