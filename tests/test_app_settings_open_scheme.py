"""F18-8: the Settings pane hands only http(s) URLs and plain paths to the shell."""

from __future__ import annotations

import pytest

from realmspinner.studio.modes.settings.ui.panes import app_settings


@pytest.fixture
def opened(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("os.startfile", calls.append, raising=False)
    return calls


@pytest.mark.parametrize(
    "target",
    ["file:///C:/Windows/System32/calc.exe", "ms-msdt:/id x", "javascript:alert(1)", "ftp://x/y"],
)
def test_open_refuses_a_url_scheme_that_is_not_http_or_https(opened, target) -> None:
    with pytest.raises(ValueError):
        app_settings._open(target)
    assert opened == []


@pytest.mark.parametrize(
    "target",
    [
        "https://example.com/notes",
        "HTTP://example.com/x",
        r"C:\Users\me\RealmspinnerSetup.exe",
        "/tmp/x",
    ],
)
def test_open_still_passes_http_urls_and_local_paths(opened, target) -> None:
    app_settings._open(target)
    assert opened == [target]
