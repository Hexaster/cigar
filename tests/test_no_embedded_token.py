from pathlib import Path


def test_no_tabpfn_token_is_embedded():
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py"))
    )

    assert "set_access_token('eyJ" not in text
    assert 'set_access_token("eyJ' not in text
