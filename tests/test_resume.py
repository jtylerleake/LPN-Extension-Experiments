from src.utils.resume import append_jsonl_row, completed_fingerprints, fingerprint, read_jsonl_rows


def test_fingerprint_is_order_independent():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_truncated_last_line_is_dropped_and_cut(tmp_path):
    path = tmp_path / "m.jsonl"
    append_jsonl_row(path, {"fingerprint": "x", "v": 1})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"fingerprint": "y", "v"')
    assert completed_fingerprints(read_jsonl_rows(path)) == {"x"}
    append_jsonl_row(path, {"fingerprint": "z"})
    assert completed_fingerprints(read_jsonl_rows(path)) == {"x", "z"}
