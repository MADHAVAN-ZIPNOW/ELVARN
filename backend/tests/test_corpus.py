import corpus


def test_split_is_by_document_with_no_overlap_or_leakage():
    docs = [{"source": "t", "category": "PLAIN",
             "text": f"document number {i} talks about topic {i * 7} and subject {i * 13} in detail"}
            for i in range(400)]
    docs.append(dict(docs[0]))                               # exact duplicate
    train, val, stats = corpus.split_documents(docs)
    train_texts = {d["text"] for d in train}
    val_texts = {d["text"] for d in val}
    assert val and train
    assert not train_texts & val_texts
    assert stats["exact_duplicates_removed"] == 1
    for v in val:
        for t in train:
            a, b = corpus._shingles(v["text"]), corpus._shingles(t["text"])
            assert len(a & b) / len(a | b) < corpus.NEAR_DUP_JACCARD


def test_block_files_become_one_document_per_block(tmp_path):
    (tmp_path / "x.txt").write_text(
        "<HUMOUR>\nUser: a\nAssistant: b\n</HUMOUR>\n\n<HUMOUR>\nUser: c\nAssistant: d\n</HUMOUR>\n",
        encoding="utf-8")
    docs = corpus.load_documents(tmp_path)
    assert len(docs) == 2 and docs[0]["text"].startswith("<HUMOUR>\nUser: a")
    assert "</HUMOUR>" not in docs[0]["text"]
