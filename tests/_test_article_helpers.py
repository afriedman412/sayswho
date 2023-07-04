import pytest
from sayswho.article_helpers import full_parse, load_doc
from sayswho.constants import json_path

def test_load_data():
    doc_id = "5SGV-F7D1-DYT5-M4YY-00000-00"
    data = load_doc(doc_id)
    t = full_parse(data, "\n")
    import os
    print("test", os.getcwd())
    t_ = open("./tests/qa_test_file.txt").read()

    assert t == t_
    

