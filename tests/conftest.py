import spacy
import pytest
from sayswho import SaysWho
import requests
from tqdm import tqdm
import gzip
import tarfile
import os

@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_lg")

@pytest.fixture(scope="module")
def load_ner_nlp():
    # TODO: smarter toggle when already downloaded
    url = "http://philadelphyinz.com/drop/nba-model-best.tar.gz"
    file_path = "./ner-model.tar.gz"
    output_dir = "./ner_model/"

    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        progress = tqdm(total=total_size, unit='B', unit_scale=True)
        
        with open(file_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress.update(len(data))

        progress.close() 
    
    if os.path.exists(file_path) and not os.path.exists(output_dir):
        with gzip.open(file_path, 'rb') as gz_file:
            with tarfile.open(fileobj=gz_file, mode='r') as tar:
                tar.extractall(path='./ner_model/')
    return os.path.abspath(output_dir)

    # ner_nlp = spacy.load("./ner-model.tar.gz")
    # return ner_nlp

@pytest.fixture(scope="module")
def says_who():
    # test_text = open("./tests/qa_test_file.txt").read()
    sw = SaysWho()
    return sw