import os
from pathlib import Path
import shlex
from typing import Optional, List, Union, Tuple

import pandas as pd
from flask import Flask, request, jsonify
from transformers import pipeline

from .utils.utils import create_directory
from .index import get_parser as index_parser, run as run_index, CollectionEncoder
from .index_faiss import get_parser as faiss_index_parser, run as run_faiss_index
from .indexing.faiss_index import FaissIndex
import faiss
from tqdm import tqdm
from .ranking.retrieval import search_engine_retrieve
from .retrieve import get_parser as retrieve_parser

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

app = Flask('hebbia-hack')

dir_path = Path(__file__).parents[2]    # Need two directories up

ROOT = os.path.join(dir_path, "experiments/")
INDEX_ROOT = os.path.join(dir_path, "indexes/")
COLLECTIONS = os.path.join(dir_path, "collections/")
# EXPERIMENT = "MSMARCO-psg"
PADDING = '_'


class SimpleIndex:

    @property
    def exists(self) -> bool:
        raise NotImplementedError()

    def add(self, items: list):
        raise NotImplementedError()

    def get(self, ids: list):
        raise NotImplementedError()


class Collection:
    """ Simple class to stop code repeating """
    def __init__(self,
                 name: str,
                 collections_dir: str = COLLECTIONS,
                 padding: str = PADDING,
                 root: str = ROOT,
                 index_dir: str = INDEX_ROOT,
                 partitions = 1):
        self.name = name
        self.collections_dir = collections_dir
        self.root = root
        self.index_dir = os.path.join(index_dir, self.name)
        self.partitions = partitions

        self.__doc_df = None
        self.padding = padding

        self.__index = None
        self.__faiss_index = None

    @property
    def fp(self):
        return os.path.join(self.collections_dir, f"{self.name}.tsv")

    @property
    def doc_df(self) -> pd.DataFrame:
        if self.__doc_df is not None:
            return self.__doc_df
        elif os.path.exists(self.fp):
            self.__doc_df = pd.read_csv(self.fp, sep='\t', names=['doc'])
        else:
            self.__doc_df = pd.DataFrame(columns=['doc'])

        return self.__doc_df

    @property
    def exists(self) -> bool:
        """
        Determines if collection exists
        Two ways of checking:
        1. Check if file exists
        2. If the file exists and len(doc_df) > 0 -> True (else, False)
        :return:
        :rtype:
        """
        return os.path.exists(self.fp)

    def save(self):
        self.__doc_df.to_csv(self.fp, sep='\t', header=False)

    def add_docs(self, docs: Union[str, List[str]]):
        if isinstance(docs, str):
            docs = [docs]
        new_df = pd.DataFrame(docs, columns=['doc'])
        self.__doc_df = pd.concat([self.doc_df, new_df], axis=0, ignore_index=True)

        if len(self.__doc_df) < 256 and not self.index_exists:
            print("INDEX DOES NOT EXIST PADDING")
            diff = 256 - len(self.doc_df)
            self.add_docs([self.padding] * diff)
            return

        self.save()
        self.add_to_index(docs)

    def add_to_index(self, docs: List[str]):

        # Make sure we have somewhere to save to
        create_directory(self.index_dir)

        if not os.path.exists(self.fp):
            print("😳 Collection does not yet exist! This will be creation of index!")

        print(f"💡 Embedding {len(docs)} docs...")
        embs = self.index.add_docs(docs)
        print("✅ Done! Adding to FAISS index...")

        embs = embs.float().numpy()

        if not self.faiss_index_exists:
            self.faiss_index.train(embs)

        self.faiss_index.add(embs)
        self.faiss_index.save(self.faiss_index_fp)
        print("✅ Done!")

    def get_docs(self, doc_ids: Union[int, List[int]]):
        if isinstance(doc_ids, int):
            doc_ids = [doc_ids]

        valid_ids = set(doc_ids).intersection(self.doc_df.index.tolist())
        if len(valid_ids):
            return self.doc_df.loc[valid_ids, 'doc'].values.tolist()

    def remove_docs(self, doc_ids: Union[int, List[int]]):
        if isinstance(doc_ids, int):
            doc_ids = [doc_ids]

        valid_ids = set(doc_ids).intersection(self.doc_df.index.tolist())
        if len(valid_ids):
            self.doc_df.drop(valid_ids)

    """
    Embedding index
    """

    @property
    def index_exists(self) -> bool:
        return os.path.exists(self.index_dir)

    @property
    def index(self):
        bsize = min(len(self.doc_df), 256)

        arg_str = f"""
            --amp --doc_maxlen 180 --mask-punctuation --bsize {bsize} \
            --checkpoint a \
            --collection  {self.fp} \
            --index_root {INDEX_ROOT} \
            --index_name {self.name} \
            --root {self.root} \
            --experiment {self.name}
            """

        parser = index_parser()
        args = parser.parse(shlex.split(arg_str))
        args.index_path = os.path.join(args.index_root, args.index_name)

        return CollectionEncoder(args, process_idx=0, num_processes=args.nranks)

    """
    FAISS index
    """

    @property
    def faiss_index_fp(self):
        partitions_info = '' if self.partitions is None else f'.{self.partitions}'
        range_info = ''
        return os.path.join(self.index_dir, f'ivfpq{partitions_info}{range_info}.faiss')

    @property
    def faiss_index_exists(self) -> bool:
        return os.path.exists(self.faiss_index_fp)

    @property
    def faiss_index(self):
        if self.__faiss_index is not None:
            return self.__faiss_index
        else:
            self.__faiss_index = FaissIndex(128, 1)
            if self.faiss_index_exists:
                self.__faiss_index.index = faiss.read_index(self.faiss_index_fp)

        return self.__faiss_index


def create_index(collection: Collection):

    bsize = min(len(collection.doc_df), 256)

    arg_str = f"""
    --amp --doc_maxlen 180 --mask-punctuation --bsize {bsize} \
    --checkpoint a \
    --collection  {collection.fp} \
    --index_root {INDEX_ROOT} \
    --index_name {collection.name} \
    --root {ROOT} \
    --experiment {collection.name}
    """

    parser = index_parser()
    args = parser.parse(shlex.split(arg_str))
    run_index(args)

    arg_str = f"""
    --index_root {INDEX_ROOT}
    --index_name {collection.name}
    --sample 1.0
    --root {ROOT}
    --partition 1
    --experiment {collection.name}
    """
    parser = faiss_index_parser()
    args = parser.parse(shlex.split(arg_str))
    run_faiss_index(args)


def index_req(name: str, docs: Union[str, List[str]], mode: str):
    """
    Handles all index requests dependent on `mode`
    :param req: HTTP request transformed into dict
    :param mode: determines what is being done to index
    :return:
    """
    assert mode in ['add', 'create', 'delete'], f"Invalid mode={mode}"
    if isinstance(docs, str):
        docs = [docs]

    # TODO save docs to TSV
    # Save as ID\tDOC

    collection = Collection(name)

    # collection_fp = os.path.join(COLLECTIONS, f"{name}.tsv")
    if mode == 'create' and os.path.exists(collection.fp):
        raise FileExistsError(f"{name} collection already exists!")

    if mode in ['add', 'delete'] and not os.path.exists(collection.fp):
        raise FileNotFoundError(f"{name} collection does not exist!")

    if mode == 'delete':
        collection.remove_docs(docs)
    else:
        collection.add_docs(docs)

    # FIXME for mode='add' we can be smarter then re-creating index
    collection.save()
    create_index(collection)


@app.route("/index/create", methods=['POST'])
def add_index():
    req: dict = jsonify(request.json)
    index_req(req['name'], req['docs'], mode='create')


@app.route("/index/add", methods=['POST'])
def add_document():
    req: dict = jsonify(request.json)
    index_req(req['name'], req['docs'], mode='add')


@app.route('/index/delete')
def remove_document():
    req: dict = jsonify(request.json)
    index_req(req['name'], req['docs'], mode='delete')


def get_top_doc(query: str, collection: Collection) -> List[Tuple[int, float]]:
    """
    Get most relevant document from a collection for a given query
    :param query:
    :param collection_name:
    :return: list of tuples containing (pid, score)
    """

    arg_str = f"""
            --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
            --checkpoint a \
            --collection  {collection.fp} \
            --index_root {INDEX_ROOT} \
            --index_name {collection.name} \
            --root {ROOT} \
            --experiment {collection.name} \
            --partitions 1 \
            --nprobe 32 
            """

    args = retrieve_parser().parse(shlex.split(arg_str))
    pids, scores = search_engine_retrieve(query, args)

    return list(zip(pids, scores))


def get_answer(question: str, text: str, model: Optional[str] = 'distilbert-base-uncased-distilled-squad') -> str:
    """
    Get the answer to a question in a piece of text using Hugginface's QA pipeline
    :param question:
    :param text:
    :param model:
    :return:
    """

    # TODO in the future we could choose our model via (you guessed it) model arg
    nlp = pipeline('question-answering', model=model)

    try:
        resp = nlp(question=question, context=text)
    except KeyError:
        # Unable to find sufficient
        resp = None

    return resp


def retrieve_and_answer(collection, query) -> Tuple[str, str]:
    results = get_top_doc(query=query, collection=collection)

    # Remove padding from results!
    final_results = []
    for pid, score in results:
        doc = collection.get_docs(pid)[0]
        if doc != collection.padding:
            final_results.append(doc)

    # We can iterate over the top K results to see which gives us the best answer
    top_k = 1
    top_ans = None
    top_doc = None
    top_score = -1

    for doc in tqdm(final_results[:top_k]):
        resp = get_answer(query, doc)

        if resp is not None and resp['score'] > top_score:
            top_score = resp['score']
            top_doc = doc
            top_ans = resp['answer']

    return top_ans, top_doc


@app.route('/index/search')
def search():
    req: dict = jsonify(request.json)
    name  = req['name']
    query = req['query']

    collection = Collection(name)
    if not collection.exists:
        raise FileNotFoundError(f"collection does not exist!")

    return retrieve_and_answer(collection, query)
