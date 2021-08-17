"""
Test end-to-end
"""
from colbert.app import *


def main():

    collection = Collection('test_index')

    # docs = pd.read_csv("../test_collection.tsv", sep='\t', names=['doc'])
    # print(docs.tail())

    doc = "This was created to say that Dennis would be very happy to work at Hebbia"
    collection.add_docs(doc)
    # collection.save()

    # print("Creating index!")
    # create_index(collection)
    #
    # collection.add_docs(docs[len(docs) // 2:])
    # print("Re-creating index!")
    # create_index(collection)

    query = "Would Dennis be happy to work at Hebbia?"

    answer, top_doc = retrieve_and_answer(collection, query)
    print("TOP DOC:", top_doc)
    print("ANSWER:", answer)


if __name__ == '__main__':
    main()

