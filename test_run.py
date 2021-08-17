"""
Test end-to-end
"""
import requests
from time import time

APP_URL   = 'http://localhost:5000'
INDEX_URL = f"{APP_URL}/index"


def color_text(t: str) -> str:
    return f'\033[96m{t}\033[0m'


def main():

    timer = time()

    assert requests.get(APP_URL).ok
    done_blurb = "done! ✅\n\n"
    # Create and add documents to a collection
    collection_name = 'dummy'
    docs = [
        'This will be our first document woohoo',
        'I think we should hack the planet!'
    ]

    print(f"Creating a new index with {len(docs)} docs...", end='')
    r = requests.post(url=INDEX_URL + '/create',
                      json={
                          'name': collection_name,
                          'docs': docs
                      })
    assert r.ok
    print(done_blurb)

    # Let's ask a question
    print("Asking a question...", end='')
    query = 'What should we do?'
    r = requests.post(url=INDEX_URL + '/search',
                      json={
                          'name': collection_name,
                          'query': query
                      })

    assert r.ok
    print(done_blurb)

    d = r.json()
    print(f"TOP DOC: {d['doc']}\n"
          f"ANSWER : {color_text(d['answer'])}")

    # Add a new document
    doc = "This was created to say that Dennis would be very happy to work at Hebbia"
    print("Adding a new document...", end='')
    r = requests.post(url=INDEX_URL + '/add',
                      json={
                          'name': collection_name,
                          'docs': doc
                      })
    assert r.ok
    print(done_blurb)

    # Finally, ask another question
    print("Asking final (most important!) question...", end='')
    query = "Would Dennis be happy to work at Hebbia?"
    r = requests.post(url=INDEX_URL + '/search',
                      json={
                          'name': collection_name,
                          'query': query
                      })
    assert r.ok
    print(done_blurb)

    d = r.json()
    print(f"TOP DOC: {d['doc']}\n"
          f"ANSWER : {color_text(d['answer'])}")

    timer = time() - timer
    print(f"All done in {round(timer, 4)} seconds!")


if __name__ == '__main__':
    main()

