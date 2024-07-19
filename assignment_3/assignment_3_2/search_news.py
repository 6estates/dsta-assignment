import json
from googleapiclient.discovery import build


def search_news(key_words, google_key, google_cx, type_num=1, sort=None, num=10):
    """
    type
    - 1: all
    - 2: news
    - 3: website
    """
    assert sort is None or sort in ["date", "relevance"]
    if sort == "relevance":
        sort = None
    assert type_num in [1,2,3]
    service = build(
        "customsearch", "v1", developerKey=google_key
    )
    
    if type_num == 2:
        query = key_words + " more:pagemap:metatags-og_type:article OR more:pagemap:metatags-og_type:news"
    elif type_num == 3:
        query = key_words + " more:pagemap:metatags-og_type:website"
    else:
        query = key_words

    res = (
        service.cse()
        .list(
            q=query,
            sort=sort,
            num=num,
            cx=google_cx,
        )
        .execute()
    )

    result = json.dumps(res['items'])
    return result
