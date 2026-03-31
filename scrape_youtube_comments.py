from itertools import islice
from youtube_comment_downloader import *

def scraper(url):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
    result = []
    for comment in islice(comments, 500):
        result.append(comment['text'])
    return result