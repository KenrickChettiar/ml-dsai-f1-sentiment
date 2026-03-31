import csv
import json
import scrape_youtube_comments
from transformers import pipeline


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") # classifier to assess the type of comment (regulation-related or not)
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True
) # model to assess the sentiment of a comment

typeofcomment = ["about F1 regulations", "about something else"]


def analyse_url(url, label):
    print(f"\n{'=' * 70}")
    print(f"Processing: {label} — {url}")
    print("=" * 70)


    raw_comments = scrape_youtube_comments.scraper(url)
    print(f"Scraped {len(raw_comments)} comments.") #scraping the youtube comments

    regulation_comments = []
    for i, comment in enumerate(raw_comments): #loop to filter out comments that are only about F1 regulations
        result = classifier(comment, typeofcomment)
        if result['labels'][0] == "about F1 regulations" and result['scores'][0] > 0.5:
            regulation_comments.append(comment)
    print(f"Regulation comments found: {len(regulation_comments)}")

    #sentiment analysis of the regulation-related comments
    results = sentiment_analysis(regulation_comments)

    positive_data = [r for r in results if r['label'] == 'POSITIVE']
    negative_data = [r for r in results if r['label'] == 'NEGATIVE']
    positive_scores = [r['score'] for r in positive_data]
    negative_scores = [r['score'] for r in negative_data]

    pos_count = len(positive_data)
    neg_count = len(negative_data)
    total = len(results)

    return {
        "url": url,
        "label": label,
        "total_scraped": len(raw_comments),
        "regulation_comments": total,
        "sentiment": {
            "positive_count": pos_count,
            "negative_count": neg_count,
            "polarity_index": round((pos_count - neg_count) / total, 4),
            "weighted_sentiment_score": round(
                (sum(positive_scores) - sum(negative_scores)) / total, 4
            ),
        }
    } # storing the results in a dictionary which will be saved in a JSON file


def load_urls(csv_path): # Loading the youtube videos URLs from a CSV file.
    urls = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append((row['url'].strip(), row.get('label', row['url']).strip()))
    return urls


def main(csv_path="urls_2022.csv", output_path="results_2022.json"):
    urls = load_urls(csv_path)
    print(f"Loaded {len(urls)} URLs from {csv_path}\n")

    all_results = []
    failed = []

    for i, (url, label) in enumerate(urls):
        print(f"\n[{i + 1}/{len(urls)}] {label}")
        result = analyse_url(url, label)
        if result:
            all_results.append(result)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Saved progress to {output_path}")
            
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()