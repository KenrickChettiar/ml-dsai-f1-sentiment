import scrape_youtube_comments
import filter_regulations
from transformers import pipeline
import math

comments = filter_regulations.filter_regulations("https://www.youtube.com/watch?v=wIYPuzWCCSw&t=19s")  # Get regulation-related comments from filter_regulations.py

# Use a more robust sentiment classifier (DistilBERT-based, fine-tuned on real sentiment data)
print("Loading advanced sentiment classifier...")
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)

results = sentiment_analysis(comments)

# Separate positive and negative with scores
positive_data = [r for r in results if r['label'] == 'POSITIVE']
negative_data = [r for r in results if r['label'] == 'NEGATIVE']

positive_count = len(positive_data)
negative_count = len(negative_data)

# Extract scores
positive_scores = [r['score'] for r in positive_data]
negative_scores = [r['score'] for r in negative_data]

avg_positive_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0
avg_negative_score = sum(negative_scores) / len(negative_scores) if negative_scores else 0

# Calculate STRONG METRICS
# 1. Sentiment Polarity Index (-1 to +1)
polarity_index = (positive_count - negative_count) / len(results) if results else 0

# 2. Overall Confidence (how certain the model is)
all_scores = [r['score'] for r in results]
avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0
confidence_std = math.sqrt(sum((s - avg_confidence) ** 2 for s in all_scores) / len(all_scores)) if len(all_scores) > 1 else 0

# 3. Sentiment Intensity (strength of sentiment)
sentiment_intensity = (avg_positive_score * positive_count + avg_negative_score * negative_count) / len(results) if results else 0

# 4. Net Sentiment Ratio (% positive)
net_sentiment_ratio = (positive_count / len(results)) * 100 if results else 0

# 5. Weighted Sentiment Score
weighted_sentiment = (sum(positive_scores) - sum(negative_scores)) / len(results) if results else 0

# 6. Strong sentiment categories
strong_positive = sum(1 for s in positive_scores if s > 0.9)
moderate_positive = sum(1 for s in positive_scores if 0.7 <= s <= 0.9)
strong_negative = sum(1 for s in negative_scores if s > 0.9)
moderate_negative = sum(1 for s in negative_scores if 0.7 <= s <= 0.9)

print("=" * 70)
print("ADVANCED SENTIMENT ANALYSIS RESULTS")
print("=" * 70)
print(f"Total Comments Analyzed: {len(results)}")
print(f"\n{'DISTRIBUTION':-^70}")
print(f"  Strong Positive (>0.9):    {strong_positive:3d} ({strong_positive/positive_count*100:.1f}% of positive)" if positive_count > 0 else f"  Strong Positive (>0.9):      0")
print(f"  Moderate Positive (0.7-0.9): {moderate_positive:3d} ({moderate_positive/positive_count*100:.1f}% of positive)" if positive_count > 0 else f"  Moderate Positive (0.7-0.9):  0")
print(f"  Positive Comments Overall:   {positive_count:3d} ({net_sentiment_ratio:.1f}%)")
print(f"  Negative Comments Overall:   {negative_count:3d} ({100-net_sentiment_ratio:.1f}%)")
print(f"  Strong Negative (>0.9):    {strong_negative:3d} ({strong_negative/negative_count*100:.1f}% of negative)" if negative_count > 0 else f"  Strong Negative (>0.9):      0")
print(f"  Moderate Negative (0.7-0.9): {moderate_negative:3d} ({moderate_negative/negative_count*100:.1f}% of negative)" if negative_count > 0 else f"  Moderate Negative (0.7-0.9):  0")

print(f"\n{'STRONG METRICS':-^70}")
print(f"  Polarity Index:           {polarity_index:+.4f}  (Range: -1 to +1)")
print(f"  Sentiment Intensity:      {sentiment_intensity:.4f}  (How strong the sentiment is)")
print(f"  Weighted Sentiment Score: {weighted_sentiment:+.4f}  (Net positive/negative)")
print(f"  Model Confidence:         {avg_confidence:.4f}  (±{confidence_std:.4f} std dev)")

print(f"\n{'DETAILED AVERAGES':-^70}")
print(f"  Avg Positive Score:       {avg_positive_score:.4f}")
print(f"  Avg Negative Score:       {avg_negative_score:.4f}")

print("=" * 70)