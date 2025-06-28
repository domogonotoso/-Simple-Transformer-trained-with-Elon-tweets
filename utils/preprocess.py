import csv
import re
import os

EOS_TOKEN = "<|endoftext|>"

def clean_text(text: str) -> str:
    """Clean a single line of tweet text."""
    text = text.lower()
    text = re.sub(r'<\|endoftext\|>', '', text)       # Remove special token if any
    text = re.sub(r'http\S+|www\.\S+', '', text)       # Remove URLs
    text = re.sub(r'@\w+', '', text)                  # Remove mentions
    text = re.sub(r'#\w+', '', text)                  # Remove hashtags
    text = re.sub(r'&\w+;', '', text)                 # Remove HTML entities like &amp;
    text = re.sub(r'[^a-zA-Z가-힣\s]', '', text)       # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()          # Normalize whitespace
    return text

def preprocess(input_csv: str, output_txt: str, min_length: int = 5):
    """Extract and clean tweet text from CSV, write to output file with EOS tokens."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    count = 0
    with open(input_csv, encoding='utf-8') as f_in, open(output_txt, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            raw_text = row.get('text', '')
            if raw_text:
                cleaned = clean_text(raw_text)
                if cleaned and len(cleaned.split()) >= min_length:
                    f_out.write(cleaned + f' {EOS_TOKEN}\n')
                    count += 1

    print(f"✅ Preprocessing complete. {count} lines written to {output_txt}")

if __name__ == "__main__":
    preprocess('data/elon_musk_tweets.csv', 'data/cleaned.txt')
