import pandas as pd
import re
import string
import nltk
# Updated import to include wordnet constants needed for POS-aware lemmatization
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from typing import List, Optional
import matplotlib.pyplot as plt
from collections import Counter

# --- Configuration for Filler Words ---
# Common "filler" or non-lexical words to be removed
FILLER_WORDS = set([
    "uh", "um", "like", "you know", "ah", "er", "hmm", "well",
    "so", "right", "okay", "i mean", "sort of", "kind of"
])

def _get_wordnet_pos(tag):
    """
    Converts Treebank POS tags (used by nltk.pos_tag) to WordNet POS tags
    (needed by WordNetLemmatizer).
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    # Default to NOUN (n) if not recognized
    return wordnet.NOUN

def _download_nltk_data() -> None:
    """
    Downloads necessary NLTK data (stopwords, wordnet, averaged_perceptron_tagger) 
    if they haven't been downloaded already. Removed 'punkt' dependency check.
    """
    try:
        # Check if necessary data is available
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        # Required for POS tagging, which ensures accurate lemmatization
        nltk.data.find('taggers/averaged_perceptron_tagger') 
    except LookupError: # LookupError is correctly raised when a resource is missing
        print("NLTK data not found. Downloading necessary components (stopwords, wordnet, tagger)...")
        # Download necessary NLTK packages
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        # Note: 'punkt' download is no longer required due to switching to regex tokenization, 
        # but 'averaged_perceptron_tagger' is still needed for POS tagging in lemmatization.
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception:
        # Catch other potential errors
        print("An unexpected error occurred during NLTK data check/download.")

def _clean_single_text(
    text: str,
    remove_punct: bool,
    remove_stopwords: bool,
    remove_fillers: bool,
    lowercase: bool,
    lemmatize: bool,
    custom_stopwords: Optional[List[str]],
    min_word_length: int,
) -> str:
    """
    Internal helper function to apply all cleaning steps to a single text string.
    """
    if not isinstance(text, str):
        return ""

    if lowercase:
        text = text.lower()

    # Simple cleanup of non-ASCII characters (like emojis) before tokenization
    # Replaces non-ASCII characters with a space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 1. TOKENIZATION: Use regex to extract alphanumeric tokens (words/numbers)
    # The pattern matches word characters (alphanumeric and underscore)
    tokens = re.findall(r'\b\w+\b', text)


    # Get the combined set of stopwords
    all_stopwords = set()
    if remove_stopwords:
        all_stopwords.update(stopwords.words('english'))
    if remove_fillers:
        # Custom filler words are generally cleaned *after* lowercasing and tokenization
        all_stopwords.update(FILLER_WORDS)
    if custom_stopwords:
        all_stopwords.update([word.lower() for word in custom_stopwords])

    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    
    # --- Robust POS Tagging / Iteration Setup ---
    if lemmatize:
        try:
            # Generate POS tags for tokens (needed for accurate lemmatization)
            tagged_tokens = nltk.pos_tag(tokens)
        except LookupError:
            # If the tagger resource is missing, fall back to default noun tagging
            print("Warning: NLTK POS Tagger resource missing. Falling back to default (noun) tags for lemmatization.")
            tagged_tokens = [(t, 'NN') for t in tokens]
    else:
        # If no lemmatization is requested, we still iterate over tokens, using a dummy tag
        # The tag here doesn't matter since lemmatize=False
        tagged_tokens = [(t, 'NN') for t in tokens]


    punctuation_table = str.maketrans('', '', string.punctuation)

    for original_token, tag in tagged_tokens:
        token = original_token
        
        # 2. IMMEDIATE STOPWORD/FILLER CHECK (using original token)
        # IMPORTANT FIX: This ensures stopwords are removed even if lemmatization 
        # is requested and changes the stopword into a non-stopword form (e.g. 'was' -> 'wa').
        if remove_stopwords or remove_fillers or custom_stopwords:
            if token in all_stopwords:
                continue
        
        # 3. Remove Punctuation
        if remove_punct:
            # Apply punctuation removal to the token used in subsequent steps
            token = token.translate(punctuation_table)

        # Skip if the token is empty
        if not token:
            continue
        
        # 4. FILTER: Remove non-alphabetic tokens (e.g., numbers)
        if not token.isalpha():
            continue

        # 5. Lemmatization
        if lemmatize:
            pos = _get_wordnet_pos(tag)
            # Use the determined POS tag for accurate lemmatization (e.g., 'was' -> 'be')
            token = lemmatizer.lemmatize(token, pos=pos) 

        # 6. Minimum Word Length Filter
        if len(token) < min_word_length:
            continue

        cleaned_tokens.append(token)

    # Rejoin the tokens into a clean string
    return " ".join(cleaned_tokens)


def clean_text(
    text_series: pd.Series,
    remove_punct: bool = True,
    remove_stopwords: bool = True,
    remove_fillers: bool = True,
    lowercase: bool = True,
    lemmatize: bool = False,
    custom_stopwords: Optional[List[str]] = None,
    min_word_length: int = 2
) -> pd.Series:
    """
    Applies a comprehensive set of cleaning and normalization steps to a pandas Series of text.

    Args:
        text_series: A pandas Series containing the text data to be cleaned.
        remove_punct: If True, removes common punctuation.
        remove_stopwords: If True, removes standard NLTK English stopwords.
        remove_fillers: If True, removes predefined filler words (e.g., 'uh', 'um').
        lowercase: If True, converts all text to lowercase.
        lemmatize: If True, reduces words to their root form (requires NLTK download).
        custom_stopwords: Optional list of additional strings to remove as stopwords.
        min_word_length: Minimum number of characters a word must have to be kept.

    Returns:
        A pandas Series containing the cleaned text.
    """
    _download_nltk_data()

    # Apply the single-text cleaning function to every element in the Series
    # lambda function passes all arguments to the internal helper
    cleaned_series = text_series.apply(
        lambda x: _clean_single_text(
            x,
            remove_punct,
            remove_stopwords,
            remove_fillers,
            lowercase,
            lemmatize,
            custom_stopwords,
            min_word_length
        )
    )
    return cleaned_series

def _plot_word_frequency(freq_df: pd.DataFrame) -> None:
    """
    Internal helper function to plot the word frequency as a horizontal bar chart.
    """
    print("\n--- Plotting Word Frequency (Displaying Top N) ---")
    if freq_df.empty:
        print("No words found to plot.")
        return

    # Use a try-except block in case plotting environment fails (e.g., non-GUI env)
    try:
        # Use a professional style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create plot
        fig, ax = plt.subplots(figsize=(10, len(freq_df) * 0.4 + 1))
        words = freq_df['Word']
        counts = freq_df['Frequency']

        ax.barh(words, counts, color='#2c7bb6')

        # Set labels and title
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(f'Top {len(freq_df)} Word Frequencies', fontsize=14, fontweight='bold')

        # Invert y-axis to have the most frequent word at the top
        ax.invert_yaxis()

        # Add counts next to the bars
        for i, count in enumerate(counts):
            ax.text(count + max(counts) * 0.01, i, str(count), va='center', fontsize=10)

        # Remove chart junk
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Matplotlib plotting failed (possibly non-GUI environment). Error: {e}")
        print("\n--- Word Frequency Dataframe ---")
        # CHANGED: Use to_string() instead of to_markdown() to avoid the 'tabulate' dependency
        print(freq_df.to_string(index=False))


def create_word_frequency(
    text_series: pd.Series,
    top_n: int = 20,
    plot: bool = True
) -> pd.DataFrame:
    """
    Calculates and optionally plots the frequency of words in a text series.

    Args:
        text_series: A pandas Series containing the cleaned text data.
        top_n: The number of most frequent words to return and plot.
        plot: If True, generates a plot of the top_n words.

    Returns:
        A pandas DataFrame with 'Word' and 'Frequency' columns.
    """
    if text_series.empty:
        return pd.DataFrame({'Word': [], 'Frequency': []})

    # Join all text into one string and split into individual words
    all_text = " ".join(text_series.astype(str).tolist())
    all_words = all_text.split()

    # Use Counter to get word frequencies
    word_counts = Counter(all_words)

    # Get the top N most common words
    top_words = word_counts.most_common(top_n)

    # Convert to DataFrame
    freq_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

    if plot:
        _plot_word_frequency(freq_df)

    return freq_df

# --- Test Case and Example Usage ---
if __name__ == "__main__":
    print("--- Running ds_helper Text Cleaner Module Test ---")

    # 1. Define Sample Data
    raw_data = {
        'id': [1, 2, 3, 4],
        'review_text': [
            "Wow! I really enjoyed this movie. It was amazing, like, totally amazing!",
            "I mean, the plot was good, but the acting, uh, was quite weak. 😔",
            "The product is amazing; fast delivery and great quality. 9/10.",
            "Um... I did not like the main character. He was too short."
        ]
    }
    df = pd.DataFrame(raw_data)
    text_to_clean = df['review_text']

    # 2. Test Case 1: Standard Cleaning (Punctuation, Stopwords, Fillers, Lowercase)
    print("\n[TEST 1: Standard Cleaning with No Lemmatization]")
    cleaned_series_standard = clean_text(
        text_to_clean,
        lemmatize=False,
        remove_fillers=True,
        remove_stopwords=True
    )
    df['cleaned_standard'] = cleaned_series_standard
    # Use to_string() for reliable output without extra dependencies
    print(df[['review_text', 'cleaned_standard']].to_string(index=False))

    # Expected transformation with the new isalpha() filter:
    # - "The product is amazing; fast delivery and great quality. 9/10." -> '910' is removed!

    # 3. Test Case 2: Advanced Cleaning (WITH Lemmatization) and Custom Stopwords
    print("\n[TEST 2: Advanced Cleaning with Lemmatization and Custom Stopwords]")
    custom_stops = ["movie", "product", "character", "delivery"]
    cleaned_series_advanced = clean_text(
        text_to_clean,
        lemmatize=True,  # Enable lemmatization (e.g., 'enjoyed' -> 'enjoy', 'was' -> 'be')
        remove_fillers=False, # Keep fillers for this test
        custom_stopwords=custom_stops
    )
    df['cleaned_advanced'] = cleaned_series_advanced
    # Use to_string() for reliable output without extra dependencies
    print(df[['review_text', 'cleaned_advanced']].to_string(index=False))
    
    # Expected transformation with the new isalpha() filter and POS lemmatization:
    # - 'was' -> 'be' (or now, simply removed)
    # - '9/10' -> removed

    # 4. Test Case 3: Word Frequency Analysis and Plotting (using Standard Cleaned Data)
    print("\n[TEST 3: Word Frequency Analysis]")
    freq_df = create_word_frequency(
        df['cleaned_standard'],
        top_n=10,
        plot=True  # Will call _plot_word_frequency to display the chart
    )

    print("\nTest Complete.")
