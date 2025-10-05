ds_helper
A powerful, reusable Python library designed to automate common data science workflows, including initial data inspection, text cleaning, and automatic data visualization. Built as a part of [Your Course Name] Project 4.

🛠️ Installation
Prerequisites: Ensure you have Python 3.8+ installed.

Clone the Repository:

git clone https://github.com/Osheen-Kumar/ds_helper
cd ds_helper_library

Install the Library: Run the setup file from the root directory.

pip install .

🚀 Usage Examples
Once installed, all key functions are available directly under the ds_helper namespace.

1. Text Cleaning (text_cleaner)
Clean a series of text data and analyze word frequency.

import pandas as pd
import ds_helper

data = pd.Series([
    "This is an amazing product, I really enjoyed it!",
    "Um... the speed was quite slow, but okay."
])

# Clean the text with lemmatization and stopword removal
cleaned_text = ds_helper.clean_text(
    data,
    lemmatize=True,
    remove_fillers=True
)

# Create and plot the word frequency (plots top 10 by default)
ds_helper.create_word_frequency(cleaned_text, top_n=10)

2. Column Detection (column_detector)
Automatically identify column types (categorical, numerical, text, etc.) in a DataFrame.

import pandas as pd
import ds_helper

df = pd.DataFrame({
    'Age': [25, 30, 45, 22],
    'City': ['NY', 'LA', 'NY', 'SF'],
    'Review': ['Great!', 'Bad!', 'Fine', 'Awful']
})

# Detect column types
types_map = ds_helper.detect_column_types(df)

print(types_map)
# Expected Output: {'Age': 'Numerical', 'City': 'Categorical', 'Review': 'Text'}

3. Automatic Visualization (auto_visualizer)
Generate appropriate visualizations based on the detected column types.

# Assuming the same 'df' as above
import ds_helper

# This will generate plots (e.g., histogram for Age, bar chart for City)
ds_helper.create_visualizations(df, types_map)

⚙️ Dependencies
This library requires the following:

pandas

nltk

matplotlib

seaborn

numpy