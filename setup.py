from setuptools import setup, find_packages

# --- 1. Define the Dependency List ---
# This function reads the requirements.txt file to get the list of dependencies.
def get_requirements():
    # It ensures only package names (not comments or empty lines) are included.
    with open('requirements.txt') as f:
        # Returns a list of strings, e.g., ['pandas>=1.0.0', 'nltk>=3.5']
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# --- 2. Define the Long Description (for PyPI/Documentation) ---
# This reads the entire README.md content. This is the source of the descriptive text.
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ds_helper',
    version='0.1.0',
    author='Data Science Students',
    description='A reusable Python library for automated data science analysis, cleaning, and visualization.',
    
    # This is the documentation text (from README.md)
    long_description=long_description, 
    long_description_content_type='text/markdown',
    
    url='https://github.com/Osheen-Kumar/ds_helper',
    packages=find_packages(),
    
    # --- 3. Pass the *clean* dependency list to install_requires ---
    # This MUST be a list of valid package strings, which get_requirements() provides.
    install_requires=get_requirements(), 
    
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha'
    ],
    python_requires='>=3.8',
)
