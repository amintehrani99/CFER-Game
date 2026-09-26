import os
import re
import pandas as pd
import kagglehub
import streamlit as st
import nltk
from nltk.corpus import wordnet

# Download required NLTK data (handles downloading if not present)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4.zip')
    except LookupError:
        nltk.download('omw-1.4')

download_nltk_data()

@st.cache_data
def load_cefr_dictionary():
    """
    Downloads and caches the CEFR dataset from Kaggle, returning a fast-lookup dictionary.
    """
    try:
        dataset_path = kagglehub.dataset_download("nezahatkk/10-000-english-words-cerf-labelled")
        csv_file = None
        for file in os.listdir(dataset_path):
            if file.endswith('.csv'):
                csv_file = os.path.join(dataset_path, file)
                break
                
        if not csv_file:
            return None
            
        df = pd.read_csv(csv_file)
        
        # Detect columns dynamically or use defaults
        word_col, level_col = None, None
        for col in df.columns:
            if 'word' in col.lower() or 'lemma' in col.lower():
                word_col = col
            if 'level' in col.lower() or 'cefr' in col.lower() or 'cerf' in col.lower():
                level_col = col
                
        if not word_col or not level_col:
            word_col = df.columns[0]
            level_col = df.columns[1]

        cefr_dict = {}
        for _, row in df.iterrows():
            word = str(row[word_col]).strip().lower()
            level = str(row[level_col]).strip().upper()
            if re.match(r'^[A-C][1-2]$', level):
                 cefr_dict[word] = level
                 
        return cefr_dict
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def process_text(text):
    if not text: return []
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower()) 
    return cleaned_text.split()

def analyze_text_cefr(text, cefr_dict):
    levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    results = {level: set() for level in levels}
    results['Uncategorized'] = set() 
    
    if not cefr_dict: return results

    words = process_text(text)
    for word in words:
        level = cefr_dict.get(word)
        if level in results:
            results[level].add(word)
        else:
            results['Uncategorized'].add(word)
    return results

def get_word_definition(word):
    """Retrieves the first definition of a word using NLTK WordNet."""
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    return "Definition not available."

def main():
    st.set_page_config(page_title="CEFR Analyzer & Word Game", layout="wide")
    st.title("📚 CEFR Text Analyzer & Word Game")

    # Load CEFR Dictionary
    with st.spinner("Loading CEFR Dictionary..."):
        cefr_dict = load_cefr_dictionary()
    
    if not cefr_dict:
        st.stop()

    st.write(f"✅ Loaded {len(cefr_dict)} words into the CEFR dictionary.")
    st.divider()

    # Step 1: Text Input and Analysis
    st.header("1. Analyze Your Text")
    default_text = """Hello! This is a simple test. We are trying to understand the complexity 
    of this particular paragraph. It contains basic words, but also some 
    slightly more advanced vocabulary like 'complexity' or 'particular'.
    Artificial intelligence is a fascinating phenomenon. Let's learn!"""
    
    user_text = st.text_area("Enter English text here:", value=default_text, height=150)
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    if st.button("Analyze Text"):
        if user_text:
            st.session_state.analysis_results = analyze_text_cefr(user_text, cefr_dict)
        else:
            st.warning("Please enter some text to analyze.")

    # Step 2: Gamification - Select Level and Stage
    if st.session_state.analysis_results:
        st.divider()
        st.header("2. Word Guessing Game")
        
        results = st.session_state.analysis_results
        # Filter levels that actually have words
        available_levels = [lvl for lvl in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'] if results.get(lvl)]
        
        if not available_levels:
            st.info("No categorizable words found in the text for the game.")
            return

        col1, col2 = st.columns(2)
        with col1:
            selected_level = st.selectbox("Select a CEFR Level to Practice:", available_levels)
        
        words_in_level = list(results[selected_level])
        words_in_level.sort() # Sort alphabetically for consistency
        
        # Chunk words into stages of 5
        chunk_size = 5
        stages = [words_in_level[i:i + chunk_size] for i in range(0, len(words_in_level), chunk_size)]
        stage_names = [f"Stage {i+1} ({len(stage)} words)" for i, stage in enumerate(stages)]
        
        with col2:
            selected_stage_index = st.selectbox("Select a Stage:", range(len(stage_names)), format_func=lambda i: stage_names[i])
        
        current_stage_words = stages[selected_stage_index]

        st.subheader(f"Level {selected_level} - Stage {selected_stage_index + 1}")

        # Initialize session state for user guesses for the CURRENT stage and level
        state_key_prefix = f"guess_{selected_level}_stage_{selected_stage_index}"
        
        for i, word in enumerate(current_stage_words):
             key = f"{state_key_prefix}_{word}"
             if key not in st.session_state:
                 st.session_state[key] = ""

        # UI Rendering: Display HTML Grid
        st.write("### The Grid")
        grid_html = """
        <style>
        .word-grid { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;}
        .word-row { display: flex; gap: 5px; }
        .letter-box {
            width: 30px; height: 30px; 
            border: 1px solid #ccc; 
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-family: monospace;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
        .correct-box { background-color: #4CAF50; color: white; border-color: #4CAF50;}
        .incorrect-box { background-color: #f44336; color: white; border-color: #f44336;}
        .empty-box { background-color: #f0f0f0; }
        </style>
        <div class="word-grid">
        """

        for word in current_stage_words:
            key = f"{state_key_prefix}_{word}"
            user_guess = st.session_state[key].lower().strip()
            
            grid_html += '<div class="word-row">'
            
            # Determine row status based on user input length
            is_correct = (user_guess == word)
            
            if user_guess: # User typed something
                if is_correct:
                    # Show all letters in green
                    for letter in word:
                         grid_html += f'<div class="letter-box correct-box">{letter.upper()}</div>'
                else:
                    # Show typed letters in red up to the word length, pad with empty boxes if needed
                    display_chars = list(user_guess)
                    for i in range(len(word)):
                        if i < len(display_chars):
                            grid_html += f'<div class="letter-box incorrect-box">{display_chars[i].upper()}</div>'
                        else:
                            grid_html += f'<div class="letter-box empty-box"></div>'
            else:
                 # Empty boxes for the word length
                 for _ in word:
                     grid_html += '<div class="letter-box empty-box"></div>'
                     
            grid_html += '</div>'
            
        grid_html += '</div>'
        st.markdown(grid_html, unsafe_allow_html=True)

        # UI Rendering: Display Clues and Inputs
        st.write("### Clues & Inputs")
        for i, word in enumerate(current_stage_words):
            definition = get_word_definition(word)
            key = f"{state_key_prefix}_{word}"
            
            # Use columns to align text input with clue
            col_clue, col_input = st.columns([2, 1])
            with col_clue:
                # Add word length hint to the clue
                st.markdown(f"**Word {i+1} ({len(word)} letters):** {definition}")
            with col_input:
                st.text_input(
                    f"Guess for Word {i+1}", 
                    key=key, 
                    label_visibility="collapsed",
                    placeholder=f"Type {len(word)} letters..."
                )
                
        # Optional: Add a clear button for the current stage
        if st.button("Reset Stage Guesses"):
            for word in current_stage_words:
                 key = f"{state_key_prefix}_{word}"
                 st.session_state[key] = ""
            st.rerun()

if __name__ == "__main__":
    main()