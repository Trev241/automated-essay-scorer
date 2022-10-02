import pandas as pd
import enchant
import re
import warnings
import language_tool_python
import spacy

from textblob import TextBlob

from readability import Readability
from readability.exceptions import ReadabilityException

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
for dependency in (
    'vader_lexicon',
):
    nltk.download(dependency)

warnings.filterwarnings('ignore')
d = enchant.Dict('en_US')
nlp = spacy.load('en_core_web_sm')
tool = language_tool_python.LanguageTool('en-US')

data = pd.read_csv('data/training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
custom_data = pd.read_csv('data/custom_input.tsv', sep='\t')
data.reset_index()

f = open('data/prompts.txt', 'r')
ESSAY_PROMPTS = {essay_set + 1: prompt for essay_set, prompt in enumerate(f)}
f.close()            
data['essay_prompt'] = [ESSAY_PROMPTS.get(entry['essay_set'], 'CUSTOM INPUT') for _, entry in data.iterrows()]

DEBUGGING_SET = 1
ESSAY_SET = {
    # 1-8 to select corresponding set, 0 to select first five essays of the debugging set only
    0: data[data['essay_set'] == DEBUGGING_SET].head(5),
    1: data[data['essay_set'] == 1],
    2: data[data['essay_set'] == 2],
    3: data[data['essay_set'] == 3],
    4: data[data['essay_set'] == 4],
    5: data[data['essay_set'] == 5],
    6: data[data['essay_set'] == 6],
    7: data[data['essay_set'] == 7],
    8: data[data['essay_set'] == 8]
}

NORMALIZED_SCALE = 4
MAX_ACHIEVABLE_SCORE = {
    1: 12,
    2: 6,
    3: 3,
    4: 3,
    5: 4,
    6: 4,
    7: 30,
    8: 60
}

PROGRESS_BAR_LENGTH = 50

def process(essay_set=None, custom_input=False, debug_text=False):
    if custom_input:
        datasubset = custom_data
        # Assuming that all custom data belong to the same set
        essay_set = custom_data['parent_set'][0]
    else:
        datasubset = ESSAY_SET[essay_set]
        # Adjust essay set to 1 to prevent key errors in case 0 was selected
        essay_set = essay_set if essay_set >= 1 and essay_set <= 8 else DEBUGGING_SET
    
    # --FEATURES--
    lexical_diversity = []
    correct_spellings = []
    spelling_errors = []
    grammar_errors = []
    
    pos_frequency_by_essay = {
        'NOUN': [],
        'CONJ': [], # Conjunctions (and, or, but). Deprecated? Perhaps spaCy retains legacy code for backwards compatability
        'VERB': [],
        'INTJ': [], # Interjections (wow, amazing, alas)
        'PART': [], # Particle ('s, not)
        'ADV': [],
        'ADJ': [],
        'ADP': [],  # Adposition (in, during)
        'CCONJ': [],    # Coordinating conjunction (and, or, but)
        'SCONJ': [],    # Subordinating conjunction (if, while, that)
    }

    word_count = []
    sent_count = []
    unique_words = []

    cleaned_essays = []

    for id, essay in enumerate(datasubset['essay']):
        e = nlp(essay)

        grammar_errors.append(len(tool.check(essay)))

        misspelt = 0
        words = []
        pos_counter = {}

        for token in e:
            misspelt += 1 if not d.check(token.text) and not token.is_punct else 0
            pos_counter[token.pos_] = pos_counter.get(token.pos_, 0) + 1

            if token.is_punct or token.is_stop or re.fullmatch('@.*', token.text, re.IGNORECASE):
                continue

            words.append(token.text)

        lexical_diversity.append(len(words) / len(set(words)))
        correct_spellings.append(len(words) - misspelt)
        spelling_errors.append(misspelt)
        
        for k in pos_frequency_by_essay.keys():
            pos_frequency_by_essay[k].append(pos_counter.get(k, 0))

        word_count.append(len(words))
        sent_count.append(len([_ for _ in e.sents]))
        unique_words.append(len(set(words)))

        cleaned_essays.append(' '.join(words))

        if debug_text:
            print(
                f'''
                --EXTRACTED FEATURES--\n
                Parts of speech: {pos_counter}\n
                Spelling errors: {spelling_errors[-1]}\n
                Lexical diversity: {lexical_diversity[-1]}\n
                Correct spellings: {correct_spellings[-1]}\n
                Word count: {len(words)}\n
                Sentence count: {sent_count[-1]}\n
                Unique words: {unique_words[-1]}\n
                Grammar Errors: {grammar_errors[-1]}\n
                '''
            )
        
        # Prints a progress bar
        # Courtesy of https://stackoverflow.com/questions/46141302/how-to-make-a-still-progress-in-python/46141777#46141777
        set_size = len(datasubset['essay'])
        chars = int(PROGRESS_BAR_LENGTH * ((id + 1) / set_size))
        print(
            f'Token-based feature extraction progress: |{"█" * chars}{"-" * (PROGRESS_BAR_LENGTH - chars)}| {str(chars * (100 / PROGRESS_BAR_LENGTH))}% ({id + 1} of {set_size})', 
            end='\r'
        )
    print() # Bring cursor to next line to prevent further output from overwriting the progress bar

    for pos, frequency in pos_frequency_by_essay.items():
        datasubset[pos] = frequency

    datasubset['spelling_errors'] = spelling_errors
    datasubset['correct_spellings'] = correct_spellings
    datasubset['lexical_diversity'] = lexical_diversity
    datasubset['grammar_errors'] = grammar_errors
    
    def readability(e):
        try:
            return Readability(e).dale_chall().score
        except ReadabilityException:
            return 0
    datasubset['readability'] = datasubset['essay'].apply(readability)

    datasubset['word_count'] = word_count
    datasubset['sent_count'] = sent_count
    datasubset['unique_words'] = unique_words

    datasubset['cleaned_essay'] = cleaned_essays
    datasubset['domain1_score'] = datasubset['domain1_score'].apply(
        lambda s : round(s * NORMALIZED_SCALE / MAX_ACHIEVABLE_SCORE[essay_set])
    )

    vd = SentimentIntensityAnalyzer()
    def flatten(stb, svd):
        return ','.join(map(str, [stb.polarity, stb.subjectivity, svd['pos'], svd['neg'], svd['neu'], svd['compound']]))
    sentiment_columns = ['polarity (TB)', 'subjectivity (TB)', 'positive (VD)', 'negative (VD)', 'neutral (VD)', 'compound (VD)']
    datasubset[sentiment_columns] = datasubset['essay'].apply(
        lambda e : flatten(TextBlob(e).sentiment, vd.polarity_scores(e))
    ).str.split(',', expand=True)

    prompt = nlp(ESSAY_PROMPTS[essay_set])
    datasubset['prompt'] = ESSAY_PROMPTS[essay_set]
    cleaned_prompt = ' '.join([token.lemma_ for token in prompt if not (token.is_stop or token.is_punct or token.text == '\n')])

    v1 = TfidfVectorizer()
    v2 = CountVectorizer()

    vec_prompt = v1.fit_transform([cleaned_prompt])
    vec_essay = v1.transform(cleaned_essays) 
    datasubset['relevance (TF-IDF)'] = cosine_similarity(vec_essay, vec_prompt).flatten()
    
    temp_df = pd.DataFrame(vec_essay.toarray(), columns=v1.get_feature_names())
    temp_df = pd.DataFrame(vec_prompt.toarray(), columns=v1.get_feature_names())
    temp_df.to_csv('data/vector-essays.tsv', sep='\t')
    temp_df.to_csv('data/vector-prompt.tsv', sep='\t')

    vec_prompt = v2.fit_transform([cleaned_prompt])
    vec_essay = v2.transform(cleaned_essays) 
    datasubset['relevance (CV)'] = cosine_similarity(vec_essay, vec_prompt).flatten()

    return datasubset

# result = process(essay_set=0, debug_text=False)
# result.to_csv('data/processed_data_TEMP.tsv', sep='\t')
# print(result)

# print(data.iloc[1785])
# print(result.iloc[2])

# selected_sets = [1, 2, 3, 4, 5, 6, 7, 8]
# for set_id in selected_sets:
#     print(f'Processing Set {set_id}...')
#     result = process(essay_set=set_id, debug_text=False)
#     result.to_csv(f'data/processed_data_{set_id}.tsv', sep='\t')
#     print(f'Finished and saved Set {set_id}')
    
result = process(custom_input=True, debug_text=True)
result.to_csv('data/processed_data_TEMP.tsv', sep='\t')
print(result)
