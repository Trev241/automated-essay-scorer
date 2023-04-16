import numpy as np
import pandas as pd
import pickle
import lzma

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

from src.feature_extraction import FeatureExtractor, CUSTOM_INPUT

np.random.seed(123)

FEATURES = (
    'ADJ', 'ADV', 'ADP', 'NOUN', 'VERB', 'INTJ', 'PART', 'SCONJ', 'CCONJ', 'word_count', 'sent_count', 'polarity (TB)', 
    'subjectivity (TB)', 'positive (VD)', 'negative (VD)', 'neutral (VD)', 'compound (VD)', 'spelling_errors', 'correct_spellings', 
    'lexical_diversity', 'unique_words', 'relevance (TF-IDF)', 'readability', 'grammar_errors'
)

with open('data/feature_descriptions.txt', 'r') as f:
    FEATURE_DESCRIPTIONS = {}
    line = f.readline()
    
    while line:
        key, value = line.split(': ')
        FEATURE_DESCRIPTIONS[key] = value
        line = f.readline()

class Autograder:
    PROCESSED_DIR = 'data/processed'
    MODEL_DIR = 'data/models'

    def __init__(self):
        print('Initializing Autograder instance...')
        self.extractor = FeatureExtractor(
            path_to_essays='data/training_set_rel3.tsv', 
            path_to_custom='data/custom_input.tsv'
        )

        for essay_set in range(1, 9):
            # Checking if processed essays exist...
            try:
                processed_set = pd.read_csv(f'{Autograder.PROCESSED_DIR}/processed_data_{essay_set}.tsv', sep='\t', compression='gzip')
            except Exception as e:
                print(f'Failed to load processed set {essay_set}. Processing now... (This might take some time): {e}')
                processed_set = self.extractor.process(essay_set=essay_set)
                processed_set.to_csv(f'{Autograder.PROCESSED_DIR}/processed_data_{essay_set}.tsv', sep='\t', compression='gzip')

            # Checking if saved models exist...
            if not Path(f'{Autograder.MODEL_DIR}/model_{essay_set}.sav').is_file():
                print(f'Could not find model for set {essay_set}. Generating model now... (This might take some time)')
                X = pd.DataFrame(processed_set, columns=FEATURES)
                y = processed_set['domain1_score']

                X_train, _, y_train, _ = train_test_split(
                    X,
                    y,
                    test_size=0.15,
                    random_state=42,
                    shuffle=True,
                )

                model = Pipeline(steps=[
                    ('random_forest', RandomForestClassifier(n_estimators=1000, criterion='entropy'))
                ])
                model.fit(X_train, y_train)
                with lzma.open(f'{Autograder.MODEL_DIR}/model_{essay_set}.sav', 'wb') as f:
                    pickle.dump(model, f)
        
        print('Successfully loaded all processed essays and models')

    def grade(self, essay: pd.DataFrame):
        # Process custom input
        essay = self.extractor.process(essay_set=CUSTOM_INPUT)
        essay_set = essay['parent_set'][0]
        X = pd.DataFrame(essay, columns=FEATURES)
        with lzma.open(f'{Autograder.MODEL_DIR}/model_{essay_set}.sav', 'rb') as f:
            model = pickle.load(f)

        report = {
            'prediction': { 
                'description': 'The predicted value of an essay',
                'value': model.predict(X)[0],
            }
        }

        for feature in FEATURES:
            report[feature] = {
                'description': FEATURE_DESCRIPTIONS[feature],
                'value': essay[feature][0]
            }

        return report
