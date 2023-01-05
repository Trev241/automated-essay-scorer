import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.feature_extraction import FeatureExtractor, CUSTOM_INPUT

np.random.seed(123)

FEATURES = (
    'ADJ', 'ADV', 'ADP', 'NOUN', 'VERB', 'INTJ', 'PART', 'SCONJ', 'CCONJ', 'word_count', 'sent_count', 'polarity (TB)', 
    'subjectivity (TB)', 'positive (VD)', 'negative (VD)', 'neutral (VD)', 'compound (VD)', 'spelling_errors', 'correct_spellings', 
    'lexical_diversity', 'unique_words', 'relevance (TF-IDF)', 'readability', 'grammar_errors'
)

class Autograder:
    def __init__(self):
        self.extractor = FeatureExtractor(
            path_to_essays='src/data/training_set_rel3.tsv', 
            path_to_custom='src/data/custom_input.tsv'
        )

    def generate_models(self, sets=[]):
        for essay_set in sets:
            try:
                processed_set = pd.read_csv(f'src/data/processed/processed_data_{essay_set}.tsv')
            except:
                print('Failed to load processed set {essay_set}. Processing essays now... (This might take some time)')
                processed_set = self.extractor.process(essay_set=essay_set)
                processed_set.to_csv(f'src/data/processed_data_{essay_set}.tsv', sep='\t')

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
            pickle.dump(model, open(f'src/data/models/model_{essay_set}.sav', 'wb'))

    def grade(self, essay: pd.DataFrame):
        # Process custom input
        essay = self.extractor.process(essay_set=CUSTOM_INPUT)
        essay_set = essay['parent_set'][0]
        X = pd.DataFrame(essay, columns=FEATURES)

        try:
            model = pickle.load(open(f'src/data/models/model_{essay_set}.sav', 'rb'))
            return model.predict(X)[0]
        except Exception as e:
            print(f'Failed to load model for set {essay_set}: {e}')
    
