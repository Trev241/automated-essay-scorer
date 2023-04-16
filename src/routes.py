import pandas as pd
import git

from src import app, autograder
from src.forms import EssayForm, PromptSelectForm

from flask import render_template, url_for, redirect
from flask import request
from flaskext.markdown import Markdown

Markdown(app)

# --PROMPTS--
ESSAY_PROMPTS = []
for i in range(8):
    with open(f'data/prompts/prompt{i + 1}.md', 'r', encoding='utf-8') as f:
        ESSAY_PROMPTS.append(''.join(f.readlines()))

selected = 0
report = {
    'prediction': {
        'value': 'N/A',
        'description': 'The predicted score of your essay'
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    essay_form = EssayForm()
    prompt_form = PromptSelectForm()

    global selected, report

    if request.method == 'POST':
        if 'submit' in request.form:
            df = pd.DataFrame({
                'essay_id': [1],
                'parent_set': [selected + 1],
                'essay': [request.form['essay']],
                'domain1_score': [0]
            })
            df.set_index('essay_id', inplace=True)
            df.to_csv('data/custom_input.tsv', sep='\t')

            report = autograder.grade(df)

            return redirect(url_for('result'))
        else:
            for i in range(8):
                if f'prompt{i + 1}' in request.form:
                    selected = i

    return render_template(
        'index.html', 
        essay_form=essay_form, 
        prompt_form=prompt_form, 
        prompt=ESSAY_PROMPTS[selected]
    )

@app.route('/result', methods=['GET'])
def result():
    global report

    return render_template(
        'result.html', 
        report=report
    )

@app.route('/update', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('aes')
        origin = repo.remotes.origin

        origin.pull('main')
        return 'Updated PythonAnywhere successfully', 200
    else:
        return 'Wrong event type', 400