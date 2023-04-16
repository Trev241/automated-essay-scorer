from flask_wtf import FlaskForm
from wtforms.fields import TextAreaField, SubmitField
from wtforms.validators import DataRequired

class EssayForm(FlaskForm):
    essay = TextAreaField(
        'Essay', 
        validators=[DataRequired()], 
        render_kw={'placeholder': 'Type in your essay here'}
    )

    submit = SubmitField('Submit')

class PromptSelectForm(FlaskForm):
    pass

for i in range(1, 9):
    setattr(PromptSelectForm, f'prompt{i}', SubmitField(i))
