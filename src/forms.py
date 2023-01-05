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
    prompt1 = SubmitField('1')
    prompt2 = SubmitField('2')
    prompt3 = SubmitField('3')
    prompt4 = SubmitField('4')
    prompt5 = SubmitField('5')
    prompt6 = SubmitField('6')
    prompt7 = SubmitField('7')
    prompt8 = SubmitField('8')
