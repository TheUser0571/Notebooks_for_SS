from ipywidgets import HBox, Layout, VBox
import ipywidgets as widgets
import IPython
from IPython.display import HTML

class FloatCheck():
    def __init__(self, minval, maxval, description='Value:', width='100px'):        
        self.text = widgets.FloatText(value=0, description=description, layout=Layout(width=width), style = {'description_width': 'initial'})
        self.text.observe(self.check_answer, names='value')
        self.min = minval
        self.max = maxval
        self.incorrect_html = '<p style="color:Red;">Incorrect</p>'
        self.correct_html = '<p style="color:Green;">Correct</p>'
        self.html = widgets.HTML('', layout=Layout(width='75px'))
        self.display = HBox([self.text, self.html])
    
    def check_answer(self, value):
        if value['new'] >= self.min and value['new'] <= self.max:
            self.html.value = self.correct_html
        else:
            self.html.value = self.incorrect_html
            
class MCQ():
    def __init__(self, options, correct, description='', value=None):
        self.correct = correct
        self.radiobuttons = widgets.RadioButtons(options=options, description=description, value=value, layout={'width': 'max-content'}, style = {'description_width': 'initial'})        
        self.submit_button = widgets.Button(description='Submit', button_style='', layout={'width': 'max-content'})
        self.submit_button.on_click(self.check_answer)
        self.display = VBox([self.radiobuttons, self.submit_button])
    
    def check_answer(self, value):
        if self.radiobuttons.value == self.correct:
            self.submit_button.button_style = 'success'
        else:
            self.submit_button.button_style = 'danger'