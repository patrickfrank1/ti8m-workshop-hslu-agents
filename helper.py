import ipywidgets as widgets
from IPython.display import display

def display_input_parameters():
    api_key_widget = widgets.Textarea(
        value='',
        description="OpenAI API Key:",
        placeholder="sk-...",
        layout=widgets.Layout(width="500px", height="100px")
    )
    example_widget = widgets.Dropdown(
        options=["sales-customer", "teacher-student"],
        value="teacher-student",
        description="Beispiel:"
    )
    conversation_length_widget = widgets.IntSlider(
        value=3,
        min=1,
        max=10,
        step=1,
        description="Länge der Unterhaltung:"
    )
    display(api_key_widget, example_widget, conversation_length_widget)
    return api_key_widget, example_widget, conversation_length_widget
