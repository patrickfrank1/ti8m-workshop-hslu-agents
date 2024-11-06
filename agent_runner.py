from io import BytesIO
from typing import Tuple

from PIL import Image as pil_Image
from dotenv import load_dotenv
from IPython.display import Markdown, Image, display
from openai import OpenAI
from openai.types.beta.threads.text_content_block import TextContentBlock
from openai.types.beta.threads.image_file_content_block import ImageFileContentBlock
from openai.types.beta.assistant import Assistant
import matplotlib.pyplot as plt

class AgentRunner:
    def __init__(self, api_key: str | None = None, example: str = "teacher-student") -> None:
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            load_dotenv()
            self.client = OpenAI()
        self.assistant, self.tester = self._get_agents(example)
        self.thread = None
        self.message_counter = None
        self.is_tester = None
        return
    
    def __del__(self) -> None:
        print(self.client.beta.assistants.delete(assistant_id=self.tester.id))
        print(self.client.beta.assistants.delete(assistant_id=self.assistant.id))
        return

    def start_conversation(self, tester_instruction: str, conversation_length: int = 2) -> None:
        """
            Provide a conversation topic and let the agents have a conversation about it.
        """
        tester_initial_instruction = f"{tester_instruction} Ask a question to the assistant."
        tester_instruction = "Ask a clarifying question or a follow up question to the tutor that relates to the contents of the conversation."
        assistant_instruction = "Answer the testers question."
        self.is_tester = True
        self.message_counter = 0
        self.thread = self.client.beta.threads.create()
        for i in range(conversation_length):
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.tester.id,
                instructions=tester_initial_instruction if i == 0 else tester_instruction
            )
            self._print_latest_message_batch(self.thread.id)
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=assistant_instruction
            )
            self._print_latest_message_batch(self.thread.id)
        return

    def _get_agents(self, example: str) -> Tuple[Assistant, Assistant]:
        if example == "teacher-student":
            assistant = self.client.beta.assistants.create(
                name="Math Tutor",
                instructions="You are a personal math tutor. You can optionally write and run code to answer math questions. Use MathJax syntax to display equations.",
                tools=[{"type": "code_interpreter"}],
                model="gpt-4o"
            )
            tester = self.client.beta.assistants.create(
                name="Math Student",
                instructions="You are a curious student studying for a math exam in trigonometry. Ask your questions to the tutor, ask for clarifications if necessary.",
                tools=[{"type": "code_interpreter"}],
                model="gpt-4o",
            )
        elif example == "sales-customer":
            assistant = self.client.beta.assistants.create(
                name="Versicherungsverkäufer",
                instructions=(
                    "- Du sprichst nur Deutsch.\n"
                    "- Du spricht auch Schweizerdeutsch und berüsst die Kunden oft mit verwendest Grüezi uns Sali\n"
                    "- Du erfindest nie etwas und lügst nie. Falls du unsicher bist, sagst du das ganz explizit und bietest an mehr Sicherheit zu schaffen indem du weitere Nachforschungen anstellst.\n"
                    "- Du verwendest einen Verkäufer Slang, bist eine sympathische 60 jährige Frau und stellst häufig Rückfragen und machst Smalltalk.\n"
                    "- Du sprichst nur über eigene Produkte und deinen eigenen Arbeitgeber.\n"
                    "- Du machst gerne Ferien in Malaysia, und an Nord- und Ostsee.\n"
                    "- Du bis verheiratet und hast 2 Kinder. Einen Jungen und ein Mädchen. 8 und 13 Jahre alt.\n"
                    "- Du heisst Sabine und wohnst in Zürich und arbeitest seit 4 Jahren  für die Helsana als Verkäufer.\n"
                    "- Du kennst dich perfekt mit dem Schweizer Gesundheitswesen aus.\n"
                    "- Du magst warme Temperaturen und befürchtest den Klimawandel.\n"
                    "- Du bis Vegetarier und treibst viel Sport. Primär Jogging.\n"
                    "- Du kennst die Website https://www.helsana.ch/de/private.html perfekt und bist Produktexperte.\n"
                    "- Du sendest die Informationen deines Gesprächspartners an die Auftragsbearbeitung deines Arbeitgebers. Dazu brauchst du die Personalien (Vorname, Nachname, eMail) deines Gesprächspartners. Du musst danach fragen und dann die Daten in die eMail einsetzen.\n"
                    "- Du sendest nie eine Mail an den Kunden.\n"
                    "- Du sendest Mail immer in einem definierten Protokoll an die deine Auftragsbearbeitung. Du nennst diese Abteilung 'unsere Auftragsverarbeitung'.\n"
                    "- In die Mail generierst du immer die eMail-Adresse des Kunden.\n"
                    "- Für alle Versicherungen musst du das Alter und den aktuellen Wohnort wissen und in die Mail generieren.\n"
                    "- Wenn du eine Mail schreibst, dann fragst auf jeden fall: 'Soll ich wirklich schicken?'. Bei Ja sagst du 'Klasse ich schicke die Mail: ' und gibst die komplette mail dahinter an. Du verwendest immer den Text 'Klasse ich schicke die Mail:' gefolgt von der Mail, damit ich diesen Text parsen kann.\n"
                    "- Du bist immer extrem gesprächig und fängst immer einen Smalltalk an z.B. über Ferien\n"
                ),
                tools=[],
                model="gpt-4o",
                #extra_headers={"OpenAI-Beta": "assistants=v1"}
            )
            tester = self.client.beta.assistants.create(
                name="Testkunde",
                instructions=(
                    "- Du sprichst nur Deutsch.\n"
                    "- Du bist ein Testkunde der als Kritiker agiert.\n"
                    "- Du fragst nicht ob es noch weitere Fragen gibt. Du bist der, der die Fragen stellt.\n"
                    "- Du stellst nur Prüfungsfragen und bewertest die Antworten immer von 0-10.\n"
                    "- Du erfindest nie etwas und lügst nie. Falls du unsicher bist, sagst du das ganz explizit und bietest an mehr Sicherheit zu schaffen indem du weitere Nachforschungen anstellst.\n"
                    "- Du verwendest einen formalen professionellen Slang.\n"
                    "- Du kennst dich perfekt mit dem Schweizer Gesundheitswesen aus.\n"
                    "- Du bewertest jede Eingabe mit einer skala von 0-10. 10 gut, 0 schlecht\n"
                ),
                tools=[],
                model="gpt-4o",
                #temperature=2.0,
                #extra_headers={"OpenAI-Beta": "assistants=v1"}
            )
        else:
            raise ValueError("The requested example scenario does not exist. Please choose example='teacher-student' or example='sales-customer'.")
        return assistant, tester
    
    def _print_latest_message_batch(self, thread_id: str) -> None:
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id
        )
        print(self.tester.name if self.is_tester else self.assistant.name)
        while self.message_counter < len(messages.data):
            item = len(messages.data) - 1 - self.message_counter
            contents = messages.data[item].content
            for j in range(len(contents)):
                if isinstance(contents[j], ImageFileContentBlock):
                    response = self.client.files.with_raw_response.content(contents[j].image_file.file_id)
                    image_binary_string = response.content
                    image_stream = BytesIO(image_binary_string)
                    img = pil_Image.open(image_stream)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
                if isinstance(contents[j], TextContentBlock):
                    latest_message = contents[j].text.value
                    display(Markdown(latest_message))
            self.message_counter += 1
        self.is_tester = not self.is_tester    
        return
