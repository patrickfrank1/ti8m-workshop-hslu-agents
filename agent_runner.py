from io import BytesIO

from PIL import Image as pil_Image
from dotenv import load_dotenv
from IPython.display import Markdown, Image, display
from openai import OpenAI
from openai.types.beta.threads.text_content_block import TextContentBlock
from openai.types.beta.threads.image_file_content_block import ImageFileContentBlock
import matplotlib.pyplot as plt

class AgentRunner:
    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            load_dotenv()
            self.client = OpenAI()
        self.teacher_assistant = self.client.beta.assistants.create(
            name="Math Tutor",
            instructions="You are a personal math tutor. You can optionally write and run code to answer math questions. Use MathJax syntax to display equations.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o",
        )
        self.student_assistant = self.client.beta.assistants.create(
            name="Math Student",
            instructions="You are a curious student studying for a math exam in trigonometry. Ask your questions to the tutor, ask for clarifications if necessary.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o",
        )
        self.thread = None
        self.message_counter = None
        self.is_student = None
        return
    
    def __del__(self) -> None:
        print(self.client.beta.assistants.delete(assistant_id=self.student_assistant.id))
        print(self.client.beta.assistants.delete(assistant_id=self.teacher_assistant.id))
        return
    
    def _print_latest_message_batch(self, thread_id: str) -> None:
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id
        )
        print("Student:" if self.is_student else "Teacher:")
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
        self.is_student = not self.is_student    
        return
    
    def start_conversation(self, student_instruction: str, conversation_length: int = 2) -> None:
        """
            Provide a conversation topic and let the agents have a conversation about it.
        """
        student_initial_instruction = f"{student_instruction} Ask a question to the tutor."
        student_instruction = "Ask a clarifying question or a follow up question."
        teacher_instruction = "Answer the students question."
        self.is_student = True
        self.message_counter = 0
        self.thread = self.client.beta.threads.create()
        for i in range(conversation_length):
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.student_assistant.id,
                instructions=student_initial_instruction if i == 0 else student_instruction
            )
            self._print_latest_message_batch(self.thread.id)
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.teacher_assistant.id,
                instructions=teacher_instruction
            )
            self._print_latest_message_batch(self.thread.id)
        return