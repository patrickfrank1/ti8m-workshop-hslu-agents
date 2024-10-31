import os
from dotenv import load_dotenv
from IPython.display import Markdown, display

from openai import OpenAI


class AgentRunner:
    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            load_dotenv()
            self.client = OpenAI()
        self.teacher_assistant = self.client.beta.assistants.create(
            name="Math Tutor",
            instructions="You are a personal math tutor. You can optionally write and run code to answer math questions.",
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
        return
    
    def __del__(self) -> None:
        print(self.client.beta.assistants.delete(assistant_id=self.student_assistant.id))
        print(self.client.beta.assistants.delete(assistant_id=self.teacher_assistant.id))
        return
    
    def _print_latest_message(self, thread_id: str) -> None:
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id
        )
        latest_messsage = messages.data[0].content[0].text.value
        if len(messages.data) % 2 == 0:
            print("Teacher:\n")
        else:
            print("Teacher:\n")
        display(Markdown(latest_messsage))
        return
    
    def start_conversation(self, student_instruction: str, conversation_length: int = 2) -> None:
        """
            Provide a conversation topic and let the agents have a conversation about it.
        """
        student_initial_instruction = f"{student_instruction} Ask a question to the tutor?"
        c = "Ask a clarifying question or a follow up question."
        teacher_instruction = "Answer the students question."
        self.thread = self.client.beta.threads.create()
        for i in range(conversation_length):
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.student_assistant.id,
                instructions=student_initial_instruction if i == 0 else student_instruction
            )
            self._print_latest_message(self.thread.id)
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.teacher_assistant.id,
                instructions=teacher_instruction
            )
            self._print_latest_message(self.thread.id)
        return