"""A simple chat client with conversation history.

>>> from unittest.mock import MagicMock, patch
>>> response = MagicMock()
>>> response.choices[0].message.content = "Hello!"
>>> with patch.object(client.chat.completions, "create", return_value=response):
...     chat = Chat(system="Be brief.")
...     answer = chat.send_message("Hi", temperature=0)
>>> answer
'Hello!'
>>> chat.messages
[{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello!'}]

Test that earlier assistant messages remain in the next API request.

>>> first = MagicMock()
>>> first.choices[0].message.content = "Ahoy!"
>>> second = MagicMock()
>>> second.choices[0].message.content = "Welcome back!"
>>> captured = []
>>> responses = iter([first, second])
>>> def fake_create(*, messages, model, temperature):
...     captured.append([dict(message) for message in messages])
...     return next(responses)
>>> with patch.object(client.chat.completions, "create", side_effect=fake_create):
...     chat = Chat(system="Be brief.")
...     _ = chat.send_message("Remember this", temperature=0)
...     _ = chat.send_message("What did I say?", temperature=0)
>>> captured[1]
[{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Remember this'}, {'role': 'assistant', 'content': 'Ahoy!'}, {'role': 'user', 'content': 'What did I say?'}]

Test the REPL: replies print, conversation history persists, and EOF exits.

>>> first = MagicMock()
>>> first.choices[0].message.content = "Ahoy!"
>>> second = MagicMock()
>>> second.choices[0].message.content = "Welcome back!"
>>> inputs = MagicMock(side_effect=["Hello", "Who are you?", EOFError])
>>> captured = []
>>> responses = iter([first, second])
>>> def fake_create(*, messages, model, temperature):
...     captured.append([dict(message) for message in messages])
...     return next(responses)
>>> with patch("builtins.input", inputs), patch.object(client.chat.completions, "create", side_effect=fake_create), patch("builtins.print") as output:
...     repl(temperature=0)
>>> [call.args[0] for call in output.call_args_list if call.args]
['Ahoy!', 'Welcome back!']
>>> captured[1]
[{'role': 'system', 'content': 'You are a friendly pirate. Reply like a pirate.'}, {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Ahoy!'}, {'role': 'user', 'content': 'Who are you?'}]

KeyboardInterrupt should also exit cleanly.

>>> inputs = MagicMock(side_effect=["Ahoy", KeyboardInterrupt])
>>> reply = MagicMock()
>>> reply.choices[0].message.content = "Arr!"
>>> with patch("builtins.input", inputs), patch.object(client.chat.completions, "create", return_value=reply), patch("builtins.print"):
...     repl(temperature=0)
>>> inputs.call_count
2
"""

import os
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def llm(messages, temperature=1):
    """Send messages to the language model and return its reply."""
    completion = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-20b",
        temperature=temperature,
    )
    return completion.choices[0].message.content


class Chat:
    """Store and update a conversation."""

    def __init__(self, system):
        self.messages = [{"role": "system", "content": system}]

    def send_message(self, message, temperature=1):
        """Send a message and save both sides of the conversation."""
        self.messages.append({"role": "user", "content": message})
        reply = llm(self.messages, temperature=temperature)
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def repl(temperature=0.8):
    """Run a simple interactive chat loop."""
    chat = Chat(system="You are a friendly pirate. Reply like a pirate.")
    try:
        while True:
            message = input("You: ")
            reply = chat.send_message(message, temperature=temperature)
            print(reply)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    repl()
