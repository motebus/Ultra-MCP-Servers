import re 
from openai import OpenAI
import json

def call_python_model(prompt):
    #Test Python JSONL
    from openai import OpenAI

    client = OpenAI()

    prompt += " Here is an example of a Echo function:"
    prompt += ''' # from langflow.field_typing import Data\nfrom langflow.custom import Component\nfrom langflow.io import MessageTextInput, Output\nfrom langflow.schema import Data\n\n\nclass CustomComponent(Component):\n    display_name = \"Custom Component\"\n    description = \"Use as a template to create your own component.\"\n    documentation: str = \"http://docs.langflow.org/components/custom\"\n    icon = \"code\"\n    name = \"CustomComponent\"\n\n    inputs = [\n        MessageTextInput(\n            name=\"input_value\",\n            display_name=\"Input Value\",\n            info=\"This is a custom component Input\",\n            value=\"Hello, World!\",\n            tool_mode=True,\n        ),\n    ]\n\n    outputs = [\n        Output(display_name=\"Output\", name=\"output\", method=\"build_output\"),\n    ]\n\n    def build_output(self) -> Data:\n        data = Data(value=self.input_value)\n        self.status = data\n        return data\n",   '''            
    
    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::B2BEJt6D",
        messages=[{"role": "user", "content": prompt}]
    )

    #print(response.choices[0].message.content)

    return response.choices[0].message.content

def convert_python_one_line(python_code):
    jsonl_line = json.dumps(python_code, ensure_ascii=False)
    return jsonl_line

def parse_python_code(python_code):
    match = re.search(r"```python\n(.*?)\n```", python_code, re.DOTALL)
    return match.group(1) if match else None

"""
def find_value(json_response, python_code_one_line):

    # Find the start of the "value" field
    value_start = json_response.find('"value":"') + len('"value":"')
    
    # Find the end of the value (next quote after a backwards slash)
    value_end = json_response.find('",', value_start)
    # Create the new JSON string by replacing the old value
    new_json = json_response[:value_start] + python_code_one_line + json_response[value_end:]
    save_json_safely(new_json)

    return new_json
"""

if __name__ == "__main__":
    #Generate code from user description

    """Sample user input
    Generate a LangFlow custom component in python code that multiplies 2 numbers. 
    The component should have 2 Message inputs and 1 Message output.
    """

    #user_input = input("Please give detailed description of Langflow custom component:")
    user_input = "Generate a LangFlow custom component in python code that multiplies 2 numbers. The component should have 2 Message inputs and 1 Message output."
    python_code = call_python_model(user_input)
    python_code_one_line = convert_python_one_line(parse_python_code(python_code))
    print(python_code_one_line)