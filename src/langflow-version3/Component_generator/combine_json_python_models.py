import json
from openai import OpenAI
import re

false = False  
true = True   

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

def get_last_sentence(text):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text.strip())
    return sentences[-1] if sentences else None

def parse_python_code(python_code):
    match = re.search(r"```python\n(.*?)\n```", python_code, re.DOTALL)
    return match.group(1) if match else None

#Prepare python data to JSONL
def convert_python_one_line(python_code):
    jsonl_line = json.dumps(python_code, ensure_ascii=False)
    #print(jsonl_line)
    return jsonl_line

def call_json_model(json_data,input_output_data):

    client = OpenAI()
    prompt = '''Generate a LangFlow component JSON for '''
    prompt += f"{json_data}."
    prompt += input_output_data
    prompt += ''' Leave 'value' field empty.'''

    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::B2YQNexS",
        messages=[{"role": "user", "content": prompt}]
    )
    save_json_safely(response.choices[0].message.content)
    return response.choices[0].message.content

def save_json_safely(content):
    try:
        # If the content is already a valid JSON object, save it directly
        if isinstance(content, dict):
            json_obj = content
        else:
            # Parse the content as JSON
            json_obj = json.loads(content)
        
        # Save with proper formatting
        with open('test.json', 'w', encoding='utf-8') as f:
            # Use ensure_ascii=False to properly handle unicode characters
            # Use separators to remove extra whitespace
            json.dump(json_obj, f, ensure_ascii=False, separators=(',', ':'))
            
        # Verify the saved file can be read back
        with open('test.json', 'r', encoding='utf-8') as f:
            verification = json.load(f)
            
        return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    #Generate code from user description

    """Sample user input
    Generate a LangFlow custom component in python code that multiplies 2 numbers. 
    The component should have 2 Message inputs and 1 Message output.
    """

    #user_input = input("Please give detailed description of Langflow custom component:")
    user_input = "Generate a LangFlow custom component in python code that Echoes the user input. The component should have 1 Message input and 1 Message output."
    python_code = call_python_model(user_input)
    input_output_data = get_last_sentence(user_input)
    #print(python_code)
    
    #Find python code from generated responce, convert python into 1 line jsonl string
    python_code_one_line = convert_python_one_line(parse_python_code(python_code))
    #print(python_code_one_line)

    #Give json generator prompt to json generator
    json_responce = call_json_model(python_code_one_line,input_output_data)
