import asyncio
import requests
import json
import os
from uuid import uuid4
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
from datetime import datetime
from typing import Optional, Dict, Any
from openai import OpenAI
import re
import logging
from logging.handlers import RotatingFileHandler

# Ensure log directory exists
log_directory = "logs"  # Change this to your preferred directory
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "mcp_server.log")

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a file handler
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=10485760,  # 10MB
    backupCount=5        # Keep 5 backup copies
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Get logger for this module and add the file handler
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

notes: dict[str, str] = {}

server = Server("langflow")

LANGFLOW_API_URL = os.environ.get("LANGFLOW_API_URL", "http://localhost:7860/api/v1/flows/")
openai_api_key = os.environ.get("OPENAI_API_KEY")

def call_python_model(prompt):
    logger.info("=" * 50)
    logger.info("STARTING call_python_model")
    logger.info(f"Input prompt: {prompt[:100]}...")
    
    client = OpenAI(api_key = openai_api_key)

    prompt += " Here is an example of a Echo function:"
    prompt += ''' # from langflow.field_typing import Data\nfrom langflow.custom import Component\nfrom langflow.io import MessageTextInput, Output\nfrom langflow.schema import Data\n\n\nclass CustomComponent(Component):\n    display_name = \"Custom Component\"\n    description = \"Use as a template to create your own component.\"\n    documentation: str = \"http://docs.langflow.org/components/custom\"\n    icon = \"code\"\n    name = \"CustomComponent\"\n\n    inputs = [\n        MessageTextInput(\n            name=\"input_value\",\n            display_name=\"Input Value\",\n            info=\"This is a custom component Input\",\n            value=\"Hello, World!\",\n            tool_mode=True,\n        ),\n    ]\n\n    outputs = [\n        Output(display_name=\"Output\", name=\"output\", method=\"build_output\"),\n    ]\n\n    def build_output(self) -> Data:\n        data = Data(value=self.input_value)\n        self.status = data\n        return data\n",   '''   

    logger.info("Sending request to OpenAI API...")
    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::B2BEJt6D",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.choices[0].message.content
    logger.info(f"Received response from OpenAI API (first 100 chars): {result}")
    logger.info("COMPLETED call_python_model")
    logger.info("=" * 50 + "\n")

    return result

def get_last_sentence(text):
    logger.info("=" * 50)
    logger.info("STARTING get_last_sentence")
    logger.info(f"Input text: {text}")
    
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text.strip())
    result = sentences[-1] if sentences else None
    
    logger.info(f"Extracted last sentence: {result}")
    logger.info("COMPLETED get_last_sentence")
    logger.info("=" * 50 + "\n")
    
    return result

def parse_python_code(python_code):
    logger.info("=" * 50)
    logger.info("STARTING parse_python_code")
    #logger.info(f"Input code (first 100 chars): {python_code[:100]}...")
    
    # Match code between ```python and ``` markers
    match = re.search(r"```python\s*(.*?)\s*```", python_code, re.DOTALL)
    result = match.group(1) if match else None
    
    logger.info(f"Parsed code (first 100 chars): {result}")
    logger.info("COMPLETED parse_python_code")
    logger.info("=" * 50 + "\n")
    
    return result

#Prepare python data to JSONL
def convert_python_one_line(python_code):
    logger.info("=" * 50)
    logger.info("STARTING convert_python_one_line")
    logger.info(f"Input code (first 100 chars): {python_code}")
    
    jsonl_line = json.dumps(python_code, ensure_ascii=False)
    
    logger.info(f"Converted to JSONL (first 100 chars): {jsonl_line}")
    logger.info("COMPLETED convert_python_one_line")
    logger.info("=" * 50 + "\n")
    
    return jsonl_line

def call_json_model(json_data, input_output_data):
    logger.info("=" * 50)
    logger.info("STARTING call_json_model")
    logger.info(f"Input json_data: {json_data[:100] if isinstance(json_data, str) else str(json_data)[:100]}...")
    logger.info(f"Input input_output_data: {input_output_data[:100]}...")
    
    client = OpenAI(api_key = openai_api_key)
    prompt = '''Generate a LangFlow component JSON for the python code that matches:  '''
    prompt += f"{json_data}."
    prompt += input_output_data
    prompt += ''' Leave 'value' field empty.'''
    
    logger.info("Sending request to OpenAI API...")
    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::B2YQNexS",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content
    logger.info(f"Received response from OpenAI API (first 100 chars): {result[:100]}...")
    
    # Extract JSON if it's embedded in markdown code blocks
    if "```json" in result:
        json_start = result.find("```json") + 7
        json_end = result.find("```", json_start)
        result = result[json_start:json_end].strip()
    elif "```" in result:
        # Handle cases where it's just ``` without json specification
        json_start = result.find("```") + 3
        json_end = result.find("```", json_start)
        result = result[json_start:json_end].strip()
    
    try:
        # Parse the JSON to get its structure
        json_obj = json.loads(result)
        
        # Since we know json_data is already a JSON string (from its creation with json.dumps),
        # we first parse it back to a regular string by loading it
        if json_data.startswith('"') and json_data.endswith('"'):
            # This looks like a JSON string, so try to parse it
            try:
                # Parse the JSON string to get the actual Python code
                actual_code = json.loads(json_data)
            except:
                # If parsing fails, use the original string
                actual_code = json_data
        else:
            # If it doesn't look like a JSON string, use it as is
            actual_code = json_data
        
        # Find the first occurrence of the code template with a value field
        if "data" in json_obj and "nodes" in json_obj["data"]:
            for node in json_obj["data"]["nodes"]:
                if "data" in node and "node" in node["data"] and "template" in node["data"]["node"]:
                    template = node["data"]["node"]["template"]
                    if "code" in template and isinstance(template["code"], dict) and "value" in template["code"]:
                        # Set the value to the actual Python code
                        template["code"]["value"] = actual_code
                        logger.info("Successfully modified the first occurrence of 'value'")
                        break
        
        # Convert back to JSON string
        new_result = json.dumps(json_obj, indent=2)
        
        # Validate the result
        json.loads(new_result)
        
        logger.info("BEFORE EXIT CALL JSON FUNCTION")
        logger.info(f"{new_result[:100]}...")
        return new_result
        
    except Exception as e:
        logger.error(f"Error processing JSON: {str(e)}")
        return result  # Return original if processing fails
    
    finally:
        logger.info("COMPLETED call_json_model")
        logger.info("=" * 50 + "\n")

def extract_component_info(component_data: dict) -> tuple[Optional[dict], Optional[str], Optional[dict]]:
    try:
        nodes = component_data.get("data", {}).get("nodes", [])
        if not nodes:
            return None, None, None
            
        node = nodes[0]
        node_data = node.get("data", {})
        
        component_type = node_data.get("type", "")
        if not component_type:
            return None, None, None
            
        return node_data.get("node", {}), component_type, node
        
    except Exception as e:
        print(f"Error extracting component info: {str(e)}")
        return None, None, None
    
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available note resources.
    Each note is exposed as a resource with a custom note:// URI scheme.
    """
    return [
        types.Resource(
            uri=AnyUrl(f"note://internal/{name}"),
            name=f"Note: {name}",
            description=f"A simple note named {name}",
            mimeType="text/plain",
        )
        for name in notes
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific note's content by its URI.
    The note name is extracted from the URI host component.
    """
    if uri.scheme != "note":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    name = uri.path
    if name is not None:
        name = name.lstrip("/")
        return notes[name]
    raise ValueError(f"Note not found: {name}")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="summarize-notes",
            description="Creates a summary of all notes",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    if name != "summarize-notes":
        raise ValueError(f"Unknown prompt: {name}")

    style = (arguments or {}).get("style", "brief")
    detail_prompt = " Give extensive details." if style == "detailed" else ""

    return types.GetPromptResult(
        description="Summarize the current notes",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Here are the current notes to summarize:{detail_prompt}\n\n"
                    + "\n".join(
                        f"- {name}: {content}"
                        for name, content in notes.items()
                    ),
                ),
            )
        ],
    )

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools for flow management.
    """
    return [
        types.Tool(
            name="list-flows",
            description="List available flows",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_name": {"type": "string", "description": "Optional flow name to filter"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="create-flow",
            description="Create a new flow",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the new flow"},
                    "description": {"type": "string", "description": "Description of the flow"},
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="delete-flow",
            description="Delete a specific flow by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "description": "ID of the flow to delete"},
                },
                "required": ["flow_id"],
            },
        ),
        types.Tool(
            name="upload-saved-component",
            description="Upload a saved flow component from JSON file",
            inputSchema={
                "type": "object",
                "properties": {
                    "json_file_path": {"type": "string", "description": "Full path to the JSON flow file"},
                },
                "required": ["json_file_path"],
            },
        ),
        types.Tool(
            name="add-component-to-flow",
            description="Add a component to an existing flow",
            inputSchema={
                "type": "object",
                "properties": {
                    "component_path": {"type": "string", "description": "Full path to the component JSON file"},
                    "flow_id": {"type": "string", "description": "ID of the flow to add the component to"},
                    "x": {"type": "integer", "description": "X coordinate for component placement", "default": 100},
                    "y": {"type": "integer", "description": "Y coordinate for component placement", "default": 100},
                },
                "required": ["component_path", "flow_id"],
            },
        ),
        types.Tool(
            name="generate-component",
            description="Generate a new LangFlow custom component",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Detailed description of the component functionality"},
                    "output_path": {"type": "string", "description": "Path where to save the generated component"},
                },
                "required": ["description", "output_path"],
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests for flow management.
    """
    try:
        base_url = LANGFLOW_API_URL
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        if name == "list-flows":
            url = base_url
            filter_name = arguments.get("filter_name") if arguments else None
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            flows = response.json()

            if filter_name:
                flows = [flow for flow in flows if flow['name'] == filter_name]

            flow_info = []
            for flow in flows:
                flow_info.append(f"ID: {flow['id']}, Name: {flow['name']}")

            return [
                types.TextContent(
                    type="text",
                    text="\n".join(flow_info) if flow_info else "No flows found."
                )
            ]

        elif name == "create-flow":
            if not arguments or not arguments.get("name"):
                raise ValueError("Flow name is required")

            payload = {
                "name": arguments["name"],
                "description": arguments.get("description", ""),
                "data": {
                    "nodes": [],
                    "edges": []
                }
            }

            response = requests.post(base_url, 
                                     headers=headers, 
                                     data=json.dumps(payload))
            response.raise_for_status()

            return [
                types.TextContent(
                    type="text",
                    text=f"Flow created successfully: {response.text}"
                )
            ]

        elif name == "delete-flow":
            if not arguments or not arguments.get("flow_id"):
                raise ValueError("Flow ID is required")

            url = f"{base_url}{arguments['flow_id']}"
            response = requests.delete(url, headers=headers)
            response.raise_for_status()

            return [
                types.TextContent(
                    type="text",
                    text=f"Flow deleted successfully: {response.text}"
                )
            ]

        elif name == "upload-saved-component":
            if not arguments or not arguments.get("json_file_path"):
                raise ValueError("JSON file path is required")

            json_file_path = arguments["json_file_path"]
            
            try:
                with open(json_file_path, 'r') as file:
                    flow_data = json.load(file)
                
                response = requests.post(
                    base_url,
                    json=flow_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                response.raise_for_status()
                result = response.json()

                return [
                    types.TextContent(
                        type="text",
                        text=f"Flow uploaded successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" +
                             "\n".join(f"{key}: {value}" for key, value in result.items())
                    )
                ]
            
            except FileNotFoundError:
                raise ValueError(f"The file {json_file_path} was not found.")
            except json.JSONDecodeError:
                raise ValueError(f"The file {json_file_path} is not a valid JSON file.")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Error making the request to Langflow API: {str(e)}")

        elif name == "add-component-to-flow":
            if not arguments or not arguments.get("component_path") or not arguments.get("flow_id"):
                raise ValueError("Component path and flow ID are required")

            component_path = arguments["component_path"]
            flow_id = arguments["flow_id"]
            position = {
                "x": arguments.get("x", 100),
                "y": arguments.get("y", 100)
            }

            # First, get the existing flow
            flow_endpoint = f"{base_url.rstrip('/')}/{flow_id}"
            response = requests.get(flow_endpoint)
            response.raise_for_status()
            flow_data = response.json()
            
            # Read the component JSON
            with open(component_path, 'r') as file:
                component_data = json.load(file)
            
            # Extract component info
            component_node, component_type, node_template = extract_component_info(component_data)
            if not component_node or not component_type or not node_template:
                raise ValueError("Could not extract component information")
            
            # Create node in the format expected by Langflow
            new_id = f"{component_type}-{str(uuid4())[:6]}"
            
            # Start with the template from the component
            new_node = {
                "id": new_id,
                "type": "genericNode",
                "position": position,
                "data": {
                    "node": component_node,
                    "id": new_id,
                    "type": component_type
                }
            }
            
            # Copy additional fields from the template
            for field in ["selected", "width", "height", "dragging", "positionAbsolute"]:
                if field in node_template:
                    new_node[field] = node_template[field]
                    
            # Copy additional data fields from the template
            for field in ["value", "showNode", "display_name", "description"]:
                if field in node_template.get("data", {}):
                    new_node["data"][field] = node_template["data"][field]
            
            # Add the component to the flow's data
            if "data" in flow_data and "nodes" in flow_data["data"]:
                flow_data["data"]["nodes"].append(new_node)
            else:
                raise ValueError("Invalid flow data structure")
            
            # Update the flow with the new component
            update_endpoint = f"{base_url.rstrip('/')}/{flow_id}"
            update_response = requests.patch(
                update_endpoint,
                json=flow_data,
                headers={'Content-Type': 'application/json'}
            )
            
            update_response.raise_for_status()
            result = update_response.json()

            return [
                types.TextContent(
                    type="text",
                    text=f"Component added successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" +
                         "\n".join(f"{key}: {value}" for key, value in result.items())
                )
            ]

        elif name == "generate-component":
            logger.info(f"Starting generate-component tool with arguments: {arguments}")
            
            if not arguments or not arguments.get("description") or not arguments.get("output_path"):
                logger.error("Missing required arguments for generate-component tool")
                raise ValueError("Component description and output path are required")

            try:
                # Generate Python code using the model
                logger.info(f"Calling Python model with description: {arguments['description'][:100]}...")
                python_code = call_python_model(arguments["description"])
                logger.debug(f"Received raw Python code response of length: {len(python_code)}")

                # Extract the Python code from the response
                logger.info("Parsing Python code from model response")
                parsed_code = parse_python_code(python_code)
                
                #DEBUG
                logger.info(f"Parsed Python code from parse_python_code function {parsed_code}")
                
                if not parsed_code:
                    logger.error("Failed to extract valid Python code from model response")
                    raise ValueError("Failed to generate valid Python code")
                logger.debug(f"Parsed Python code of length: {len(parsed_code)}")

                # Convert Python code to JSONL format
                logger.info("Converting Python code to one-line format")
                python_code_one_line = convert_python_one_line(parsed_code)
                logger.debug(f"Converted one-line code length: {len(python_code_one_line)}")
               
                # Generate JSON using the model
                input_output_data = get_last_sentence(arguments["description"])
                logger.info(f"Calling JSON model with input/output data: {input_output_data[:100]}...")
                json_response = call_json_model(python_code_one_line, input_output_data)
                logger.debug(f"Received JSON response of length: {len(json_response)}")
                
                # Save both Python and JSON versions
                output_base = arguments["output_path"].rstrip("/")
                logger.info(f"Preparing to save files with base path: {output_base}")

                # Save Python file
                python_path = f"{output_base}_component.py"
                logger.info(f"Saving Python file to: {python_path}")
                with open(python_path, 'w', encoding='utf-8') as f:
                    f.write(parsed_code)
                logger.info(f"Successfully saved Python file ({len(parsed_code)} bytes)")

                # Save JSON file
                json_path = f"{output_base}_component.json"
                logger.info(f"Saving JSON file to: {json_path}")
                    
                try:
                    parsed_json = json.loads(json_response)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_json, f, indent=2)
                    logger.info(f"Successfully saved JSON file ({len(json_response)} bytes)")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    raise ValueError(f"Failed to parse generated JSON: {e}")
                
                logger.info("Component generation completed successfully")
                return [
                    types.TextContent(
                        type="text",
                        text=f"Component generated successfully!\nPython file: {python_path}\nJSON file: {json_path}"
                    )
                ]
            except Exception as e:
                logger.error(f"Error in generate-component tool: {str(e)}", exc_info=True)
                raise
        
        else:
            logger.error(f"Unknown tool requested: {name}")
            raise ValueError(f"Unknown tool: {name}")

    except (requests.RequestException, ValueError) as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error in flow operation: {str(e)}"
            )
        ]
    
async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="langflow",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )