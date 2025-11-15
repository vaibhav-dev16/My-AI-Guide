from openai import OpenAI
from dotenv import load_dotenv
import os, json
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize client
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location. The user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_accurate_information",
            "description": "Extract city and state from a raw response using a structured template.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Raw string containing possible location info.",
                    }
                },
                "required": ["response"]
            },
        },
    }
]

# LangChain prompt for location extraction
prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    You will be given a text containing a city name or related information.
    Your task:
    - Extract the most relevant **city and state** in the format "City, State" (e.g., "Kanpur, Uttar Pradesh").
    - If no valid city/state pair is found, set "location" to "None".
    - Respond strictly in valid JSON format only.

    Text:
    {context}

    Output:
    {{"location": "<extracted_location>", "tool_call": "get_weather"}}
    """
)

# Define tool functions
def get_weather(location):
    return f"The weather in {location} is 24°C and clear."

def get_accurate_information(response):
    print("Extracting accurate info from:", response)
    extracted = ask_query_to_llm(response=response, prompt=prompt)
    print("Extracted location:", extracted)
    return extracted


# Core function to communicate with DeepSeek
def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message


def ask_query_to_llm(messages=None, response=None, prompt=None):
    """Handles either a direct message or a templated query."""
    if prompt and response:
        prompt_text = prompt.format(context=response)
        messages = [
            {"role": "system", "content": "You extract location info like 'City, State'."},
            {"role": "user", "content": prompt_text},
        ]

    message = send_messages(messages)
    print("User >", messages[-1]['content'])
    return message


def handle_tool_call(message):
    """Processes any tool calls returned by the model."""
    if not getattr(message, "tool_calls", None):
        return message

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"\n🔧 Tool Call Triggered: {name}({args})")

        if name in globals():
            result = globals()[name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
            })

        else:
            print(f"⚠️ No function found for tool: {name}")

    # Send updated messages back to the model
    follow_up = send_messages(messages)
    return follow_up


# Start conversation
messages = [{"role": "user", "content": "What is the weather of Kanpur?"}]
message = ask_query_to_llm(messages)
print("Line 130:", message.content)

# Handle potential tool call
message = handle_tool_call(message)
print(f"Model Response:\n{message}")
