from openai import OpenAI
from dotenv import load_dotenv
import sys, os
import json
from langchain.prompts import PromptTemplate
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    }, 
    {
        "type": "function",
        "function": {
            "name": "get_accurate_information",
            "description": "The user give the raw response to this and it will exract location from this in given structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "raw string response having location state and city",
                    }
                },
                "required": ["response"]
            },
        }
    }
]
def send_messages(messages, prompt=None):
    if prompt:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        
        )

    return response.choices[0].message

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    You will be given a text containing a city name or related information.

    Your task:
    - Extract the most relevant **city and state** in the format "City, State" (e.g., "Kanpur, Uttar Pradesh").
    - If no valid city/state pair is found, set "location" to "None".
    - Respond **strictly** in valid JSON format only — no extra text.

    Text:
    {context}

    Output:
    {{"location": "<extracted_location>", "tool_call": "get_weather"}}
    """
)


# input("Ask from AI bot : ")

messages = [{"role": "user", "content":"what is the wheather of kanpur?" }]
# messages = [{"role": "user", "content": "How's the weather in Hangzhou, Zhejiang?"}]
# messages = [{"role": "user", "content": "How are you?"}]
def get_weather(location):
    # Simulate a real function
    return f"The weather in {location} is 24°C."

def get_accurate_information(response):
    print("response >>>>>>", response)
    # pass the string to the LLM with the proper prompt
    location = ask_query_to_llm(response=response, prompt=prompt)
    print("location===========>", location)
    return location





def ask_query_to_llm(messages=None, response=None, prompt=None):
    # Build messages properly if prompt+response are given
    if prompt and response:
        prompt_text = prompt.format(context=response)  # render template
        print("prompt_text===========>", prompt_text)
        messages = [
            {"role": "system", "content": "You are a location extraction assistant, e.g. San Francisco, CA"},
            {"role": "user", "content": prompt_text},
        ]
    elif not messages:
        raise ValueError("Either messages or (prompt and response) must be provided.")

    message = send_messages(messages)
    print(f"User>\t {messages[-1]['content']}")
    return message

def is_tool_call(message, tool_call = None):
    messages.append(message)  # Add model’s tool call request
    print("Tool call happening and message printing :", message.tool_calls)
    # if "location" in message.content.lower():
    #     response = get_accurate_information(message)
    #     print("----------125--------->",response)
    #     if response.get("tool_call"):
    #         print("----------126--------->",response.get("tool_call") )
    #         is_tool_call(response.location, response.get("tool_call"))
    if message.tool_calls:
        for tool_call in message.tool_calls:
            print("Tool call happening :", tool_call)
            name = tool_call.function.name
            print("tool name for this call:", name)
            print("tool message for this call:", message)

            args = json.loads(message.tool_calls)
            print("#########", args)
            # Dynamically find and run the function
            result = globals()[name](**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
            ask_query_to_llm(messages=messages)
            print("------------->",tool_call, result)
            # Append tool’s response for each tool_call

        message = send_messages(messages)
    else:
        return message
    return message
    # Send all results back together
message = ask_query_to_llm(messages)
print("line130: ", message.content)
# if message.tool_calls or :
message = is_tool_call(message)
print(f"Model>\t {message}")