import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import END, MessageGraph
from langgraph.graph.graph import START
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from tools import ALL_TOOLS

openai_key_rtkflow = os.environ.get("OPENAI_KEY_RTKFLOW")
model = ChatOpenAI(temperature=0, openai_api_key=openai_key_rtkflow).bind_tools(ALL_TOOLS)
tool_executor = ToolExecutor(ALL_TOOLS)


def is_message(messages):
    last_message = messages[-1]
    # If there is no function call, then regard the last message as a message
    if "tool_calls" not in last_message.additional_kwargs:
        return "message"
    else:
        return "function_call"


def call_tool(messages):
    # Based on the conditional edge, we know that the last message is a function call
    last_message = messages[-1]
    tool_call = last_message.additional_kwargs["tool_calls"][0]
    function = tool_call["function"]
    function_name = function["name"]
    _tool_input = json.loads(function["arguments"] or "{}")
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
            tool=function_name,
            tool_input=_tool_input,
        )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a ToolMessage
    tool_messages = ToolMessage(
        tool_call_id=tool_call["id"],
        content=str(response),
        additional_kwargs={"name": tool_call["function"]["name"]},
        )
    return tool_messages


def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs


workflow = MessageGraph()
workflow.add_node("agent", model)
workflow.add_node("tools", call_tool)
workflow.add_conditional_edges("agent", is_message, {
    "function_call": "tools",
    "message": END,
})
workflow.add_edge("tools", "agent")
workflow.set_entry_point("agent")

graph = workflow.compile()

history = [SystemMessage(content="You are a RTKLIB agent that knows how to use the tool, but do not use tool call unless it is really necessary and you have all the information needed. Maximum 1 tool call per user message.")]
while True:
    user = input('User (q/Q to quit): ')
    if user in {'q', 'Q'}:
        print('AI: Byebye')
        break
    history.append(HumanMessage(content=user))
    for output in graph.stream(history):
        if END in output or START in output:
            continue
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            if _is_tool_call(value):
                print(value.additional_kwargs['tool_calls'][0]['function'])
            else:
                print(value.content)
        print("\n---\n")
    history = output[END]

