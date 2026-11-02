#https://chatgpt.com/share/6ae78927-5924-83ec-bca6-74a6ce4cc962

from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient #this library is used for making mcp clients

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model="gpt-5")
#to see how mcp server looks -> see How to build MCP Client using LangGraph | Agentic AI using LangGraph | CampusX


# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio", #tells if our server is remote or in same machine.
            "command": "python3",          
            "args": ["/Users/nitish/Desktop/mcp-math-server/main.py"], #in this file we have mcp server. 
            #can add mutliple servers together too.
        },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        } #a remote mcp server (2nd mcp server)
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools() #this fetches all the tools we have on that server

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Give me all my expenses for the month of Nov from 1 Nov to 30 Nov")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())