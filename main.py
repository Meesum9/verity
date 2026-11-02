import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from chains import first_responder, revisor
from tool_executor import execute_tools

load_dotenv()

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "1"))


def draft_node(state: MessagesState):
    """Draft the initial response."""
    print("\n✍️  [Node: draft] Generating initial response, critique, and research queries...")
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def execute_tools_node(state: MessagesState):
    """Execute search queries using GPT's web search capability."""
    print("\n🔍 [Node: execute_tools] Performing live web searches with current GPT model...")
    return execute_tools(state)


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    print("\n📝 [Node: revise] Revising draft using sourced search evidence and citations...")
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    if count_tool_visits >= MAX_ITERATIONS:
        print(f"\n🏁 Reflexion complete! Reached {count_tool_visits}/{MAX_ITERATIONS} iteration(s).")
        return END
    print(f"\n🔄 Continuing reflexion loop ({count_tool_visits}/{MAX_ITERATIONS} iterations)...")
    return "execute_tools"


# Build the Reflexion StateGraph
builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools_node)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("📊 REFLEXION AGENT GRAPH STRUCTURE")
    print("=" * 60)

    # 1. Mermaid Graph Representation
    print("\n[Mermaid Syntax]:")
    print(graph.get_graph().draw_mermaid())

    # 2. ASCII Graph Representation
    print("\n[ASCII Graph]:")
    try:
        graph.get_graph().print_ascii()
    except Exception as e:
        print(f"Could not print ASCII graph: {e}")

    # 3. Save graph as PNG image
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("\n✅ Successfully rendered and saved graph diagram to: graph.png")
    except Exception as e:
        print(f"\n⚠️  Could not save graph.png (requires network or pyppeteer): {e}")

    print("\n" + "=" * 60)
    print("🚀 RUNNING REFLEXION AGENT WORKFLOW")
    print("=" * 60)

    query = (
        "Write about AI-Powered SOC / autonomous soc problem domain, "
        "list startups that do that and raised capital."
    )
    print(f"\n❓ User Question:\n{query}\n")

    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    print("\n" + "=" * 60)
    print("📋 FINAL AGENT OUTPUT")
    print("=" * 60)

    # Extract final answer, reflection, and references
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        args = last_message.tool_calls[0]["args"]
        print("\n🏆 REVISED ANSWER:\n")
        print(args.get("answer", ""))

        refs = args.get("references", [])
        if refs:
            print("\n📚 REFERENCES & CITATIONS:")
            for ref in refs:
                print(f"  - {ref}")

        reflection = args.get("reflection", {})
        if reflection:
            print("\n🧐 REFLECTION / SELF-CRITIQUE:")
            if isinstance(reflection, dict):
                print(f"  - Missing: {reflection.get('missing', '')}")
                print(f"  - Superfluous: {reflection.get('superfluous', '')}")
            else:
                print(f"  - {reflection}")
    else:
        print(last_message.content)