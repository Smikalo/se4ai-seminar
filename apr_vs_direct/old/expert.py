from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from mcp import mcp_tool, rag_tool

def create_expert_agent(domain: str):
    """
    Create a LangChain agent specialized in a given domain.
    Domain can be "Compliance", "Technical", etc. The agent will be prompted to use tools and domain knowledge.
    """
    # Choose an LLM (we use GPT-4 via ChatOpenAI, could use gpt-3.5-turbo for lower latency if needed)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Domain-specific prompting
    prefix = f"""
    You are a knowledgeable {domain} expert for a large car leasing company.

    Your job is to answer {domain.lower()} questions accurately using the available tools ONLY.

    - Do not guess or use outside knowledge.
    - You must ALWAYS use a tool to answer the question.
    - If the tools do not return relevant data, you MUST say: "I could not find information relevant to this query."
    - NEVER fabricate data or rely on general knowledge.
    - NEVER fabricate data or rely on general knowledge.
    - NEVER fabricate data or rely on general knowledge.

    Use tools to search before answering.
    """
    format_instructions = """Use the following format for reasoning:
    Thought: think about what is asked and what information you need.
    Action: the name of the tool to use, one of {tool_names}
    Action Input: the input to the tool
    Observation: the result of the action
    ... (you can Thought/Action/Observation multiple times if needed) ...
    Thought: once you have enough information, formulate the final answer.
    Final Answer: your answer to the user's question.
    """

    suffix = "\nBegin!\n{input}\n"

    # Create the agent with tools
    agent = initialize_agent(
        tools=[mcp_tool, rag_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix,
            "format_instructions": format_instructions,
            "suffix": suffix
        },
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


# Create instances of agents for different domains
compliance_agent = create_expert_agent("Compliance")
technical_agent = create_expert_agent("Technical")
# (We could similarly create a business_process_agent if needed)
