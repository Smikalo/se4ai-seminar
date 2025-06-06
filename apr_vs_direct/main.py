# main.py (updated with debugging)
import os
import requests
import json
import concurrent.futures
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

load_dotenv()


class AgentState(TypedDict):
    user_query: str
    plan: List[Dict[str, Any]]
    executed_steps: List[str]
    results: Dict[str, str]
    final_answer: str


class Expert:
    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

        # Initialize tools properly
        self.rag_search_tool = self._create_rag_search_tool()
        self.mcp_metadata_tool = self._create_mcp_metadata_tool()
        self.tools = [self.rag_search_tool, self.mcp_metadata_tool]

    def _create_rag_search_tool(self):
        @tool
        def rag_search_tool(query: str) -> str:
            """Searches the document database for specific content."""
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/rag/search",
                    json={"query": query},
                    timeout=10
                )
                response.raise_for_status()
                results = response.json()

                if not results:
                    return "No results found"

                content = results[0].get('content', '')
                source = os.path.basename(results[0].get('source', 'unknown'))
                return f"From {source}: {content}"

            except requests.exceptions.RequestException as e:
                return f"Error connecting to document service: {str(e)}"
            except Exception as e:
                return f"Unexpected error: {str(e)}"

        return rag_search_tool

    def _create_mcp_metadata_tool(self):
        @tool
        def mcp_metadata_tool() -> str:
            """Retrieves metadata about available documents."""
            try:
                response = requests.get("http://127.0.0.1:5000/mcp/metadata", timeout=5)
                response.raise_for_status()
                metadata = response.json()
                return "\n".join([os.path.basename(doc['source']) for doc in metadata])
            except requests.exceptions.RequestException as e:
                return f"Error getting metadata: {str(e)}"

        return mcp_metadata_tool

    @tool
    def rag_search_tool(self, query: str) -> str:
        """Searches the document database for specific content."""
        try:
            response = requests.post(
                "http://127.0.0.1:5000/rag/search",
                json={"query": query},
                timeout=10
            )
            response.raise_for_status()
            results = response.json()

            if not results:
                return "No results found"

            # Extract the first relevant result
            content = results[0].get('content', '')
            source = os.path.basename(results[0].get('source', 'unknown'))
            return f"From {source}: {content}"

        except requests.exceptions.RequestException as e:
            return f"Error connecting to document service: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    @tool
    def mcp_metadata_tool(self) -> str:
        """Retrieves metadata about available documents."""
        try:
            response = requests.get("http://127.0.0.1:5000/mcp/metadata", timeout=5)
            response.raise_for_status()
            metadata = response.json()
            return "\n".join([os.path.basename(doc['source']) for doc in metadata])
        except requests.exceptions.RequestException as e:
            return f"Error getting metadata: {str(e)}"

    def run(self, query: str) -> Dict[str, str]:
        try:
            # Directly use the appropriate tool based on query
            if "vacation" in query.lower() and self.name == "HR_Expert":
                result = self.rag_search_tool.invoke({"query": query})
            elif "revenue" in query.lower() and self.name == "Finance_Expert":
                result = self.rag_search_tool.invoke({"query": query})
            else:
                result = "Query not relevant to this expert"

            return {"output": result}
        except Exception as e:
            return {"output": f"Error processing query: {str(e)}"}

class Orchestrator:
    def __init__(self, experts: List[Expert]):
        self.experts = {expert.name: expert for expert in experts}
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")

    def create_plan(self, state: AgentState) -> dict:
        query = state['user_query']
        plan = []

        if "vacation" in query.lower():
            plan.append({
                "expert": "HR_Expert",
                "query": "What is the company's vacation policy?"
            })

        if "revenue" in query.lower():
            plan.append({
                "expert": "Finance_Expert",
                "query": "What was the total revenue in 2023?"
            })

        if not plan:
            plan.append({
                "expert": "HR_Expert",
                "query": query
            })

        return {"plan": plan}

    def execute_plan(self, state: AgentState) -> dict:
        print(f"\n⚡ Executing plan: {state['plan']}")
        results = {}
        executed_steps = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_step = {}
            for step in state["plan"]:
                future = executor.submit(self._run_step, step)
                future_to_step[future] = step

            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                key = f"{step['expert']}: {step['query']}"
                try:
                    result = future.result()
                    print(f"✔️ Step result: {key} → {result}")
                    results[key] = result
                    executed_steps.append(str(step))
                except Exception as e:
                    error_msg = f"Step failed: {str(e)}"
                    print(f"❌ {error_msg}")
                    results[key] = error_msg
                    executed_steps.append(f"{step} - FAILED")

        return {"results": results, "executed_steps": executed_steps}

    def _run_step(self, step: dict) -> str:
        expert = self.experts.get(step["expert"])
        if not expert:
            return "Expert not found"

        # Properly invoke the expert with the query
        try:
            result = expert.run(step["query"])
            return result.get("output", "No output")
        except Exception as e:
            return f"Error: {str(e)}"

    # In the Orchestrator's aggregate_results method, update the prompt to:
    def aggregate_results(self, state: AgentState) -> dict:
        print("\n🧩 Aggregating results...")
        prompt = f"""Please provide direct answers to these specific questions:

        Original Query: '{state['user_query']}'

        Extracted Information:
        {json.dumps(state['results'], indent=2)}
        
        Only include information that directly answers the question. 
        If information is missing, state exactly what is missing."""

        response = self.llm.invoke(prompt)
        return {"final_answer": response.content}


if __name__ == '__main__':
    print("🚀 Starting system...")
    try:
        print("🔌 Checking Flask server connection...")
        response = requests.get("http://127.0.0.1:5000/mcp/metadata", timeout=5)
        if response.status_code == 200:
            print("✅ Flask server is running")
            print(f"Available documents: {response.json()}")
        else:
            print(f"⚠️ Unexpected server response: {response.status_code}")
    except Exception as e:
        print(f"❌ Flask server check failed: {str(e)}")
        print("Please start data_endpoints.py first!")
        exit(1)

    hr_expert = Expert(name="HR_Expert", expertise="Human Resources")
    finance_expert = Expert(name="Finance_Expert", expertise="Financial Reports")
    experts = [hr_expert, finance_expert]

    orchestrator = Orchestrator(experts)
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", orchestrator.create_plan)
    workflow.add_node("executor", orchestrator.execute_plan)
    workflow.add_node("aggregator", orchestrator.aggregate_results)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "aggregator")
    workflow.add_edge("aggregator", END)
    app = workflow.compile()

    user_query = "What is the company's vacation policy and what was the total revenue in 2023 and who is Donald Trump and what was revenue in 2024 considering what it was in 2023?"
    initial_state = {
        "user_query": user_query,
        "plan": [],
        "executed_steps": [],
        "results": {},
        "final_answer": ""
    }

    print(f"\n🤖 Processing query: {user_query}")
    final_state = app.invoke(initial_state)

    print("\n💡 Final Answer:")
    print(final_state["final_answer"])