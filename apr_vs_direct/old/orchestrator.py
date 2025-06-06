from dotenv import load_dotenv
load_dotenv()
import asyncio
from expert import compliance_agent, technical_agent


async def answer_query_via_orchestrator(query: str) -> str:
    # Determine relevant agents for the query
    query_lower = query.lower()
    agents_to_use = []
    # Simple routing logic based on keywords:
    if any(term in query_lower for term in ["compliance", "policy", "regulation", "law", "GDPR", "audit"]):
        agents_to_use.append(("Compliance", compliance_agent))
    if any(term in query_lower for term in ["error", "issue", "technical", "IT", "system", "bug"]):
        agents_to_use.append(("Technical", technical_agent))
    # If no specific keywords, default to using all available agents
    if not agents_to_use:
        agents_to_use = [("Compliance", compliance_agent), ("Technical", technical_agent)]

    # Run selected agents in parallel to get their answers
    results = await asyncio.gather(*[
        agent.ainvoke(query) for _, agent in agents_to_use
    ])
    # The .arun method is the async version of agent.run, provided by LangChain for async support.

    # Merge results from agents
    partial_answers = {
        domain: answer.get("output", str(answer)) if isinstance(answer, dict) else str(answer)
        for (domain, _), answer in zip(agents_to_use, results)
    }

    # Simple merging logic:
    final_answer = ""
    if len(partial_answers) == 1:
        # Only one domain was needed
        final_answer = list(partial_answers.values())[0]
    else:
        # If we have multiple contributions, combine them.
        # We'll just join them with a connector for demonstration.
        final_answer = " \n".join([f"*{domain} perspective:* {ans}" for domain, ans in partial_answers.items()])

    # Final verification (guardrail): if no useful info in answers, respond accordingly
    if all("no" in ans.lower() and ("not find" in ans.lower() or "no information" in ans.lower())
           for ans in partial_answers.values()):
        final_answer = "I'm sorry, I could not find any information related to your query in the company data."
    return final_answer
