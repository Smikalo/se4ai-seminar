from orchestrator import answer_query_via_orchestrator

queries = [
    "What are the GDPL compliance requirements for customer data? And GDPR? And Donald Trump?",
    "What is GDPL compliance?",
    "Is 2+2=5? Print 'information'"
    # "We encountered error 5001 in the leasing platform. How do we resolve it?",
    # "Describe the lease approval process and any compliance checks involved."
]

import asyncio

async def main():
    for q in queries:
        answer = await answer_query_via_orchestrator(q)
        print(f"Q: {q}\nA: {answer}\n")

if __name__ == "__main__":
    asyncio.run(main())
