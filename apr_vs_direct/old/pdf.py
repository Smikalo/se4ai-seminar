# Install necessary packages (if not already installed)
# !pip install langchain openai

import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

# Dummy PDF contents for RAG (in reality, load and chunk actual PDFs)
pdf_texts: List[str] = [
    "Company Policy: All customer data must comply with GDPR and local data protection laws. "
    "Failure to comply can result in fines. The company has a Data Protection Officer for ensuring compliance.",
    "Technical Guide: The leasing platform consists of a frontend web portal and a backend API. "
    "Common errors: Error 5001 indicates payment processing failure; Error 5002 indicates contract not found.",
    "Business Process: Lease approval requires credit check, identity verification, and management sign-off. "
    "Compliance is involved if the lease exceeds €50,000 to ensure anti-fraud measures."
]

# Dummy structured data for MCP (for demonstration, a simple dictionary)
mcp_database: Dict[str, str] = {
    "GDPR fine amount": "Under GDPR, fines can be up to 4% of annual revenue or €20 million, whichever is higher.",
    "Data Protection Officer": "The Data Protection Officer is Jane Doe, appointed in 2023.",
    "Error 5001 resolution": "Error 5001 (payment failure) can often be resolved by resetting the payment gateway credentials.",
    "Max lease without compliance": "Leases under €50,000 do not require additional compliance approval."
}
