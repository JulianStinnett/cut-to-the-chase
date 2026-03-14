import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
sample = dataset["test"][0]
article = sample["article"]
reference_summary = sample["highlights"]

parser = StrOutputParser()

# Step 1 — Classify
classify_prompt = ChatPromptTemplate.from_template(
    "You are a document classifier. Respond with only one of: news article, research paper, email, legal document, other.\n\nWhat type of document is this?\n\n{article}"
)
classify_chain = classify_prompt | llm | parser

# Step 2 — Decide strategy
strategy_prompt = ChatPromptTemplate.from_template(
    "You are a summarization strategist. Given a document type, describe in one sentence the best summarization strategy.\n\nDocument type: {doc_type}"
)
strategy_chain = strategy_prompt | llm | parser

# Step 3 — Summarize
summarize_prompt = ChatPromptTemplate.from_template(
    "You are a document summarization assistant.\n\nUsing this strategy: {strategy}\n\nSummarize the following document in 3-5 sentences:\n\n{article}"
)
summarize_chain = summarize_prompt | llm | parser

# Step 4 — Critique and refine
critique_prompt = ChatPromptTemplate.from_template(
    "You are a summarization quality reviewer. Review this summary and improve it if needed.\n\nOriginal article (first 500 chars): {article}\n\nSummary to review: {summary}"
)
critique_chain = critique_prompt | llm | parser

# Run the agent loop
print("Step 1: Classifying document...")
doc_type = classify_chain.invoke({"article": article[:500]})
print(f"Document type: {doc_type}")

print("\nStep 2: Deciding summarization strategy...")
strategy = strategy_chain.invoke({"doc_type": doc_type})
print(f"Strategy: {strategy}")

print("\nStep 3: Summarizing...")
summary = summarize_chain.invoke({"strategy": strategy, "article": article})
print(f"Summary: {summary}")

print("\nStep 4: Critiquing and refining...")
final_summary = critique_chain.invoke({"article": article[:500], "summary": summary})
print(f"\nFinal Summary: {final_summary}")
print(f"\nReference Summary: {reference_summary}")