import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rouge_score import rouge_scorer

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

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

# ROUGE Evaluation on 75 samples
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

print("Running evaluation on 75 samples...")

for i in range(75):
    sample = dataset["test"][i]
    article = sample["article"]
    reference = sample["highlights"]

    doc_type = classify_chain.invoke({"article": article[:500]})
    strategy = strategy_chain.invoke({"doc_type": doc_type})
    summary = summarize_chain.invoke({"strategy": strategy, "article": article})
    final_summary = critique_chain.invoke({"article": article[:500], "summary": summary})

    scores = scorer.score(reference, final_summary)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

    print(f"Sample {i+1}/75 done — ROUGE-1: {scores['rouge1'].fmeasure:.4f}")

print("\n--- AVERAGE ROUGE SCORES ACROSS 75 SAMPLES ---")
print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")