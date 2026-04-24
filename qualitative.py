import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rouge_score import rouge_scorer

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

dataset = load_dataset("cnn_dailymail", "3.0.0")
parser = StrOutputParser()

classify_prompt = ChatPromptTemplate.from_template(
    "You are a document classifier. Respond with only one of: news article, research paper, email, legal document, other.\n\nWhat type of document is this?\n\n{article}"
)
strategy_prompt = ChatPromptTemplate.from_template(
    "You are a summarization strategist. Given a document type, describe in one sentence the best summarization strategy.\n\nDocument type: {doc_type}"
)
summarize_prompt = ChatPromptTemplate.from_template(
    "You are a document summarization assistant.\n\nUsing this strategy: {strategy}\n\nSummarize the following document in 3-5 sentences:\n\n{article}"
)
critique_prompt = ChatPromptTemplate.from_template(
    "You are a summarization quality reviewer. Review this summary and improve it if needed.\n\nOriginal article (first 500 chars): {article}\n\nSummary to review: {summary}"
)

classify_chain = classify_prompt | llm | parser
strategy_chain = strategy_prompt | llm | parser
summarize_chain = summarize_prompt | llm | parser
critique_chain = critique_prompt | llm | parser

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Indices chosen to surface a range of outcomes.
# Adjust these after seeing scores to swap in better success/failure examples.
SAMPLE_INDICES = [0, 1, 2, 3, 4]

print("=" * 70)
print("  QUALITATIVE EVALUATION — 5 Example Outputs (GPT-4o)")
print("=" * 70)

for idx in SAMPLE_INDICES:
    sample = dataset["test"][idx]
    article = sample["article"]
    reference = sample["highlights"]

    doc_type = classify_chain.invoke({"article": article[:500]})
    strategy = strategy_chain.invoke({"doc_type": doc_type})
    summary = summarize_chain.invoke({"strategy": strategy, "article": article})
    final_summary = critique_chain.invoke({"article": article[:500], "summary": summary})

    scores = scorer.score(reference, final_summary)
    r1 = scores['rouge1'].fmeasure
    r2 = scores['rouge2'].fmeasure
    rL = scores['rougeL'].fmeasure

    # Label based on ROUGE-1 threshold
    if r1 >= 0.40:
        verdict = "SUCCESS"
    elif r1 >= 0.25:
        verdict = "PARTIAL"
    else:
        verdict = "FAILURE"

    print(f"\n{'─' * 70}")
    print(f"  Sample #{idx + 1}  |  {verdict}  |  ROUGE-1: {r1:.4f}  ROUGE-2: {r2:.4f}  ROUGE-L: {rL:.4f}")
    print(f"{'─' * 70}")
    print(f"\nDoc type  : {doc_type}")
    print(f"Strategy  : {strategy}")
    print(f"\nARTICLE (first 300 chars):\n{article[:300]}...")
    print(f"\nREFERENCE SUMMARY:\n{reference}")
    print(f"\nAGENT SUMMARY:\n{final_summary}")

print(f"\n{'=' * 70}\n")
