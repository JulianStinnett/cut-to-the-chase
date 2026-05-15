import os
import json
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

dataset = load_dataset("cnn_dailymail", "3.0.0")
parser = StrOutputParser()

# Load fine-tuned FLAN-T5 for Step 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_tokenizer = AutoTokenizer.from_pretrained("./flan-t5-finetuned")
ft_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-finetuned").to(device)

def finetuned_summarize(strategy, article):
    prompt = f"summarize: {article[:2048]}"
    inputs = ft_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = ft_model.generate(**inputs, max_new_tokens=128)
    return ft_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 1
classify_prompt = ChatPromptTemplate.from_template(
    "You are a document classifier. Respond with only one of: news article, research paper, email, legal document, other.\n\nWhat type of document is this?\n\n{article}"
)
classify_chain = classify_prompt | llm | parser

# Step 2
strategy_prompt = ChatPromptTemplate.from_template(
    "You are a summarization strategist. Given a document type, describe in one sentence the best summarization strategy.\n\nDocument type: {doc_type}"
)
strategy_chain = strategy_prompt | llm | parser

# Step 4
critique_prompt = ChatPromptTemplate.from_template(
    "You are a summarization quality reviewer. Review this summary and improve it if needed.\n\nOriginal article (first 500 chars): {article}\n\nSummary to review: {summary}"
)
critique_chain = critique_prompt | llm | parser

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores, rouge2_scores, rougeL_scores, per_sample = [], [], [], []

print("Running evaluation with fine-tuned FLAN-T5 on 25 samples...")

for i in range(25):
    sample = dataset["test"][i]
    article = sample["article"]
    reference = sample["highlights"]

    doc_type = classify_chain.invoke({"article": article[:500]})
    strategy = strategy_chain.invoke({"doc_type": doc_type})
    summary = finetuned_summarize(strategy, article)
    final_summary = critique_chain.invoke({"article": article[:500], "summary": summary})

    scores = scorer.score(reference, final_summary)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)
    per_sample.append({
        "sample": i + 1,
        "rouge1": round(scores['rouge1'].fmeasure, 4),
        "rouge2": round(scores['rouge2'].fmeasure, 4),
        "rougeL": round(scores['rougeL'].fmeasure, 4),
    })

    print(f"Sample {i+1}/25 done — ROUGE-1: {scores['rouge1'].fmeasure:.4f}")

avg_r1 = sum(rouge1_scores) / len(rouge1_scores)
avg_r2 = sum(rouge2_scores) / len(rouge2_scores)
avg_rL = sum(rougeL_scores) / len(rougeL_scores)

print("\n--- AVERAGE ROUGE SCORES (Fine-tuned FLAN-T5 in pipeline) ---")
print(f"ROUGE-1: {avg_r1:.4f}")
print(f"ROUGE-2: {avg_r2:.4f}")
print(f"ROUGE-L: {avg_rL:.4f}")

results = {
    "model": "flan-t5-finetuned",
    "num_samples": 25,
    "rouge1": round(avg_r1, 4),
    "rouge2": round(avg_r2, 4),
    "rougeL": round(avg_rL, 4),
    "per_sample": per_sample,
}
with open("results_finetuned.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results_finetuned.json")