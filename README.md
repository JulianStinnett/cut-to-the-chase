Setup Instructions
Step 1 - Clone the repo
Open a terminal and run:
git clone https://github.com/JulianStinnett/cut-to-the-chase.git
cd cut-to-the-chase
Step 2 - Install dependencies
pip install langchain langchain-openai langchain-core datasets python-dotenv rouge_score bert_score
If pip is not recognized, use:
python -m pip install langchain langchain-openai langchain-core datasets python-dotenv rouge_score bert_score
Step 3 - Add your API key
Create a new file in the project folder called .env (no other extension, just .env)
Inside that file, add this one line:
OPENAI_API_KEY=your_openai_key_here
Replace your_openai_key_here with the actual key Julian sent you.
Save the file. Do NOT commit this file to GitHub.
Step 4 - Run the agent
python agent.py

Expected Output
When it runs successfully you should see something like this:
Step 1: Classifying document...
Document type: news article

Step 2: Deciding summarization strategy...
Strategy: Focus on the main headline and first few paragraphs...

Step 3: Summarizing...
Summary: ...

Step 4: Critiquing and refining...
Final Summary: ...

Reference Summary: ...

Common Errors
pip not recognized:
Use python -m pip install ... instead of pip install ...
OPENAI_API_KEY error or AuthenticationError:
Make sure your .env file exists in the project folder and contains the correct key with no extra spaces or quotes.
Model not found error:
Make sure your OpenAI account has GPT-4o access. If not, change model="gpt-4o" to model="gpt-3.5-turbo" in agent.py line 10.
HuggingFace warning about HF_TOKEN:
This is harmless — ignore it. The dataset will still download fine.
Pydantic warning about Python 3.14:
This is harmless — ignore it. The agent will still run fine.
