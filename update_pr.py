import os
import sys
from github import Github
import numpy as np
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers.string import StrOutputParser
import pinecone
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel, AutoTokenizer
import torch

# Pinecone setup
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name="ai-code-analyzer")

# CodeBERT setup for embeddings
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

print("Model loaded successfully")
def generate_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    if np.issubdtype(embeddings.dtype, np.number):
        return embeddings.flatten()
    else:
        raise ValueError("Embeddings contain non-numeric values")


def fetch_and_index_codebase(repo):
    try:
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            elif file_content.name.endswith(('.js', '.java', '.cpp')):
                try:
                    code = file_content.decoded_content.decode('utf-8')
                    embedding = generate_embedding(code)
                    if embedding.dtype.type is np.str_:
                        raise TypeError("Embedding is not purely numeric.")
                    print(f"Indexing file {file_content.path}")
                    print(f"Embedding: {embedding.tolist()}")
                    index.upsert([(file_content.path, embedding.tolist())])
                except Exception as inner_e:
                    print(f"Failed processing file {file_content.path}: {inner_e}")
    except Exception as e:
        print(f"Error processing repository files: {e}")




def get_pull_request_diffs(pull_request):
    return [
        {"filename": file.filename, "patch": file.patch, "path": file.filename}
        for file in pull_request.get_files()
    ]


def format_data_for_openai(diffs):
    print("Formatting data for OpenAI...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))
    document_vectorstore = PineconeVectorStore(index_name="ai-code-analyzer", embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_API_KEY'))
    retriever = document_vectorstore.as_retriever()

    print("Retrieving context...")
    formatted_text = '\n'.join([f"File: {diff['filename']}\nDiff:\n{diff['patch']}" for diff in diffs])

    try:
        context = retriever.invoke(formatted_text)
        print(f"Context retrieved successfully: {context[:100]}")  # Print a snippet of the context for debugging
    except Exception as e:
        print(f"An unexpected error occurred while retrieving context: {e}")
        return None

    changes = "\n".join([f"File: {file['filename']}\nDiff:\n{file['patch']}" for file in diffs])
    prompt = (
        "Analyze the following code changes for potential refactoring opportunities to make the code more readable and efficient, "
        "and point out areas that could cause potential bugs and performance issues.\n\n"
        "Context of changes:\n" + context +  # Using the context directly in the prompt
        "\n\nDetailed changes:\n" + changes +
        "\n\nProvide suggestions based on the details and context provided above."
    )

    print("Generating suggestions using AI...")
    llm = ChatOpenAI(temperature=0.5, api_key=os.getenv('OPENAI_API_KEY'))
    try:
        results = llm.invoke(prompt)
        print(f"Results: {results.content}")
    except Exception as e:
        print(f"Error invoking AI model: {e}")
        return "Failed to generate suggestions due to an error."

    return results.content


def call_openai(prompt):
    client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-3.5-turbo-0125")
    try:
        print("Making LLM call...")
        if isinstance(prompt, list):
            prompt = ' '.join(map(str, prompt))  # Convert list elements to string if not already, and join them

        print(f"Prompt: {prompt}")


        messages = [
            {"role": "system", "content": "You are an AI trained to help refactor code by giving suggestions for improvements as well as code snippets to replace the existing code."},
            {"role": "user", "content": prompt}
        ]
        response = client.invoke(input=messages)
        parser = StrOutputParser()
        return parser.invoke(input=response)
    except Exception as e:
        print(f'Error making LLM call: {e}')
        # Consider adding more specific logging here or a method to handle repeated failures gracefully
        return "Failed to generate suggestions due to an error. Please try again."


def post_comments_to_pull_request(pull_request, comments):
    # Check if the comments contain more than just whitespace
    if comments.strip():
        # Attempt to post a single, consolidated comment to the pull request
        try:
            # Assuming you're posting a general PR comment, not an in-line comment
            pull_request.create_issue_comment(comments)
        except Exception as e:
            print(f"Failed to post comment: {e}")







def main():
    try:
        print("Starting the code analysis process...")
        g = Github(os.getenv('GITHUB_TOKEN'))  # Initialize GitHub API with token
        repo_path = os.getenv('REPO_PATH')
        repo = g.get_repo(repo_path)  # Get the repo object

        # Fetch and index the codebase at the start or update if necessary
        fetch_and_index_codebase(repo)
        
        
        pr_number = int(os.getenv('PR_NUMBER'))
        pull_request = repo.get_pull(pr_number)  # Get the pull request

        diffs = get_pull_request_diffs(pull_request)  # Get the diffs of the pull request
        print(f"about to call openai with diffs")
        prompt = format_data_for_openai(diffs)  # Format data for OpenAI

        print(f"about to call openai with prompt")
        suggestions = call_openai(prompt)  # Call OpenAI to get suggestions for code improvement
        print(f"Suggestions: {suggestions}")

        post_comments_to_pull_request(pull_request, suggestions)  # Post suggestions as comments on the PR
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
