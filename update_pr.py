import os
import sys
import numpy as np
from github import Github
import numpy as np
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers.string import StrOutputParser
from pinecone import Pinecone

# Pinecone setup
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name="ai-code-analyzer")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_embedding(text: str, model: str = "text-embedding-3-large") -> np.array:
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return np.array(embedding)
    except Exception as e:
        print(f"Error in generating embedding: {e}")
        return None

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
                    if embedding is not None:
                        print(f"Indexing file {file_content.path}")
                        index.upsert([(file_content.path, embedding.tolist())])
                    else:
                        print(f"Failed to generate embedding for file {file_content.path}")
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))
    document_vectorstore = PineconeVectorStore(index_name="ai-code-analyzer", embedding=embeddings, pinecone_api_key=os.getenv('PINECONE_API_KEY'))
    retriever = document_vectorstore.as_retriever()

    formatted_text = '\n'.join([f"File: {diff['filename']}\nDiff:\n{diff['patch']}" for diff in diffs])

    try:
        context_response = retriever.invoke(formatted_text)
        if context_response and context_response.matches:
            context = ' '.join([match.metadata.get('text', 'No metadata found') for match in context_response.matches])
            print(f"Context retrieved successfully: {context[:100]}")  # Print the first 100 characters of context
        else:
            print("No relevant context found")
            context = "No relevant context available."
    except Exception as e:
        print(f"An unexpected error occurred while retrieving context: {e}")
        return None

    changes = "\n".join([f"File: {file['filename']}\nDiff:\n{file['patch']}" for file in diffs])
    prompt = (
        "Analyze the following code changes for potential refactoring opportunities to make the code more readable and efficient, "
        "and point out areas that could cause potential bugs and performance issues.\n\n"
        "Also check the context and make sure the new code changes are not conflicting with the existing code and do not create duplicate code.\n\n"
        "Each suggestion should be concise and limited to a few short sentences of explanation followed by a code snippet if applicable. Make sure to reference the file where the suggestion should be made.\n\n"
        "Context of changes:\n" + context +
        "\n\nDetailed changes:\n" + changes + "\n\n"
    )

    return prompt

def call_openai(prompt):
    client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-3.5-turbo-0125")
    try:
        if isinstance(prompt, list):
            prompt = ' '.join(map(str, prompt))

        messages = [
            {"role": "system", "content": "You are an AI trained to help refactor code by giving suggestions for improvements as well as code snippets to replace the existing code."},
            {"role": "user", "content": prompt}
        ]
        response = client.invoke(input=messages)
        parser = StrOutputParser()
        return parser.invoke(input=response)
    except Exception as e:
        print(f'Error making LLM call: {e}')
        return "Failed to generate suggestions due to an error. Please try again."

def post_comments_to_pull_request(pull_request, comments):
    if comments.strip():
        try:
            pull_request.create_issue_comment(comments)
        except Exception as e:
            print(f"Failed to post comment: {e}")

def main():
    try:
        g = Github(os.getenv('GITHUB_TOKEN'))
        repo_path = os.getenv('REPO_PATH')
        repo = g.get_repo(repo_path)

        fetch_and_index_codebase(repo)
        
        pr_number = int(os.getenv('PR_NUMBER'))
        pull_request = repo.get_pull(pr_number)

        diffs = get_pull_request_diffs(pull_request)
        prompt = format_data_for_openai(diffs)

        if prompt:
            suggestions = call_openai(prompt)
            print(f"Suggestions: {suggestions}")
            post_comments_to_pull_request(pull_request, suggestions)
        else:
            print("Failed to format data for OpenAI.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
