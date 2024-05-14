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
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


import openai

# Pinecone setup
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name="ai-code-analyzer")
model="text-embedding-3-large"

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)


def generate_embedding(text, model="text-embedding-3-large"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    document = Document(page_content=text)
    documents = text_splitter.split_documents([document])

    print(f"Going to add {len(documents)} to Pinecone.")
    print(f"Documents: {documents}")

    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', api_key=os.getenv('OPENAI_API_KEY'))

    PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name='ai-code-analyzer', pinecone_api_key=PINECONE_API_KEY)
    print("Loading to vectorstore (pinecone) complete")

    print("Model loaded successfully")
    return embeddings



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
                    print(f"Processing file {file_content.path}")
                    print(f"Code: {code}")
                    generate_embedding(code)
                
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

    print(f"Formatted text: {formatted_text}")

    try:
        context = retriever.get_relevant_documents(formatted_text)
        print(f"Context response: {context}")
        

    except Exception as e:
        print(f"An unexpected error occurred while retrieving context: {e}")
        return None

    changes = "\n".join([f"File: {file['filename']}\nDiff:\n{file['patch']}" for file in diffs])

    template = PromptTemplate(
        template=
            "Analyze the following code changes for potential refactoring opportunities to make the code more readable and efficient, "
            "and point out areas that could cause potential bugs and performance issues. Also point out if the new code in changes is duplicated based on the context. "
            "Keep the suggestions concise and clear and limit the suggestions to just 1 or 2 sentences."
            "Add suggestions for error handling, code optimization, and code readability improvements."
            "\n\nContext of changes: {context}\n\nDetailed changes: {changes}\n\n "
            "Provide suggestions based on the details and context provided above and only make suggestions for the new code changes, and do not suggest changes for code in the context", input_variables=["context", "changes"])


    prompt_with_context = template.invoke({"context": context, "changes": changes})

    print("Generating suggestions using AI...")
    llm = ChatOpenAI(temperature=0.5, api_key=os.getenv('OPENAI_API_KEY'))
    try:
        results = llm.invoke(prompt_with_context)
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