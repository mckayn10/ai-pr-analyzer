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
import openai

# Pinecone setup
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name="ai-code-analyzer")
model="text-embedding-3-large"

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

print("Model loaded successfully")
def generate_embedding(text, model="text-embedding-3-large"):
    """
    Generate an embedding for the given text using the specified model.

    Args:
    text (str): The input text to generate an embedding for.
    model (str): The model to use for generating the embedding. Default is 'text-embedding-3-large'.

    Returns:
    np.array: A NumPy array of the embedding.
    """

    print(f"Generating embedding for text: {text}")

    try:
        response = client.embeddings.create(
            input=[text],
            model=model  # Choose "text-embedding-3-small" or "text-embedding-3-large"
        )
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
                        raise ValueError("Invalid embedding data type or embedding generation failed.")
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

    formatted_text = '\n'.join([f"File: {diff['filename']}\nDiff:\n{diff['patch']}" for diff in diffs])
    print(f"Formatted text: {formatted_text}")

    try:
        print("Retrievin‘g context...")
        context_response = retriever.invoke(formatted_text)
        print(f"Context response: {context_response}")
        # Properly check and extract metadata if available
        if context_response and context_response.matches:
            context = ' '.join([match.metadata.get('additional_info', 'No metadata found') for match in context_response.matches])
            print(f"Context retrieved successfully: {context[:100]}")
        else:
            print("No context found")
            context = "No relevant context available."
    except Exception as e:
        print(f"An unexpected error occurred while retrieving context: {e}")
        return None

    changes = "\n".join([f"File: {file['filename']}\nDiff:\n{file['patch']}" for file in diffs])
    prompt = (
        "Analyze the following code changes for potential refactoring opportunities to make the code more readable and efficient, "
        "and point out areas that could cause potential bugs and performance issues.\n\n"
        "Also check the context and make sure the new code changes are not conflicting with the existing code and does not create duplicate code.\n\n"
        "Each suggestion should be concise and limitied to a few short sentences of explanation followed by a code snippet if applicable. Make sure to reference the file where the suggestion should be made\n\n"
        "Context of changes:\n" + str(context) +  # Convert context to string before concatenating
        "\n\nDetailed changes:\n" + changes + "\n\n"
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
