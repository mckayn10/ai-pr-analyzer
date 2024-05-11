import os
from github import Github
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser

def get_pull_request_diffs(pull_request):
    return [
        {"filename": file.filename, "patch": file.patch}
        for file in pull_request.get_files()
    ]

def format_data_for_openai(diffs):
    changes = "\n".join([
        f"File: {file['filename']}\nDiff:\n{file['patch']}\n"
        for file in diffs
    ])
    prompt = f"Analyze the following code changes for potential refactoring opportunities to make the code more readable and efficient, and pointing out areas that could cause potential bugs and performance issues. If there are no suggestions for improvement, leave a one comment saying that the code is perfect! :\n{changes}"

            
    return prompt

def call_openai(prompt):
    client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-3.5-turbo-0125")
    try:
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
    # Initialize GitHub API with token
    g = Github(os.getenv('MY_GITHUB_TOKEN'))

    # Get the repo path and PR number from the environment variables
    repo_path = os.getenv('REPO_PATH')
    pr_number = int(os.getenv('PR_NUMBER'))
    
    # Get the repo object and pull request
    repo = g.get_repo(repo_path)
    pull_request = repo.get_pull(pr_number)

    # Get the diffs of the pull request
    diffs = get_pull_request_diffs(pull_request)

    # Format data for OpenAI
    prompt = format_data_for_openai(diffs)

    # Call OpenAI to get suggestions for code improvement
    suggestions = call_openai(prompt)
    print(f"Suggestions: {suggestions}")

    # Post suggestions as comments on the PR
    post_comments_to_pull_request(pull_request, suggestions)

if __name__ == '__main__':
    main()

