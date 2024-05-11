import os
import sys
from github import Github
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser

def get_pull_request_diffs(pull_request):
    return [
        {"filename": file.filename, "patch": file.patch, "path": file.filename}
        for file in pull_request.get_files()
    ]


def format_data_for_openai(diffs):
    changes = "\n".join([
        f"File: {file['filename']}\nDiff:\n{file['patch']}\n"
        for file in diffs
    ])
    prompt = (
        f"Analyze the following code changes for potential refactoring opportunities to make the code more readable and efficient, and pointing out areas that could cause potential bugs and performance issues.\n\n"
        "Make a special focus on components and functions that are too big, overly complicated, and can have parts extracted to become more reusable.\n\n"
        "Also, point out any code that is redundant, unnecessary, or can be replaced with more efficient alternatives.\n\n"
        "For each suggestion, provide the line number where the change should be made, the type of change that should be made, and a brief explanation of why the change is necessary.\n\n"
        "The format for each suggestion should be as follows:\n"
        "Line: [beginning line number]\n"
        "Type: [type of change]\n"
        "Explanation: [brief explanation]\n"
        "Code Suggestions: [code snippets to replace the existing code for this specific suggestion]\n\n"
        "If there are multiple suggestions for the same line, separate them with a comma.\n\n"
        "If there are no suggestions for improvement, leave a one comment saying that the code is perfect!:\n"
        f"{changes}"
    )

            
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
    import re

    # Regular expression to find suggestion blocks within the comments
    suggestion_pattern = re.compile(
        r"Line: (\d+)\n"
        r"Type: (.+)\n"
        r"Explanation: (.+)\n"
        r"Code Suggestions: (.+?)(?=\n\n|\Z)", re.DOTALL
    )

    # Find all suggestions within the comments
    suggestions = suggestion_pattern.findall(comments)
    print(f"Suggestions found: {suggestions}")  # Debug print

    for suggestion in suggestions:
        try:
            line, change_type, explanation, code_suggestions = suggestion
            path = 'determined/path/here'  # Assuming you determine or extract path somehow
            position = int(line)  # Simplification for example purposes
            pull_request.create_review_comment(
                body=f"Type: {change_type}\nExplanation: {explanation}\nCode Suggestions: {code_suggestions}",
                commit_id=pull_request.head.sha,
                path=path,
                position=position
            )
        except Exception as e:
            print(f"Failed to post inline comment: {e}")





def main():
    try:
        g = Github(os.getenv('GITHUB_TOKEN'))  # Initialize GitHub API with token
        repo_path = os.getenv('REPO_PATH')
        pr_number = int(os.getenv('PR_NUMBER'))
        
        repo = g.get_repo(repo_path)  # Get the repo object
        pull_request = repo.get_pull(pr_number)  # Get the pull request

        diffs = get_pull_request_diffs(pull_request)  # Get the diffs of the pull request
        prompt = format_data_for_openai(diffs)  # Format data for OpenAI

        suggestions = call_openai(prompt)  # Call OpenAI to get suggestions for code improvement
        print(f"Suggestions: {suggestions}")

        post_comments_to_pull_request(pull_request, suggestions)  # Post suggestions as comments on the PR
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
