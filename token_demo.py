import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

text = "Python is fun to learn! I am learning AI, prompt engineering, and automation with Gemini."

tokens = encoding.encode(text)

print("Original text:", text)
print("Token IDs:", tokens)
print("Number of tokens:", len(tokens))