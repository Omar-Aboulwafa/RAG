import google.generativeai as genai
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyC1PMI35riaAvnoouw3UYuuWc93awzdLjo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# List available models
print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  ✓ {m.name}")

# Test generation
print("\nTesting generation...")
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Say hello")
print(f"Response: {response.text}")

# Test embeddings
print("\nTesting embeddings...")
result = genai.embed_content(
    model="models/embedding-001",
    content="test text"
)
print(f"Embedding dimension: {len(result['embedding'])}")

print("\n✅ All tests passed!")
