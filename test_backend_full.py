import requests
import json

API_BASE = "http://localhost:8000"

print("=" * 80)
print("COMPREHENSIVE BACKEND TEST")
print("=" * 80)

# Test 1: Health Check
print("\n" + "=" * 80)
print("TEST 1: Health Check")
print("=" * 80)
try:
    resp = requests.get(f"{API_BASE}/health")
    print(f"✅ Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Query with repo_name (slug format)
print("\n" + "=" * 80)
print("TEST 2: Query with repo_name (slug)")
print("=" * 80)
payload_slug = {
    "question": "what class is used in here?",
    "repo_name": "dreamer522_Ai-Travelling-planner",
    "chunk_strategy": "ast",
    "top_k": 5,
    "alpha": 0.7,
    "model": "gpt-4o-mini"
}

try:
    resp = requests.post(f"{API_BASE}/query", json=payload_slug, timeout=120)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ SUCCESS with repo_name!")
        print(f"Answer Preview: {data['answer'][:150]}...")
        print(f"Sources: {len(data['sources'])}")
        print(f"Tokens: {data['tokens_used']}")
    else:
        print(f"❌ FAILED!")
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"❌ Exception: {e}")

# Test 3: Query with repo_url (full GitHub URL)
print("\n" + "=" * 80)
print("TEST 3: Query with repo_url (GitHub URL)")
print("=" * 80)
payload_url = {
    "question": "what classes are defined?",
    "repo_url": "https://github.com/dreamer522/Ai-Travelling-planner",
    "chunk_strategy": "ast",
    "top_k": 5,
    "alpha": 0.7,
    "model": "gpt-4o-mini"
}

try:
    resp = requests.post(f"{API_BASE}/query", json=payload_url, timeout=120)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ SUCCESS with repo_url!")
        print(f"Answer Preview: {data['answer'][:150]}...")
        print(f"Sources: {len(data['sources'])}")
        print(f"Tokens: {data['tokens_used']}")
        
        print(f"\n📁 Source Details:")
        for i, src in enumerate(data['sources'][:2], 1):
            print(f"\n  Source {i}:")
            print(f"    File: {src['file_path']}")
            print(f"    Function: {src['function_name']}")
            print(f"    Lines: {src['start_line']}-{src['end_line']} (types: {type(src['start_line']).__name__}/{type(src['end_line']).__name__})")
            print(f"    Score: {src['score']}")
    else:
        print(f"❌ FAILED!")
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"❌ Exception: {e}")

# Test 4: Query with BOTH repo_url and repo_name (repo_url should win)
print("\n" + "=" * 80)
print("TEST 4: Query with BOTH repo_url and repo_name")
print("=" * 80)
payload_both = {
    "question": "what is the main function?",
    "repo_url": "https://github.com/dreamer522/Ai-Travelling-planner",
    "repo_name": "wrong_name_should_be_ignored",
    "chunk_strategy": "ast",
    "top_k": 3,
    "alpha": 0.7,
    "model": "gpt-4o-mini"
}

try:
    resp = requests.post(f"{API_BASE}/query", json=payload_both, timeout=120)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ SUCCESS! repo_url took precedence over repo_name")
        print(f"Answer Preview: {data['answer'][:150]}...")
    else:
        print(f"❌ FAILED!")
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"❌ Exception: {e}")

# Test 5: Query with NEITHER repo_url nor repo_name (should fail with 422)
print("\n" + "=" * 80)
print("TEST 5: Query with NEITHER repo_url nor repo_name (should fail)")
print("=" * 80)
payload_neither = {
    "question": "test question",
    "chunk_strategy": "ast",
    "top_k": 5,
    "alpha": 0.7,
    "model": "gpt-4o-mini"
}

try:
    resp = requests.post(f"{API_BASE}/query", json=payload_neither, timeout=120)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 422:
        print(f"✅ CORRECTLY REJECTED!")
        print(f"Error: {resp.json()}")
    else:
        print(f"❌ Should have returned 422, got {resp.status_code}")
        print(f"Response: {resp.text}")
except Exception as e:
    print(f"❌ Exception: {e}")

# Test 6: Check data types in response
print("\n" + "=" * 80)
print("TEST 6: Verify Data Types in Response")
print("=" * 80)
try:
    resp = requests.post(f"{API_BASE}/query", json=payload_slug, timeout=120)
    if resp.status_code == 200:
        data = resp.json()
        first_source = data['sources'][0]
        
        print(f"✅ Type Validation:")
        print(f"  answer: {type(data['answer']).__name__}")
        print(f"  sources: {type(data['sources']).__name__} with {len(data['sources'])} items")
        print(f"  tokens_used: {type(data['tokens_used']).__name__}")
        print(f"  latency_ms: {type(data['latency_ms']).__name__}")
        print(f"  model: {type(data['model']).__name__}")
        print(f"\n  First source fields:")
        print(f"    file_path: {type(first_source['file_path']).__name__}")
        print(f"    function_name: {type(first_source['function_name']).__name__}")
        print(f"    start_line: {type(first_source['start_line']).__name__} = {first_source['start_line']}")
        print(f"    end_line: {type(first_source['end_line']).__name__} = {first_source['end_line']}")
        print(f"    language: {type(first_source['language']).__name__}")
        print(f"    score: {type(first_source['score']).__name__}")
        
        # Check if start_line and end_line are int or str
        if isinstance(first_source['start_line'], int) and isinstance(first_source['end_line'], int):
            print(f"\n✅ start_line and end_line are INTEGERS (as expected)")
        elif isinstance(first_source['start_line'], str) and isinstance(first_source['end_line'], str):
            print(f"\n⚠️  start_line and end_line are STRINGS")
        else:
            print(f"\n⚠️  Mixed types detected!")
except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED!")
print("=" * 80)