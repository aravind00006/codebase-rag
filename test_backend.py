import requests
import json

API_BASE = "http://localhost:8000"

# Test 1: Health Check
print("=" * 60)
print("TEST 1: Health Check")
print("=" * 60)
try:
    resp = requests.get(f"{API_BASE}/health")
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\n")

# Test 2: Query
print("=" * 60)
print("TEST 2: Query Endpoint")
print("=" * 60)
payload = {
    "question": "what class is used in here?",
    "repo_name": "dreamer522_Ai-Travelling-planner",
    "chunk_strategy": "ast",
    "top_k": 5,
    "alpha": 0.7,
    "model": "gpt-4o-mini"
}

try:
    resp = requests.post(
        f"{API_BASE}/query",
        json=payload,
        timeout=120
    )
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"\n✅ SUCCESS!")
        print(f"Answer Preview: {data['answer'][:200]}...")
        print(f"Number of Sources: {len(data['sources'])}")
        print(f"Tokens Used: {data['tokens_used']}")
        print(f"Latency: {data['latency_ms']} ms")
        print(f"Model: {data['model']}")
        
        print(f"\n📁 First Source Details:")
        first_source = data['sources'][0]
        print(f"  File: {first_source['file_path']}")
        print(f"  Function: {first_source['function_name']}")
        print(f"  Lines: {first_source['start_line']} - {first_source['end_line']}")
        print(f"  Line Types: start={type(first_source['start_line']).__name__}, end={type(first_source['end_line']).__name__}")
        print(f"  Score: {first_source['score']}")
        print(f"  Preview: {first_source['chunk_preview'][:100]}...")
        
    else:
        print(f"\n❌ ERROR!")
        print(f"Response Text: {resp.text}")
        try:
            print(f"Error Detail: {resp.json()}")
        except:
            print("(Could not parse JSON)")
            
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()