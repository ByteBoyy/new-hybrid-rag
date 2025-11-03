import requests
import json
from typing import Dict


def test_query(query: str, use_fanout: bool = None):
    """Test a query with the API."""
    print("\n" + "="*80)
    print(f"Query: {query}")
    print(f"Fan-out: {use_fanout if use_fanout is not None else 'auto-detect'}")
    print("="*80)
    
    try:
        response = requests.post(
            "http://localhost:8000/query/ollama",
            json={
                "query": query,
                "mode": "ultra",
                "top_k": 5,
                "use_fanout": use_fanout
            },
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return
        
        result = response.json()
        
        # Print fan-out info
        fanout_info = result['metadata'].get('fanout_info', {})
        if fanout_info.get('used_fanout'):
            print(f"\n🎯 Fan-out queries generated:")
            for i, fo in enumerate(fanout_info['fanout_queries'], 1):
                print(f"   {i}. [{fo['purpose']}] {fo['query']}")
            print(f"\n📊 Consensus articles: {fanout_info.get('consensus_articles', 0)}")
        else:
            print(f"\nℹ️  No fan-out used (direct retrieval)")
        
        # Print top results
        print(f"\n📚 Top {len(result['retrieved'])} results:")
        for i, doc in enumerate(result['retrieved'], 1):
            appearances = doc.get('appearances', 1)
            from_original = doc.get('from_original', False)
            purposes = doc.get('purposes', [])
            
            status = "⭐" if from_original else ""
            consensus = f" ({appearances}x)" if appearances > 1 else ""
            purpose_str = f" [{', '.join(purposes[:2])}]" if purposes else ""
            
            print(f"   {i}. {status} {doc['title']}{consensus}{purpose_str}")
            print(f"      Score: {doc['score']:.3f}")
        
        # Print answer
        print(f"\n💬 Answer ({result['confidence']} confidence):")
        answer = result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer']
        print(f"   {answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run test queries."""
    print("="*80)
    print("Fan-Out Retrieval Testing")
    print("="*80)
    print("\nMake sure:")
    print("1. API is running: python -m uvicorn src.api.app_fanout:app --reload")
    print("2. Ollama is running: ollama serve")
    print("3. Model is available: ollama pull qwen2.5:0.5b")
    
    input("\nPress Enter to start tests...")
    
    # Test 1: Complex how-to query (should use fan-out)
    test_query("how do i make caramel at home")
    
    # Test 2: Troubleshooting query (should use fan-out)
    test_query("fix car overheating problem")
    
    # Test 3: Simple factual query (should NOT use fan-out)
    test_query("what is photosynthesis")
    
    # Test 4: Multi-aspect query (should use fan-out)
    test_query("learn to build a website with HTML CSS and JavaScript")
    
    # Test 5: Complex technical query (should use fan-out)
    test_query("understand machine learning neural networks and training")
    
    print("\n" + "="*80)
    print("Tests complete!")
    print("="*80)


if __name__ == "__main__":
    main()