# test_custom_reranker.py
#!/usr/bin/env python3

def test_custom_reranker():
    """Test the custom BGE reranker"""
    import sys
    sys.path.append('.')
    
    from controllers.QueryController import QueryController
    
    print("🧪 Testing Custom BGE Reranker")
    print("=" * 50)
    
    # Initialize controller
    controller = QueryController(project_id="test_project")
    
    # Check if reranker is initialized
    if controller.reranker:
        print("✅ Custom BGE Reranker initialized successfully")
        print(f"   Model: {controller.reranker.model}")
        print(f"   Base URL: {controller.reranker.base_url}")
        print(f"   Top N: {controller.reranker.top_n}")
    else:
        print("❌ Reranker initialization failed")
        return
    
    # Test query
    query = "What are the maximum salary deduction limits?"
    
    print(f"\n📝 Test Query: {query}")
    print("-" * 50)
    
    try:
        # Execute query with reranking
        result = controller.regulatory_hybrid_query(query, top_k=5)
        
        print(f"✅ Query executed successfully")
        print(f"🔄 Reranked: {result.get('reranked', False)}")
        print(f"📊 Initial candidates: {result.get('initial_candidates', 'N/A')}")
        print(f"🎯 Final results: {result.get('final_results', len(result.get('source_nodes', [])))}")
        print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
        
        # Show top results
        sources = result.get('source_nodes', [])
        if sources:
            print(f"\n📄 Top {len(sources)} Results:")
            for i, source in enumerate(sources[:3], 1):
                score = source.get('score', 0)
                rerank_score = source.get('metadata', {}).get('rerank_score', 'N/A')
                content = source.get('content', '')[:100]
                print(f"\n{i}. Original Score: {score:.4f} | Rerank Score: {rerank_score}")
                print(f"   Content: {content}...")
        
        print("\n" + "=" * 50)
        print("✅ Custom BGE Reranker Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_reranker()
