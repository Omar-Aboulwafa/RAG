# test_query_improvements.py - PRODUCTION TEST SCRIPT
from controllers.QueryController import QueryController
import logging
import sys

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_hr_bylaw_query():
    """Test the HR Bylaw maximum deduction query"""
    
    print("\n" + "="*80)
    print("🧪 TESTING ENHANCED QUERY CONTROLLER")
    print("="*80)
    
    # Initialize controller
    try:
        controller = QueryController("rag")
        print("✅ QueryController initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize QueryController: {e}")
        sys.exit(1)
    
    # Test query
    test_query = (
        "According to the HR Bylaws, what is the maximum number of days "
        "an employee's salary can be deducted in a single year as a disciplinary penalty?"
    )
    
    print(f"\n📝 Test Query:\n{test_query}\n")
    
    # Expected answer
    expected_answer = "60 days" or "sixty days"
    print(f"✅ Expected Answer: {expected_answer}\n")
    
    # Run diagnostic
    print("="*80)
    print("🔬 RUNNING DIAGNOSTIC")
    print("="*80)
    
    try:
        diag_result = controller.diagnose_retrieval(test_query, top_k=10)
        print("✅ Diagnostic completed\n")
    except Exception as e:
        print(f"⚠️ Diagnostic failed (non-critical): {e}\n")
    
    # Run production query
    print("="*80)
    print("🚀 RUNNING PRODUCTION QUERY")
    print("="*80)
    
    try:
        result = controller.regulatory_hybrid_query(test_query, top_k=10)
        
        print("\n" + "="*80)
        print("📊 RESULTS")
        print("="*80)
        
        print(f"\n✅ RESPONSE:\n{result['response']}\n")
        
        print("="*80)
        print("📈 METADATA")
        print("="*80)
        print(f"  • Filters applied: {result['filters_applied']}")
        print(f"  • Expanded query: {result.get('expanded_query', 'No expansion')[:100]}...")
        print(f"  • Total sources: {result['total_sources']}")
        print(f"  • Processing time: {result['processing_time']:.2f}s")
        print(f"  • Search type: {result['search_type']}")
        print(f"  • Fusion method: {result['fusion_method']}")
        
        print("\n" + "="*80)
        print("📚 TOP 3 SOURCES")
        print("="*80)
        
        for i, source in enumerate(result['source_nodes'][:3], 1):
            meta = source['metadata']
            print(f"\nSource {i}:")
            print(f"  • Article: {meta.get('primary_article_number', 'N/A')}")
            print(f"  • Document Type: {meta.get('doc_type', 'N/A')}")
            print(f"  • Score: {source['score']:.4f}")
            print(f"  • Rank: {source['rank']}")
            preview = source['content'][:200].replace('\n', ' ')
            print(f"  • Preview: {preview}...")
        
        # Validate answer
        print("\n" + "="*80)
        print("✓ VALIDATION")
        print("="*80)
        
        response_text = result['response'].lower()
        
        checks = {
            "Contains '60' or 'sixty'": ('60' in response_text or 'sixty' in response_text),
            "Contains 'article 110'": 'article 110' in response_text,
            "Contains 'year' or 'annual'": ('year' in response_text or 'annual' in response_text),
            "Contains 'basic salary'": 'basic salary' in response_text,
            "Does NOT incorrectly say '3 days'": 'three days' not in response_text or '60' in response_text,
        }
        
        all_passed = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check}")
            if not passed:
                all_passed = False
        
        print("\n" + "="*80)
        if all_passed:
            print("🎉 ALL VALIDATION CHECKS PASSED!")
            print("The system correctly retrieved Article 110, Clause 3 about 60-day maximum.")
        else:
            print("⚠️ SOME VALIDATION CHECKS FAILED")
            print("Review the response above and check:")
            print("  1. Is Article 110, Clause 3 in your database?")
            print("  2. Run: python test_query_improvements.py to see diagnostic output")
        print("="*80)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Query execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_hr_bylaw_query()
