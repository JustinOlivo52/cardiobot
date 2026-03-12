"""
CardioBot Evaluation Script
Tests RAG retrieval quality and answer accuracy
"""
from agents.claude_agent import ask_cardiobot
from agents.rag_agent import retrieve_context

# Test questions with expected keywords in answers
TEST_CASES = [
    {
        "question": "What antiplatelet therapy is recommended for STEMI?",
        "expected_keywords": ["aspirin", "p2y12", "prasugrel", "ticagrelor"],
        "category": "Antiplatelet"
    },
    {
        "question": "What is the recommended door-to-balloon time for STEMI?",
        "expected_keywords": ["60", "90", "minutes", "primary pci"],
        "category": "STEMI Timelines"
    },
    {
        "question": "When should high-sensitivity troponin be measured?",
        "expected_keywords": ["troponin", "0h", "1h", "2h", "3h", "hours"],
        "category": "Biomarkers"
    },
    {
        "question": "What oxygen saturation threshold requires supplemental oxygen in ACS?",
        "expected_keywords": ["90", "sao2", "oxygen", "hypoxemia"],
        "category": "Supportive Care"
    },
    {
        "question": "What is the recommended heparin dose for STEMI PCI?",
        "expected_keywords": ["heparin", "units", "kg", "weight"],
        "category": "Anticoagulation"
    }
]

def run_evaluation():
    print("=" * 60)
    print("🫀 CardioBot Evaluation")
    print("=" * 60)

    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {test['category']}")
        print(f"Q: {test['question']}")

        try:
            # Test retrieval
            context, hits = retrieve_context(test["question"])
            retrieval_score = hits[0]["score"] if hits else 0

            # Test answer
            result = ask_cardiobot(test["question"])
            answer = result["answer"].lower()

            # Check keywords
            keywords_found = [k for k in test["expected_keywords"] if k.lower() in answer]
            keyword_score = len(keywords_found) / len(test["expected_keywords"])

            status = "✅ PASS" if keyword_score >= 0.5 else "⚠️ PARTIAL" if keyword_score > 0 else "❌ FAIL"

            print(f"Status: {status}")
            print(f"Retrieval score: {retrieval_score:.3f}")
            print(f"Keywords found: {keywords_found} ({keyword_score:.0%})")

            results.append({
                "category": test["category"],
                "status": status,
                "retrieval_score": retrieval_score,
                "keyword_score": keyword_score
            })

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "category": test["category"],
                "status": "❌ ERROR",
                "retrieval_score": 0,
                "keyword_score": 0
            })

    # Summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if "PASS" in r["status"])
    avg_retrieval = sum(r["retrieval_score"] for r in results) / len(results)
    avg_keywords = sum(r["keyword_score"] for r in results) / len(results)

    print(f"Tests passed:        {passed}/{len(results)}")
    print(f"Avg retrieval score: {avg_retrieval:.3f}")
    print(f"Avg keyword match:   {avg_keywords:.0%}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
