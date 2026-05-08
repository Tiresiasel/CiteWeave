#!/usr/bin/env python3
"""Manual smoke script for intelligent query planning.

This is intentionally not part of default pytest discovery. It requires the
runtime LLM/database configuration used by LangGraphResearchSystem.
"""

from src.agents.multi_agent_research_system import LangGraphResearchSystem


def main() -> None:
    print("🧠 Manual intelligent query planning smoke check")
    print("=" * 60)

    try:
        system = LangGraphResearchSystem()
        print("✅ System initialized successfully")
    except Exception as exc:
        print(f"❌ System initialization failed: {exc}")
        return

    questions = [
        "引用波特的所有文章，他们引用的观点分别是什么",
        "Porter的论文有哪些",
        "什么是竞争战略",
    ]

    for question in questions:
        print(f"\n🔍 Question: {question}")
        try:
            result = system.research_question_details(question)
            print(f"✅ Completed. Keys: {sorted(result.keys())}")
            response = result.get("response") or result.get("answer") or ""
            if response:
                print(f"📝 Response preview: {response[:240]}...")
        except Exception as exc:
            print(f"❌ Error: {exc}")


if __name__ == "__main__":
    main()
