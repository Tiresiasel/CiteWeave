#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.multi_agent_research_system import LangGraphResearchSystem

TEST_TITLES = [
    "Network Defense: Pruning, Grafting, and Closing to Prevent Leakage of Strategic Knowledge to Rivals",
    "Stakeholder Relationships and Social Welfare: A Behavioral Theory of Contributions to Joint Value Creation",
    "Using the SHAPLEY value approach to variance decomposition in strategy research: Diversification, internationalization, and corporate group effects on affiliate profitability",
]

FAIL_PHRASES = [
    "no detailed content",
    "not provided in the sampled content",
    "no specific content",
    "not detailed in the available content",
]


def run_test():
    system = LangGraphResearchSystem(config_path='config')
    results = []

    for title in TEST_TITLES:
        question = f'What is the main argument of the paper titled "{title}"?'
        step1 = system.interactive_research_chat(question)
        step2 = system.interactive_research_chat(
            question,
            menu_choice='1',
            collected_data=step1['collected_data'],
        )
        answer = step2['text']
        lowered = answer.lower()
        passed = (
            len(answer) > 500
            and all(phrase not in lowered for phrase in FAIL_PHRASES)
            and ("argument" in lowered or "authors argue" in lowered or "the paper" in lowered)
        )
        results.append({
            'title': title,
            'passed': passed,
            'answer_preview': answer[:1500],
        })

    report = {
        'passed': all(r['passed'] for r in results),
        'results': results,
    }
    report_path = Path('tests/_artifacts/recent_paper_argument_smoke.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    run_test()
