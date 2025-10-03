"""
RAGAs Evaluation Application - Main Entry Point
Console application for evaluating RAG system with RAGAs framework
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import get_settings
from evaluator import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


def save_results(report: dict, results_dir: str):
    """Save evaluation results"""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    json_path = Path(results_dir) / f"evaluation_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    console.print(f"[green]✓[/green] Saved JSON report: {json_path}")
    
    # Save Markdown report
    md_path = Path(results_dir) / f"evaluation_report_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write(generate_markdown_report(report))
    console.print(f"[green]✓[/green] Saved Markdown report: {md_path}")
    
    # Save CSV
    csv_path = Path(results_dir) / f"per_question_results_{timestamp}.csv"
    df = pd.DataFrame(report['per_question_results'])
    df.to_csv(csv_path, index=False)
    console.print(f"[green]✓[/green] Saved CSV results: {csv_path}")


def generate_markdown_report(report: dict) -> str:
    """Generate Markdown report"""
    md = "# RAGAs Evaluation Report\n\n"
    md += f"**Generated:** {report['evaluation_metadata']['timestamp']}\n\n"
    md += f"**Total Questions:** {report['evaluation_metadata']['total_questions']}\n\n"
    
    md += "## Overall Scores\n\n"
    md += "| Metric | Score |\n"
    md += "|--------|-------|\n"
    for metric, score in report['overall_scores'].items():
        md += f"| {metric.replace('_', ' ').title()} | {score:.4f} |\n"
    
    md += "\n## Recommendations\n\n"
    for i, rec in enumerate(report['recommendations'], 1):
        md += f"{i}. {rec}\n\n"
    
    return md


def display_results_summary(report: dict):
    """Display results summary"""
    scores_table = Table(title="Overall Evaluation Scores", box=box.ROUNDED)
    scores_table.add_column("Metric", style="cyan", no_wrap=True)
    scores_table.add_column("Score", justify="right", style="green")
    scores_table.add_column("Status", justify="center")
    
    for metric, score in report['overall_scores'].items():
        status = "✓ GOOD" if score >= 0.7 else "⚠ NEEDS IMPROVEMENT"
        status_style = "green" if score >= 0.7 else "yellow"
        scores_table.add_row(
            metric.replace('_', ' ').title(),
            f"{score:.4f}",
            f"[{status_style}]{status}[/{status_style}]"
        )
    
    console.print(scores_table)
    
    # Recommendations
    if report['recommendations']:
        rec_text = "\n".join([f"{i}. {rec}" for i, rec in enumerate(report['recommendations'], 1)])
        console.print(Panel(rec_text, title="Recommendations", border_style="yellow"))


def main():
    """Main evaluation workflow"""
    console.print(Panel.fit(
        "[bold cyan]RAGAs Evaluation Application[/bold cyan]\n"
        "Evaluating HR Bylaw RAG System with RAGAs Framework",
        border_style="cyan"
    ))
    
    settings = get_settings()
    
    # Step 1: Load JSONL dataset
    console.print("\n[bold]Step 1: Loading JSONL Test Dataset[/bold]")
    evaluator = RAGEvaluator()
    test_questions = evaluator.load_jsonl_dataset(settings.TEST_DATASET_PATH)
    
    # Step 2: Prepare evaluation dataset
    console.print("\n[bold]Step 2: Querying RAG System[/bold]")
    dataset = evaluator.prepare_evaluation_dataset(test_questions)
    
    # Step 3: Run evaluation
    console.print("\n[bold]Step 3: Running RAGAs Evaluation[/bold]")
    result = evaluator.run_evaluation(dataset)
    
    # Step 4: Generate report
    console.print("\n[bold]Step 4: Generating Report[/bold]")
    report = evaluator.generate_evaluation_report(result, test_questions)
    
    # Step 5: Save results
    console.print("\n[bold]Step 5: Saving Results[/bold]")
    save_results(report, settings.RESULTS_DIR)
    
    # Step 6: Display summary
    console.print("\n[bold]Step 6: Evaluation Summary[/bold]")
    display_results_summary(report)
    
    console.print("\n[bold green]✓ Evaluation completed successfully![/bold green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]✗ Evaluation failed: {e}[/bold red]")
        logger.exception("Fatal error during evaluation")
        sys.exit(1)

