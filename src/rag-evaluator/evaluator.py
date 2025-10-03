import logging
import json
import re
from typing import List, Dict, Any
from datetime import datetime
import requests
from datasets import Dataset
from rich.console import Console
from rich.progress import track
import pandas as pd

# RAGAs imports
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

# Import custom Gemini wrapper
from gemini_wrapper import GeminiLLM, GeminiEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import get_settings

console = Console()
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """RAGAs evaluation engine using direct Gemini SDK"""
    
    def __init__(self):
        self.settings = get_settings()
        self.evaluator_llm = self._initialize_evaluator_llm()
        self.evaluator_embeddings = self._initialize_embeddings()
        self.metrics = self._initialize_metrics()
    
    def _initialize_evaluator_llm(self):
        """Initialize Gemini LLM using direct SDK"""
        try:
            gemini_llm = GeminiLLM(
                model_name=self.settings.EVALUATOR_LLM_MODEL,
                api_key=self.settings.GOOGLE_API_KEY
            )
            
            console.print(f"[green]✓[/green] Initialized Gemini: {self.settings.EVALUATOR_LLM_MODEL}")
            return gemini_llm
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize Gemini: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize Gemini embeddings using direct SDK"""
        try:
            embeddings = GeminiEmbeddings(
                model_name=self.settings.EMBEDDING_MODEL,
                api_key=self.settings.GOOGLE_API_KEY
            )
            
            # Wrap for RAGAs compatibility
            wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
            
            console.print(f"[green]✓[/green] Initialized embeddings: {self.settings.EMBEDDING_MODEL}")
            return wrapped_embeddings
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_metrics(self):
        """Initialize RAGAs metrics"""
        try:
            metrics = []
            
            if "faithfulness" in self.settings.METRICS_TO_EVALUATE:
                metrics.append(Faithfulness(llm=self.evaluator_llm))
                
            if "answer_relevancy" in self.settings.METRICS_TO_EVALUATE:
                metrics.append(AnswerRelevancy(
                    llm=self.evaluator_llm,
                    embeddings=self.evaluator_embeddings
                ))
                
            if "context_precision" in self.settings.METRICS_TO_EVALUATE:
                metrics.append(ContextPrecision(llm=self.evaluator_llm))
                
            if "context_recall" in self.settings.METRICS_TO_EVALUATE:
                metrics.append(ContextRecall(llm=self.evaluator_llm))
            
            console.print(f"[green]✓[/green] Initialized {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize metrics: {e}")
            raise
    
    # Keep all other methods the same (query_rag_system, load_jsonl_dataset, etc.)
    # ... [rest of the code from previous evaluator.py]
    
    def query_rag_system(self, question: str) -> Dict[str, Any]:
        """Query the RAG backend system and extract response + contexts"""
        try:
            url = f"{self.settings.BACKEND_API_URL}{self.settings.API_CHAT_ENDPOINT}"
            payload = {
                "model": "regulatory-rag",
                "messages": [
                    {"role": "user", "content": question}
                ],
                "stream": False
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.settings.TIMEOUT_SECONDS
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data['choices'][0]['message']['content']
                
                # Extract contexts from response
                contexts = self._extract_contexts_from_response(answer)
                
                # Ensure at least one context exists
                if not contexts or len(contexts) == 0:
                    logger.warning(f"No contexts extracted, using answer as context")
                    contexts = [answer]
                
                return {
                    "answer": answer,
                    "contexts": contexts
                }
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {
                    "answer": f"Error: {response.status_code}",
                    "contexts": ["Error occurred"]
                }
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                "answer": f"Exception: {str(e)}",
                "contexts": ["Exception occurred"]
            }
    
    def _extract_contexts_from_response(self, answer: str) -> List[str]:
        """
        Robustly extract context chunks from RAG response
        
        Strategy:
        1. Try to parse as JSON with 'contexts' field
        2. Fallback to regex-based extraction
        3. Always return at least one context (non-empty for Gemini)
        """
        # Try parsing as JSON first
        try:
            resp = json.loads(answer)
            if "contexts" in resp and isinstance(resp["contexts"], list):
                contexts = [c for c in resp["contexts"] if isinstance(c, str) and len(c) > 50]
                if contexts:
                    logger.info(f"Extracted {len(contexts)} contexts from JSON")
                    return contexts
        except json.JSONDecodeError:
            pass  # Not JSON, continue to regex
        
        # Fallback: Regex-based extraction for formatted text
        contexts = []
        doc_markers = re.split(r'\*\*([^:]+):\*\*', answer)
        
        for i in range(1, len(doc_markers), 2):
            if i + 1 < len(doc_markers):
                content = doc_markers[i + 1].strip()
                # Remove session info footer
                content = re.sub(r'\n\*📊 Session:.*\*', '', content)
                content = re.sub(r'Session.*', '', content)
                
                if content and len(content) > 50:
                    contexts.append(content)
        
        # Final fallback: Split answer into meaningful chunks
        if not contexts:
            # Clean answer of metadata
            clean_answer = re.sub(r'\n\*📊 Session:.*\*', '', answer)
            clean_answer = re.sub(r'Session.*', '', clean_answer)
            
            # Gemini requires non-empty strings for embeddings
            if not clean_answer or len(clean_answer.strip()) == 0:
                clean_answer = "No context available"
            
            # Split by sentences if answer is long
            if len(clean_answer) > 200:
                sentences = [s.strip() + '.' for s in clean_answer.split('.') if len(s.strip()) > 30]
                if sentences:
                    contexts = sentences[:5]  # Max 5 sentences as contexts
                else:
                    contexts = [clean_answer]
            else:
                contexts = [clean_answer]
        
        logger.info(f"Extracted {len(contexts)} contexts")
        return contexts
    
    def load_jsonl_dataset(self, file_path: str) -> List[Dict]:
        """Load JSONL dataset"""
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            question_data = json.loads(line)
                            
                            # Add default metadata
                            if 'id' not in question_data:
                                question_data['id'] = f"hr_bylaw_{i:03d}"
                            if 'doc_type' not in question_data:
                                question_data['doc_type'] = "HR Bylaw"
                            
                            questions.append(question_data)
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Line {i}: Invalid JSON - {e}")
                            continue
            
            console.print(f"[green]✓[/green] Loaded {len(questions)} questions from JSONL")
            return questions
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load JSONL: {e}")
            raise
    
    def prepare_evaluation_dataset(self, test_questions: List[Dict]) -> Dataset:
        """Prepare dataset for RAGAs evaluation"""
        console.print("\n[yellow]Querying RAG system for all test questions...[/yellow]")
        
        evaluation_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for item in track(test_questions, description="Processing questions"):
            question = item['question']
            ground_truth = item['ground_truth']
            
            # Query the RAG system
            rag_response = self.query_rag_system(question)
            
            evaluation_data["question"].append(question)
            evaluation_data["answer"].append(rag_response["answer"])
            evaluation_data["contexts"].append(rag_response["contexts"])
            evaluation_data["ground_truth"].append(ground_truth)
            
            # Log first sample for debugging
            if len(evaluation_data["question"]) == 1:
                console.print(f"\n[dim]Sample question: {question[:80]}...[/dim]")
                console.print(f"[dim]Sample answer: {rag_response['answer'][:100]}...[/dim]")
                console.print(f"[dim]Contexts extracted: {len(rag_response['contexts'])}[/dim]")
        
        dataset = Dataset.from_dict(evaluation_data)
        console.print(f"[green]✓[/green] Prepared evaluation dataset with {len(dataset)} samples")
        
        return dataset
    
    def run_evaluation(self, dataset: Dataset) -> Any:
        """Run RAGAs evaluation with Gemini rate limit handling"""
        console.print("\n[yellow]Running RAGAs evaluation with Gemini...[/yellow]")
        console.print(f"[dim]Processing {len(dataset)} samples with rate limit handling...[/dim]")
        
        try:
            from ragas import RunConfig
            
            # Configure for free tier: 15 RPM = 1 request per 4 seconds
            run_config = RunConfig(
                max_workers=1,  # Sequential processing
                max_wait=60,  # Wait up to 60 seconds on rate limits
                max_retries=5,  # Retry up to 5 times
                timeout=120  # 2 min timeout per request
            )
            
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                run_config=run_config,
                raise_exceptions=False  # Continue on errors
            )
            
            console.print("[green]✓[/green] Evaluation completed")
            return result
            
        except Exception as e:
            console.print(f"[red]✗[/red] Evaluation failed: {e}")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise

    
    def generate_evaluation_report(self, result: Any, test_questions: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        df = result.to_pandas()
        
        report = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(test_questions),
                "evaluator_llm": self.settings.EVALUATOR_LLM_MODEL,
                "embedding_model": self.settings.EMBEDDING_MODEL,
                "metrics_evaluated": [m.__class__.__name__.lower() for m in self.metrics],
                "dataset_format": "JSONL",
                "dataset_file": self.settings.TEST_DATASET_PATH
            },
            "overall_scores": {},
            "per_question_results": [],
            "document_type_analysis": {},
            "recommendations": []
        }
        
        # Calculate overall scores
        for metric in self.metrics:
            metric_name = metric.__class__.__name__.lower()
            if metric_name in df.columns:
                score = df[metric_name].mean()
                report["overall_scores"][metric_name] = round(float(score), 4)
        
        # Per-question results
        for idx, row in df.iterrows():
            question_result = {
                "id": test_questions[idx].get('id', f"q_{idx}"),
                "question": test_questions[idx]['question'],
                "ground_truth": test_questions[idx]['ground_truth'],
                "doc_type": test_questions[idx].get('doc_type', 'HR Bylaw'),
                "scores": {}
            }
            
            for metric in self.metrics:
                metric_name = metric.__class__.__name__.lower()
                if metric_name in df.columns:
                    score_value = row[metric_name]
                    if pd.isna(score_value):
                        question_result["scores"][metric_name] = None
                    else:
                        question_result["scores"][metric_name] = round(float(score_value), 4)
            
            report["per_question_results"].append(question_result)
        
        # Document type analysis
        doc_types = set([q.get('doc_type', 'HR Bylaw') for q in test_questions])
        
        for doc_type in doc_types:
            doc_type_indices = [i for i, q in enumerate(test_questions) 
                              if q.get('doc_type', 'HR Bylaw') == doc_type]
            if doc_type_indices:
                doc_type_df = df.iloc[doc_type_indices]
                
                report["document_type_analysis"][doc_type] = {}
                for metric in self.metrics:
                    metric_name = metric.__class__.__name__.lower()
                    if metric_name in df.columns:
                        score = doc_type_df[metric_name].mean()
                        if pd.isna(score):
                            report["document_type_analysis"][doc_type][metric_name] = None
                        else:
                            report["document_type_analysis"][doc_type][metric_name] = round(float(score), 4)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["overall_scores"])
        
        return report
    
    def _generate_recommendations(self, overall_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on scores"""
        recommendations = []
        
        faithfulness = overall_scores.get('faithfulness')
        if faithfulness is not None and not pd.isna(faithfulness):
            if faithfulness < 0.7:
                recommendations.append(
                    "🔴 LOW FAITHFULNESS (<0.7): Responses contain factual inconsistencies. "
                    "Review Article citations and improve context retrieval accuracy."
                )
            elif faithfulness < 0.85:
                recommendations.append(
                    "🟡 MEDIUM FAITHFULNESS (0.7-0.85): Good accuracy but room for improvement."
                )
        
        answer_rel = overall_scores.get('answerrelevancy')
        if answer_rel is not None and not pd.isna(answer_rel) and answer_rel < 0.7:
            recommendations.append(
                "🔴 LOW ANSWER RELEVANCY (<0.7): Responses not directly addressing questions."
            )
        
        context_prec = overall_scores.get('contextprecision')
        if context_prec is not None and not pd.isna(context_prec) and context_prec < 0.6:
            recommendations.append(
                "🔴 LOW CONTEXT PRECISION (<0.6): Retrieving too many irrelevant documents."
            )
        
        context_rec = overall_scores.get('contextrecall')
        if context_rec is not None and not pd.isna(context_rec) and context_rec < 0.6:
            recommendations.append(
                "🔴 LOW CONTEXT RECALL (<0.6): Missing relevant content."
            )
        
        # Positive feedback
        valid_scores = [s for s in overall_scores.values() if s is not None and not pd.isna(s)]
        if valid_scores:
            all_high = all(score >= 0.85 for score in valid_scores)
            all_good = all(score >= 0.7 for score in valid_scores)
            
            if all_high:
                recommendations.append(
                    "✅ EXCELLENT PERFORMANCE: All metrics above 0.85. System ready for production!"
                )
            elif all_good:
                recommendations.append(
                    "✅ GOOD PERFORMANCE: All metrics above 0.7. System functional."
                )
        
        if not recommendations:
            recommendations.append(
                "⚠️ Some metrics returned NaN. Check LLM responses and context extraction."
            )
        
        return recommendations
