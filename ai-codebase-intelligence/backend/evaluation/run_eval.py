"""Week 15 evaluation entrypoint script.

Usage:
  python evaluation/run_eval.py
  python evaluation/run_eval.py --repos psf/requests --questions 5 --skip-load-test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from core.config import get_settings
from evaluation.cost_tracker import CostTracker
from evaluation.eval_framework import EvalConfig, EvaluationRunner
from evaluation.latency_benchmark import LatencyBenchmark
from evaluation.quality_scorer import QualityScorer
from evaluation.report_generator import ReportGenerator, ReportPaths
from evaluation.repos import EVAL_REPOS, get_repo_by_name


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 15 benchmark pipeline.")
    parser.add_argument("--repos", type=str, default="", help="Comma-separated repo names.")
    parser.add_argument("--questions", type=int, default=20, help="Questions per repo.")
    parser.add_argument("--skip-load-test", action="store_true", help="Skip Locust load test stage.")
    parser.add_argument("--skip-quality-scoring", action="store_true", help="Skip LLM quality scoring stage.")
    parser.add_argument("--budget-usd", type=float, default=10.0, help="Max evaluation budget in USD.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_project_root() / "data" / "benchmarks"),
        help="Benchmark output directory.",
    )
    parser.add_argument("--base-url", type=str, default="https://localhost", help="Target API base URL.")
    parser.add_argument("--latency-samples", type=int, default=50, help="Samples per endpoint.")
    parser.add_argument("--report-only", action="store_true", help="Generate benchmark report from existing JSON artifacts.")
    return parser.parse_args()


def _resolve_repo_names(raw_repos: str) -> list[str]:
    if not raw_repos.strip():
        return [repo.name for repo in EVAL_REPOS]

    requested = [item.strip() for item in raw_repos.split(",") if item.strip()]
    valid: list[str] = []
    for name in requested:
        if get_repo_by_name(name):
            valid.append(name)
    return valid


def _run_parallel_ingestion(runner: EvaluationRunner, repo_names: list[str]) -> dict[str, dict]:
    repos = [get_repo_by_name(name) for name in repo_names]
    repos = [repo for repo in repos if repo is not None]

    output: dict[str, dict] = {}
    if not repos:
        return output

    with ThreadPoolExecutor(max_workers=min(5, len(repos))) as executor:
        futures = {executor.submit(runner.ensure_repo_ingested, repo): repo for repo in repos}
        with tqdm(total=len(futures), desc="Ingesting repositories") as progress:
            for future in as_completed(futures):
                repo = futures[future]
                repo_id, task_id, ingest_seconds, error = future.result()
                output[repo.name] = {
                    "repo_id": repo_id,
                    "task_id": task_id,
                    "ingestion_time_seconds": ingest_seconds,
                    "error": error,
                }
                progress.update(1)
    return output


def _run_quality_scoring(output_dir: Path, judge_model: str, api_key: str | None) -> Path:
    eval_summary_files = sorted(output_dir.glob("results/eval_summary_*.json"))
    if not eval_summary_files:
        raise RuntimeError("No evaluation summary found for quality scoring.")

    latest_summary = eval_summary_files[-1]
    payload = json.loads(latest_summary.read_text(encoding="utf-8"))

    scorer = QualityScorer(judge_model=judge_model, openrouter_api_key=api_key)
    items: list[dict] = []

    for repo in payload.get("repos", []):
        repo_name = repo.get("repo_name", "unknown")
        for question in repo.get("questions", []):
            if question.get("error"):
                continue
            context_strings = []
            for source in question.get("sources", []):
                file_path = source.get("file_path") or source.get("file") or "unknown"
                function_name = source.get("function_name") or source.get("function") or "unknown"
                snippet = source.get("snippet") or ""
                context_strings.append(f"{file_path}::{function_name}\n{snippet}")

            items.append(
                {
                    "question_id": question.get("question_id", "unknown"),
                    "repo_name": repo_name,
                    "question": question.get("question", ""),
                    "retrieved_context": context_strings,
                    "generated_answer": question.get("answer", ""),
                }
            )

    scores = []
    with tqdm(total=len(items), desc="Quality scoring") as progress:
        for item in items:
            scores.append(
                scorer.score_answer(
                    question_id=str(item["question_id"]),
                    repo_name=str(item["repo_name"]),
                    question=str(item["question"]),
                    retrieved_context=item["retrieved_context"],
                    generated_answer=str(item["generated_answer"]),
                )
            )
            progress.update(1)

    output_path = output_dir / "quality" / "quality_scores.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scorer.save_scores(scores, output_path)
    return output_path


def _run_load_test(base_url: str, output_dir: Path, repo_id: str | None, task_id: str | None) -> Path | None:
    if not repo_id:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_prefix = output_dir / "load" / f"locust_stats_{timestamp}"
    csv_prefix.parent.mkdir(parents=True, exist_ok=True)

    locustfile = _project_root() / "locustfile.py"
    command = [
        "locust",
        "-f",
        str(locustfile),
        "--headless",
        "--users",
        "10",
        "--spawn-rate",
        "1",
        "--run-time",
        "5m",
        "--host",
        base_url,
        "--csv",
        str(csv_prefix),
        "--only-summary",
    ]

    env = os.environ.copy()
    env["EVAL_REPO_ID"] = repo_id
    if task_id:
        env["EVAL_TASK_ID"] = task_id

    subprocess.run(command, check=True, env=env, cwd=str(_project_root()))
    stats_path = Path(f"{csv_prefix}_stats.csv")
    return stats_path if stats_path.exists() else None


def main() -> int:
    args = _parse_args()
    start = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        report_generator = ReportGenerator(output_dir=output_dir)
        latest_summary_files = sorted(results_dir.glob("eval_summary_*.json"))
        if not latest_summary_files:
            raise RuntimeError("No evaluation summary found for report-only mode.")

        report_paths = ReportPaths(
            eval_summary=latest_summary_files[-1],
            latency_results=output_dir / "latency" / "latency_results.json",
            quality_scores=output_dir / "quality" / "quality_scores.json",
            cost_report=output_dir / "costs" / "cost_report.json",
            load_csv=None,
        )

        markdown_path, html_path = report_generator.write_reports(report_paths)
        print(f"Report generated: {markdown_path}")
        print(f"HTML generated:   {html_path}")
        return 0

    repo_names = _resolve_repo_names(args.repos)
    if not repo_names:
        raise RuntimeError("No valid repositories selected.")

    settings = get_settings()
    budget_usd = float(args.budget_usd)

    cost_tracker = CostTracker(daily_budget_usd=budget_usd)
    config = EvalConfig(
        repos_to_eval=repo_names,
        questions_per_repo=args.questions,
        timeout_per_question=120,
        skip_already_ingested=True,
        base_url=args.base_url,
        skip_tls_verify=True,
        output_dir=results_dir,
    )

    print("=== AI Codebase Intelligence Platform - Evaluation ===")
    print(f"Target stack:  {args.base_url}")
    print(f"Repos:         {repo_names}")
    print(f"Questions:     {args.questions} per repo")
    print("Quality judge: qwen/qwen-max via OpenRouter")
    print(f"Cost limit:    ${budget_usd:.2f} USD per session")
    print("")

    with EvaluationRunner(config=config, cost_tracker=cost_tracker) as runner:
        print("[1/5] Checking repo ingestion status...")
        ingestion_map = _run_parallel_ingestion(runner, repo_names)
        for repo_name in repo_names:
            info = ingestion_map.get(repo_name, {})
            if info.get("error"):
                print(f"      {repo_name}: failed ({info['error']})")
            else:
                ingest_time = float(info.get("ingestion_time_seconds", 0.0) or 0.0)
                if ingest_time > 0:
                    print(f"      {repo_name}: ingested in {ingest_time:.1f}s")
                else:
                    print(f"      {repo_name}: already ingested")

        print("")
        print("[2/5] Running latency benchmarks...")

        benchmark_repo = next((name for name in repo_names if ingestion_map.get(name, {}).get("repo_id")), None)
        if benchmark_repo is None:
            raise RuntimeError("Unable to run latency benchmarks: no repo_id available.")

        benchmark_repo_id = str(ingestion_map[benchmark_repo]["repo_id"])
        benchmark_task_id = str(ingestion_map[benchmark_repo].get("task_id") or "")

        with LatencyBenchmark(
            base_url=args.base_url,
            samples_per_endpoint=max(5, int(args.latency_samples)),
            skip_tls_verify=True,
            timeout_seconds=30,
        ) as latency:
            latency_result = latency.run(
                repo_id=benchmark_repo_id,
                task_id=benchmark_task_id,
                ingest_payload={"github_url": get_repo_by_name(benchmark_repo).github_url, "branch": get_repo_by_name(benchmark_repo).branch},
            )
            for stat in latency_result.stats:
                print(
                    f"      {stat.endpoint:<28} p50={stat.p50_ms:7.1f}ms "
                    f"p95={stat.p95_ms:7.1f}ms p99={stat.p99_ms:7.1f}ms"
                )
                if stat.ttft_p50_ms is not None:
                    print(
                        f"        -> TTFT             p50={stat.ttft_p50_ms:7.1f}ms "
                        f"p95={stat.ttft_p95_ms:7.1f}ms p99={stat.ttft_p99_ms:7.1f}ms"
                    )

        print("")
        print("[3/5] Running evaluation questions through RAG chain...")
        with tqdm(total=len(repo_names), desc="Question evaluation") as progress:
            eval_result = runner.run()
            progress.update(len(repo_names))

    print("")
    quality_scores_path = output_dir / "quality" / "quality_scores.json"
    if args.skip_quality_scoring:
        print("[4/5] Skipping quality scoring (--skip-quality-scoring)")
    else:
        print("[4/5] Scoring response quality with qwen/qwen-max judge...")
        quality_scores_path = _run_quality_scoring(
            output_dir=output_dir,
            judge_model="qwen/qwen-max",
            api_key=settings.openrouter_api_key,
        )

    print("")
    load_stats_path: Path | None = None
    if args.skip_load_test:
        print("[5/5] Skipping load test (--skip-load-test)")
    else:
        print("[5/5] Running Locust load test (10 users, 5m)...")
        first_repo = repo_names[0]
        first_repo_info = ingestion_map.get(first_repo, {})
        load_stats_path = _run_load_test(
            base_url=args.base_url,
            output_dir=output_dir,
            repo_id=first_repo_info.get("repo_id"),
            task_id=first_repo_info.get("task_id"),
        )

    report_generator = ReportGenerator(output_dir=output_dir)
    latest_summary_files = sorted((output_dir / "results").glob("eval_summary_*.json"))
    if not latest_summary_files:
        raise RuntimeError("No eval summary found to generate report.")

    report_paths = ReportPaths(
        eval_summary=latest_summary_files[-1],
        latency_results=output_dir / "latency" / "latency_results.json",
        quality_scores=quality_scores_path,
        cost_report=output_dir / "costs" / "cost_report.json",
        load_csv=load_stats_path,
    )

    cost_report_path = output_dir / "costs" / "cost_report.json"
    cost_report_path.parent.mkdir(parents=True, exist_ok=True)
    cost_report_path.write_text(json.dumps(cost_tracker.to_report(), indent=2), encoding="utf-8")

    markdown_path, html_path = report_generator.write_reports(report_paths)

    total_questions = sum(len(repo.questions) for repo in eval_result.repos)
    avg_quality = None
    if quality_scores_path.exists():
        quality_payload = json.loads(quality_scores_path.read_text(encoding="utf-8"))
        avg_quality = quality_payload.get("avg_composite")

    elapsed_seconds = time.perf_counter() - start
    print("")
    print("=== EVALUATION COMPLETE ===")
    print(f"Repos evaluated:    {len(eval_result.repos)}")
    print(f"Total questions:    {total_questions}")
    print(f"Total tokens used:  {cost_tracker.total_prompt_tokens + cost_tracker.total_completion_tokens:,}")
    print(f"Total cost (USD):   ${cost_tracker.total_cost_usd:.4f}")
    print(f"Avg quality score:  {avg_quality if avg_quality is not None else 'N/A'} / 1.00")
    print(f"Report saved to:    {markdown_path}")
    print(f"HTML report saved:  {html_path}")
    print(f"Elapsed time:       {elapsed_seconds:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
