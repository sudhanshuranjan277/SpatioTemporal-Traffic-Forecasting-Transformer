from __future__ import annotations

from workspace_common.core.discovery import run_discovered_modules


def run(context):
    context.progress.info("Traffic-Forecasting-Research entrypoint loaded")
    result = run_discovered_modules(context)
    context.output_manager.save_json("summaries", "project_summary.json", result)
    return result

