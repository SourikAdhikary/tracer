"""Tracer CLI — autonomous sponsorship auditor for sports broadcasts.

Usage:
    python -m tracer.cli audit --video match.mp4 --brands "Fly Emirates,Etihad,Adidas"
    python -m tracer.cli audit --video match.mp4 --brands-file brands.txt
"""

import argparse
import sys
from pathlib import Path

from tracer.config import Config
from tracer.pipeline import run_pipeline


def parse_brands_file(path: str) -> list[str]:
    """Read brand names from a text file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def cmd_audit(args: argparse.Namespace) -> None:
    """Run the audit pipeline."""
    # Parse brands
    if args.brands:
        brands = [b.strip() for b in args.brands.split(",")]
    elif args.brands_file:
        brands = parse_brands_file(args.brands_file)
    else:
        print("Error: specify --brands or --brands-file", file=sys.stderr)
        sys.exit(1)

    # Build config
    config = Config()

    if args.output:
        config.paths.output_dir = Path(args.output)
        config.paths.crops_dir = Path(args.output) / "crops"

    if args.fps:
        config.pipeline.extraction_fps = args.fps

    if args.token_budget:
        config.models.auditor_token_budget = args.token_budget

    # Run pipeline
    result = run_pipeline(
        video_path=args.video,
        brands=brands,
        config=config,
    )

    if result["status"] == "complete":
        print(f"\nDone. Reports at: {result['json_report']}")
    else:
        print(f"\nPipeline ended: {result['status']}")


def main():
    parser = argparse.ArgumentParser(
        prog="tracer",
        description="Tracer v4 — Autonomous Sponsorship Auditor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Run sponsorship audit on a video")
    audit_parser.add_argument("--video", required=True, help="Path to video file or YouTube URL")
    audit_parser.add_argument("--brands", help='Comma-separated brand names (e.g., "Fly Emirates,Etihad")')
    audit_parser.add_argument("--brands-file", help="Path to text file with brand names (one per line)")
    audit_parser.add_argument("--output", help="Output directory (default: ./output)")
    audit_parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS (default: 1.0)")
    audit_parser.add_argument("--token-budget", type=int, choices=[70, 140, 280, 560, 1120],
                              help="Auditor vision token budget (default: 560)")
    audit_parser.set_defaults(func=cmd_audit)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
