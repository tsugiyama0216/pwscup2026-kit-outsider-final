"""python -m mock_TS: generate / export / attack / score / benchmark / run."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .common import OUTPUT_ROOT, verify_originals


def parser():
    root = argparse.ArgumentParser(description='本選の既存Cを独立に評価するmock_TS')
    sub = root.add_subparsers(dest='action', required=True)
    for action in ('generate', 'run', 'export', 'attack', 'score', 'benchmark'):
        p = sub.add_parser(action)
        p.add_argument('--output', type=Path)
        if action in ('generate', 'run'):
            p.add_argument('--seed', type=int, default=2026100701)
            p.add_argument('--count', type=int, default=3)
            p.add_argument('--selection-seed', type=int, default=20261007)
            p.add_argument('--r-count', type=int, default=10)
        if action in ('export', 'score', 'benchmark'):
            p.add_argument('--suite', type=Path, required=True)
        if action in ('export', 'score'):
            p.add_argument('--case', dest='case_id', required=True)
            p.add_argument('--model', required=True)
        if action == 'attack':
            p.add_argument('--bundle', type=Path, required=True)
        if action == 'score':
            p.add_argument('--predictions', type=Path, required=True)
        if action in ('attack', 'run', 'benchmark'):
            choices = p.add_mutually_exclusive_group()
            choices.add_argument('--plugin', type=Path)
            choices.add_argument('--method', choices=('official', 'background_only', 'constant'), default='official')
            p.add_argument('--timeout', type=int, default=600)
        if action in ('run', 'benchmark', 'score'):
            p.add_argument('--floor-repetitions', type=int, default=1000)
        if action in ('benchmark', 'run'):
            p.add_argument('--models', nargs='+')
            p.add_argument('--cases', nargs='+')
    return root


def main():
    args = parser().parse_args()
    if args.output is None:
        args.output = OUTPUT_ROOT / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + args.action)
    verify_originals()
    from . import runner
    if args.action in ('generate', 'run'):
        from .generation import generate
        suite = args.output if args.action == 'generate' else args.output / 'suite'
        generate(suite, seed=args.seed, count=args.count,
                 selection_seed=args.selection_seed, r_count=args.r_count)
        if args.action == 'generate':
            print(suite, flush=True)
            return
        args.suite = suite
        args.output = args.output / 'evaluation'
        args.action = 'benchmark'
    if args.action == 'export':
        result = runner.export_bundle(args.suite, args.case_id, args.model, args.output)
    elif args.action == 'attack':
        result = runner.attack_bundle(args.bundle, args.output, plugin=args.plugin,
                                      method=args.method, timeout=args.timeout)
    elif args.action == 'score':
        runner.score_submission(args.suite, args.case_id, args.model, args.predictions,
                                args.output, floor_repetitions=args.floor_repetitions)
        result = args.output / 'detailed_results.html'
    else:
        result = runner.benchmark(args.suite, args.output, plugin=args.plugin, method=args.method,
                 floor_repetitions=args.floor_repetitions, models=args.models, cases=args.cases,
                 timeout=args.timeout)
    print(result, flush=True)


if __name__ == '__main__':
    main()
