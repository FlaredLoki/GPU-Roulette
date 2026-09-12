"""
src/cli.py — §3.13

Purpose: Command-line interface. The guaranteed demo.
"""
import argparse
import sys
import json
from src import predictor
from src import diagnostics

def _cmd_analyze(args):
    if not args.file and not args.code:
        print("Error: Must provide either a file or --code", file=sys.stderr)
        sys.exit(1)
        
    code_str = args.code
    filename = "<stdin>"
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                code_str = f.read()
            filename = args.file
        except Exception as e:
            print(f"Error reading file {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
            
    try:
        pred = predictor.GPUGatePredictor()
        results, _ = pred.predict(code_str, preprocess=not args.no_preprocess)
    except Exception as e:
        print(f"Error predicting: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not results:
        print(f"{filename}: remark: No loop candidates found to analyze.")
        return
        
    if args.format == 'diagnostic':
        for r in results:
            print(diagnostics.format_diagnostic(r, filename=filename))
            print()
    elif args.format == 'json':
        out_list = []
        for r in results:
            d = {
                "line": r.loop_line,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "crossover_n": r.crossover_n,
                "blockers": [b.get("reason", str(b)) if isinstance(b, dict) else str(b) for b in r.blockers],
                "fix_suggestions": [(f.get("problem", '') + ' -> ' + f.get("fix", '')) if isinstance(f, dict) else str(f) for f in r.fix_suggestions]
            }
            out_list.append(d)
        print(json.dumps(out_list, indent=2))
    elif args.format == 'table':
        print(diagnostics.format_batch_summary(results, filename))


def _cmd_batch(args):
    all_results = []
    pred = predictor.GPUGatePredictor()
    for fpath in args.files:
        try:
            results, _ = pred.predict_file(fpath)
            all_results.extend([(fpath, r) for r in results])
            print(diagnostics.format_batch_summary(results, fpath))
            print()
        except Exception as e:
            print(f"Error analyzing {fpath}: {e}", file=sys.stderr)
            
    if args.report:
        try:
            out_list = []
            for fpath, r in all_results:
                d = {
                    "file": fpath,
                    "line": r.loop_line,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "crossover_n": r.crossover_n
                }
                out_list.append(d)
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(out_list, f, indent=2)
            print(f"Saved batch report to {args.report}")
        except Exception as e:
            print(f"Error saving report: {e}", file=sys.stderr)


def _cmd_train(args):
    print("Training command invoked (mocked for now, needs teammate's train.py)")
    import sys
    try:
        from src import train
        # Assume teammate provides train module with appropriate functions or main script
        # Alternatively, we could just execute the script
    except ImportError:
        print("Error: train module not found. Check with teammate.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        prog='gpugate',
        description='GPU Parallelization Profitability Predictor'
    )
    subparsers = parser.add_subparsers(dest='command')

    # analyze
    p_analyze = subparsers.add_parser('analyze')
    p_analyze.add_argument('file', nargs='?', help='C source file')
    p_analyze.add_argument('--code', help='Inline C code string')
    p_analyze.add_argument('--format', choices=['diagnostic', 'json', 'table'],
                           default='diagnostic')
    p_analyze.add_argument('--no-preprocess', action='store_true',
                           help='Skip gcc -E preprocessing')

    # batch
    p_batch = subparsers.add_parser('batch')
    p_batch.add_argument('files', nargs='+')
    p_batch.add_argument('--report', help='Output JSON report path')

    # train
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--evaluate', action='store_true')
    p_train.add_argument('--compare-baseline', action='store_true')
    p_train.add_argument('--n-samples', type=int, default=5000)
    p_train.add_argument('--output', default='models/gpugate_model.pkl')

    args = parser.parse_args()
    
    if args.command == 'analyze':
        _cmd_analyze(args)
    elif args.command == 'batch':
        _cmd_batch(args)
    elif args.command == 'train':
        _cmd_train(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()