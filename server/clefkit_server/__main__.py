"""Command-line entry point: ``python -m clefkit_server``."""

from __future__ import annotations

import argparse
import sys

from clefkit_server.app import DEFAULT_PORT, make_server, require_loopback
from clefkit_server.engines import Engine, FakeEngine, MlxEngine, TorchEngine, check_torch_memory

DEFAULT_MIN_FREE_GIB = 14.0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the server command line."""
    parser = argparse.ArgumentParser(prog="clefkit_server", description="Local SystemOne server.")
    parser.add_argument("--engine", choices=("torch", "mlx", "fake"), required=True)
    parser.add_argument("--model-dir", help="Clef release folder (required for torch and mlx)")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--allow-remote", action="store_true", help="permit binding a non-loopback host"
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=DEFAULT_MIN_FREE_GIB,
        help="refuse the torch engine below this much free memory (macOS only)",
    )
    return parser


def build_engine(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Engine:
    """Create the engine named on the command line, running the torch memory guard first."""
    if args.engine == "fake":
        return FakeEngine()
    if args.model_dir is None:
        parser.error(f"--model-dir is required for --engine {args.engine}")
    if args.engine == "mlx":
        return MlxEngine(args.model_dir)
    free = check_torch_memory(args.min_free_gib)
    if free is None:
        print("clefkit_server: memory check skipped (not macOS)", file=sys.stderr)
    else:
        print(f"clefkit_server: {free:.1f} GiB free, minimum {args.min_free_gib:g}", file=sys.stderr)
    return TorchEngine(args.model_dir, args.device)


def _address(host: str, port: int) -> str:
    """Format ``host:port``, bracketing IPv6 literals."""
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def main(argv: list[str] | None = None) -> int:
    """Start the server and block until interrupted; return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_loopback(args.host, args.allow_remote)
        engine = build_engine(args, parser)
    except (ValueError, RuntimeError) as exc:
        print(f"clefkit_server: {exc}", file=sys.stderr)
        return 2
    try:
        server = make_server(engine, args.host, args.port)
    except OSError as exc:
        print(f"clefkit_server: cannot listen on {_address(args.host, args.port)}: {exc}", file=sys.stderr)
        return 1
    host, port = server.server_address[:2]
    print(f"clefkit_server: {engine.name} engine on http://{_address(host, port)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
