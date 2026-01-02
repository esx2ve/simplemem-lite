"""SimpleMem Lite CLI with beautiful Rich output.

Provides commands for:
- Backend status and info
- FalkorDB management via Apptainer/Docker
- Memory search and storage
- Project management
- User-space installation

Usage:
    simplemem status           # Show backend status
    simplemem start            # Start FalkorDB (Apptainer or Docker)
    simplemem stop             # Stop FalkorDB
    simplemem search "query"   # Search memories
    simplemem install          # Install to ~/.claude
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize Typer app and Rich console
app = typer.Typer(
    name="simplemem",
    help="SimpleMem Lite - Long-term memory for Claude Code",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# Default paths - can be overridden by environment variables
DEFAULT_APPTAINER_IMAGE_DIR = os.environ.get(
    "SIMPLEMEM_APPTAINER_DIR",
    str(Path.home() / ".simplemem" / "apptainer" / "images")
)
DEFAULT_APPTAINER_DATA_DIR = os.environ.get(
    "SIMPLEMEM_DATA_DIR",
    str(Path.home() / ".simplemem" / "data")
)
FALKORDB_IMAGE = "falkordb_latest.sif"
FALKORDB_INSTANCE = "simplemem-falkordb"
FALKORDB_PORT = int(os.environ.get("SIMPLEMEM_FALKORDB_PORT", "6379"))


def print_banner():
    """Print SimpleMem banner."""
    banner = Text()
    banner.append("SimpleMem", style="bold cyan")
    banner.append(" Lite", style="cyan")
    console.print(Panel(banner, border_style="cyan", box=box.ROUNDED))


def _run_cmd(cmd: list[str], timeout: int = 30, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _check_apptainer() -> tuple[bool, str | None]:
    """Check if Apptainer is available.

    Returns:
        (is_available, path_to_binary)
    """
    try:
        result = _run_cmd(["which", "apptainer"], check=False)
        if result.returncode == 0:
            return True, result.stdout.strip()
    except Exception:
        pass

    # Check common user-space locations
    for path in [
        Path.home() / ".local" / "apptainer" / "bin" / "apptainer",
        Path.home() / ".local" / "bin" / "apptainer",
        Path("/usr/local/bin/apptainer"),
    ]:
        if path.exists():
            return True, str(path)

    return False, None


def _check_docker() -> tuple[bool, str | None]:
    """Check if Docker is available and running.

    Returns:
        (is_available, path_to_binary)
    """
    try:
        result = _run_cmd(["which", "docker"], check=False)
        if result.returncode != 0:
            return False, None

        docker_path = result.stdout.strip()

        # Check if Docker daemon is running
        result = _run_cmd(["docker", "info"], timeout=10, check=False)
        if result.returncode == 0:
            return True, docker_path
        return False, docker_path  # Docker installed but not running
    except Exception:
        return False, None


def _get_apptainer_instances() -> list[dict]:
    """Get list of running Apptainer instances."""
    has_apptainer, apptainer_path = _check_apptainer()
    if not has_apptainer:
        return []

    try:
        result = _run_cmd([apptainer_path, "instance", "list", "--json"], check=False)
        if result.returncode == 0 and result.stdout.strip():
            import json
            data = json.loads(result.stdout)
            return data.get("instances", [])
    except Exception:
        pass
    return []


def _is_falkordb_running() -> tuple[bool, str]:
    """Check if FalkorDB is running.

    Returns:
        (is_running, method) where method is 'apptainer', 'docker', or 'native'
    """
    # Check Apptainer instances
    instances = _get_apptainer_instances()
    for inst in instances:
        if FALKORDB_INSTANCE in inst.get("instance", ""):
            return True, "apptainer"

    # Check Docker
    try:
        result = _run_cmd(
            ["docker", "ps", "--filter", f"name={FALKORDB_INSTANCE}", "--format", "{{.Names}}"],
            check=False,
        )
        if result.returncode == 0 and FALKORDB_INSTANCE in result.stdout:
            return True, "docker"
    except Exception:
        pass

    # Check native Redis/FalkorDB on default port
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', FALKORDB_PORT))
        sock.close()
        if result == 0:
            return True, "native"
    except Exception:
        pass

    return False, ""


@app.command()
def status():
    """Show backend status and availability."""
    from simplemem_lite.db.graph_factory import get_backend_info

    print_banner()

    info = get_backend_info()

    # Backend status table
    table = Table(title="Graph Backend Status", box=box.ROUNDED)
    table.add_column("Backend", style="cyan")
    table.add_column("Installed", justify="center")
    table.add_column("Available", justify="center")
    table.add_column("Error", style="dim")

    for backend in ["falkordb", "kuzu"]:
        bi = info[backend]
        installed = "[green]Yes[/green]" if bi["installed"] else "[red]No[/red]"
        available = "[green]Yes[/green]" if bi["available"] else "[yellow]No[/yellow]"
        error = bi.get("error") or "-"
        table.add_row(backend.upper(), installed, available, error[:40])

    console.print(table)

    # Active backend
    active = info.get("active", "none")
    if active:
        console.print(f"\n[bold]Active backend:[/bold] [green]{active}[/green]")
    else:
        console.print("\n[bold red]No backend available![/bold red]")

    # Environment override
    if info.get("env_override"):
        console.print(f"[dim]Override: SIMPLEMEM_GRAPH_BACKEND={info['env_override']}[/dim]")

    # Check FalkorDB running status
    is_running, method = _is_falkordb_running()
    if is_running:
        console.print(f"\n[bold]FalkorDB:[/bold] [green]Running[/green] via {method} on port {FALKORDB_PORT}")
    else:
        console.print(f"\n[bold]FalkorDB:[/bold] [yellow]Not running[/yellow] (use: simplemem start)")


@app.command()
def start(
    backend: str = typer.Option(
        "auto",
        "--backend", "-b",
        help="Backend to use: auto, apptainer, or docker",
    ),
    port: int = typer.Option(
        FALKORDB_PORT,
        "--port", "-p",
        help="Port for FalkorDB",
    ),
    image_dir: str = typer.Option(
        DEFAULT_APPTAINER_IMAGE_DIR,
        "--image-dir",
        help="Directory containing FalkorDB SIF image",
    ),
    data_dir: str = typer.Option(
        DEFAULT_APPTAINER_DATA_DIR,
        "--data-dir",
        help="Directory for FalkorDB data persistence",
    ),
    pull: bool = typer.Option(
        False,
        "--pull",
        help="Pull/download the FalkorDB image if not present",
    ),
):
    """Start FalkorDB graph database.

    Automatically detects and uses Apptainer (preferred for HPC) or Docker.
    Apptainer is preferred as it doesn't require root access.

    Examples:
        simplemem start                    # Auto-detect backend
        simplemem start --backend docker   # Force Docker
        simplemem start --pull             # Download image if missing
    """
    print_banner()

    # Check if already running
    is_running, method = _is_falkordb_running()
    if is_running:
        console.print(f"[yellow]FalkorDB already running via {method} on port {port}[/yellow]")
        return

    # Detect available backends
    has_apptainer, apptainer_path = _check_apptainer()
    has_docker, docker_path = _check_docker()

    # Select backend
    selected_backend = None
    if backend == "auto":
        # Prefer Apptainer (works without root on HPC)
        if has_apptainer:
            selected_backend = "apptainer"
        elif has_docker:
            selected_backend = "docker"
    elif backend == "apptainer":
        if has_apptainer:
            selected_backend = "apptainer"
        else:
            console.print("[red]Apptainer not found![/red]")
            console.print("Install with: See docs/APPTAINER.md")
            raise typer.Exit(1)
    elif backend == "docker":
        if has_docker:
            selected_backend = "docker"
        else:
            console.print("[red]Docker not available![/red]")
            console.print("Make sure Docker is installed and running.")
            raise typer.Exit(1)

    if not selected_backend:
        console.print("[red]No container runtime available![/red]")
        console.print("\nOptions:")
        console.print("  1. Install Apptainer (see docs/APPTAINER.md)")
        console.print("  2. Install and start Docker")
        console.print("  3. Use KuzuDB instead (embedded, no container needed)")
        raise typer.Exit(1)

    console.print(f"[bold]Using backend:[/bold] {selected_backend}")

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if selected_backend == "apptainer":
        _start_apptainer(apptainer_path, image_dir, data_dir, port, pull)
    else:
        _start_docker(docker_path, data_dir, port, pull)


def _start_apptainer(apptainer_path: str, image_dir: str, data_dir: str, port: int, pull: bool):
    """Start FalkorDB via Apptainer."""
    image_path = Path(image_dir) / FALKORDB_IMAGE

    # Check if image exists
    if not image_path.exists():
        if pull:
            console.print(f"[yellow]Pulling FalkorDB image...[/yellow]")
            Path(image_dir).mkdir(parents=True, exist_ok=True)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading falkordb/falkordb:latest...", total=None)
                try:
                    result = _run_cmd(
                        [apptainer_path, "pull", f"{image_path}", "docker://falkordb/falkordb:latest"],
                        timeout=600,
                        check=False,
                    )
                    if result.returncode != 0:
                        console.print(f"[red]Failed to pull image:[/red] {result.stderr}")
                        raise typer.Exit(1)
                    progress.update(task, description="Done!")
                except subprocess.TimeoutExpired:
                    console.print("[red]Timeout pulling image (10 min limit)[/red]")
                    raise typer.Exit(1)
        else:
            console.print(f"[red]FalkorDB image not found at:[/red] {image_path}")
            console.print("\nTo download the image, run:")
            console.print(f"  simplemem start --pull")
            console.print("\nOr manually:")
            console.print(f"  apptainer pull {image_path} docker://falkordb/falkordb:latest")
            raise typer.Exit(1)

    # Start instance
    console.print(f"[cyan]Starting FalkorDB instance...[/cyan]")

    try:
        result = _run_cmd(
            [
                apptainer_path, "instance", "start",
                "--bind", f"{data_dir}:/data",
                str(image_path),
                FALKORDB_INSTANCE,
            ],
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            console.print(f"[red]Failed to start instance:[/red] {result.stderr}")
            raise typer.Exit(1)
    except subprocess.TimeoutExpired:
        console.print("[red]Timeout starting instance[/red]")
        raise typer.Exit(1)

    # Start redis-server with FalkorDB module
    console.print(f"[cyan]Starting FalkorDB server on port {port}...[/cyan]")

    try:
        result = _run_cmd(
            [
                apptainer_path, "exec", f"instance://{FALKORDB_INSTANCE}",
                "redis-server",
                "--port", str(port),
                "--loadmodule", "/var/lib/falkordb/bin/falkordb.so",
                "--daemonize", "yes",
                "--dir", "/data",
            ],
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            console.print(f"[red]Failed to start redis-server:[/red] {result.stderr}")
            # Clean up instance
            _run_cmd([apptainer_path, "instance", "stop", FALKORDB_INSTANCE], check=False)
            raise typer.Exit(1)
    except subprocess.TimeoutExpired:
        console.print("[red]Timeout starting redis-server[/red]")
        raise typer.Exit(1)

    # Verify it's running
    import time
    time.sleep(1)

    is_running, _ = _is_falkordb_running()
    if is_running:
        console.print(f"\n[green]✓ FalkorDB started successfully![/green]")
        console.print(f"  Port: {port}")
        console.print(f"  Data: {data_dir}")
        console.print(f"\n[dim]Stop with: simplemem stop[/dim]")
    else:
        console.print("[red]FalkorDB started but not responding[/red]")
        raise typer.Exit(1)


def _start_docker(docker_path: str, data_dir: str, port: int, pull: bool):
    """Start FalkorDB via Docker."""
    console.print(f"[cyan]Starting FalkorDB via Docker...[/cyan]")

    cmd = [
        docker_path, "run", "-d",
        "--name", FALKORDB_INSTANCE,
        "-p", f"{port}:6379",
        "-v", f"{data_dir}:/data",
        "falkordb/falkordb:latest",
    ]

    if pull:
        # Pull latest image first
        console.print("[yellow]Pulling latest FalkorDB image...[/yellow]")
        _run_cmd([docker_path, "pull", "falkordb/falkordb:latest"], timeout=600, check=False)

    try:
        result = _run_cmd(cmd, timeout=60, check=False)
        if result.returncode != 0:
            if "already in use" in result.stderr:
                console.print("[yellow]Container name already in use. Removing old container...[/yellow]")
                _run_cmd([docker_path, "rm", "-f", FALKORDB_INSTANCE], check=False)
                result = _run_cmd(cmd, timeout=60, check=False)

            if result.returncode != 0:
                console.print(f"[red]Failed to start Docker container:[/red] {result.stderr}")
                raise typer.Exit(1)
    except subprocess.TimeoutExpired:
        console.print("[red]Timeout starting Docker container[/red]")
        raise typer.Exit(1)

    # Verify it's running
    import time
    time.sleep(2)

    is_running, _ = _is_falkordb_running()
    if is_running:
        console.print(f"\n[green]✓ FalkorDB started successfully![/green]")
        console.print(f"  Port: {port}")
        console.print(f"  Data: {data_dir}")
        console.print(f"\n[dim]Stop with: simplemem stop[/dim]")
    else:
        console.print("[red]FalkorDB started but not responding[/red]")
        raise typer.Exit(1)


@app.command()
def stop():
    """Stop FalkorDB graph database.

    Stops the running FalkorDB instance (Apptainer or Docker).
    """
    print_banner()

    is_running, method = _is_falkordb_running()
    if not is_running:
        console.print("[yellow]FalkorDB is not running.[/yellow]")
        return

    console.print(f"[cyan]Stopping FalkorDB ({method})...[/cyan]")

    if method == "apptainer":
        has_apptainer, apptainer_path = _check_apptainer()
        if has_apptainer:
            try:
                result = _run_cmd(
                    [apptainer_path, "instance", "stop", FALKORDB_INSTANCE],
                    timeout=30,
                    check=False,
                )
                if result.returncode == 0:
                    console.print("[green]✓ FalkorDB stopped.[/green]")
                else:
                    console.print(f"[red]Failed to stop:[/red] {result.stderr}")
            except Exception as e:
                console.print(f"[red]Error stopping instance:[/red] {e}")

    elif method == "docker":
        has_docker, docker_path = _check_docker()
        if docker_path:
            try:
                _run_cmd([docker_path, "stop", FALKORDB_INSTANCE], timeout=30, check=False)
                _run_cmd([docker_path, "rm", FALKORDB_INSTANCE], timeout=10, check=False)
                console.print("[green]✓ FalkorDB stopped.[/green]")
            except Exception as e:
                console.print(f"[red]Error stopping container:[/red] {e}")

    elif method == "native":
        console.print("[yellow]Native FalkorDB detected - stop it manually or use redis-cli SHUTDOWN[/yellow]")


@app.command()
def info():
    """Show detailed system information."""
    import platform

    print_banner()

    table = Table(title="System Information", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("Machine", platform.machine())

    # Check GLIBC version (Linux only)
    if platform.system() == "Linux":
        try:
            import subprocess

            result = subprocess.run(
                ["ldd", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            glibc = result.stdout.split("\n")[0] if result.stdout else "Unknown"
            table.add_row("GLIBC", glibc)
        except Exception:
            table.add_row("GLIBC", "Unknown")

    # Config paths
    home = Path.home()
    table.add_row("Config Dir", str(home / ".simplemem"))
    table.add_row("Claude Skills", str(home / ".claude" / "skills"))

    console.print(table)


@app.command()
def install(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be done"),
    skip_mcp: bool = typer.Option(False, "--skip-mcp", help="Skip MCP server registration"),
    skip_skills: bool = typer.Option(False, "--skip-skills", help="Skip skills installation"),
    skip_agents: bool = typer.Option(False, "--skip-agents", help="Skip agents installation"),
    skip_permissions: bool = typer.Option(False, "--skip-permissions", help="Skip permission setup"),
    venv_path: Optional[str] = typer.Option(None, "--venv", help="Path to venv (auto-detected if not specified)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="SimpleMem data directory"),
    use_local_embeddings: bool = typer.Option(True, "--local-embeddings/--no-local-embeddings", help="Use local sentence-transformers"),
):
    """Install SimpleMem to Claude Code (user scope).

    This command installs:
    - MCP server config to ~/.claude.json (user-scope, works across all projects)
    - Skills to ~/.claude/skills/
    - Agents to ~/.claude/agents/
    - Auto-allow permissions for simplemem tools

    For HPC clusters, specify --venv to point to your installation:
        simplemem install --venv /path/to/simplemem/.venv

    Examples:
        simplemem install                    # Full installation
        simplemem install --dry-run          # Preview changes
        simplemem install --force            # Overwrite existing
        simplemem install --venv ~/.venvs/simplemem  # Custom venv
    """
    import shutil
    import json

    print_banner()
    console.print("[bold]Installing SimpleMem to Claude Code (user scope)...[/bold]\n")

    home = Path.home()
    claude_json = home / ".claude.json"
    claude_settings = home / ".claude" / "settings.json"

    # Auto-detect venv path
    if venv_path:
        venv = Path(venv_path)
    else:
        # Check if we're running from a venv
        import sys
        if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
            venv = Path(sys.prefix)
        else:
            # Fall back to looking in common locations
            src_root = Path(__file__).parent.parent
            if (src_root / ".venv").exists():
                venv = src_root / ".venv"
            else:
                venv = None

    if not venv or not venv.exists():
        console.print("[red]Could not auto-detect venv path.[/red]")
        console.print("Please specify with: simplemem install --venv /path/to/venv")
        raise typer.Exit(1)

    # Find simplemem-lite executable
    simplemem_bin = venv / "bin" / "simplemem-lite"
    if not simplemem_bin.exists():
        console.print(f"[red]simplemem-lite not found at {simplemem_bin}[/red]")
        console.print("Make sure simplemem-lite is installed: uv pip install -e .")
        raise typer.Exit(1)

    console.print(f"[dim]Using venv: {venv}[/dim]")
    console.print(f"[dim]MCP binary: {simplemem_bin}[/dim]\n")

    # ===== MCP Server Registration =====
    if not skip_mcp:
        console.print("[bold cyan]MCP Server Registration[/bold cyan]")

        # Build MCP server config
        mcp_config = {
            "type": "stdio",
            "command": str(simplemem_bin),
            "args": [],
            "env": {
                "HOME": str(home),
                "SIMPLEMEM_LITE_DATA_DIR": data_dir or str(home / ".simplemem_lite"),
                "SIMPLEMEM_LITE_CLAUDE_TRACES_DIR": str(home / ".claude" / "projects"),
            }
        }

        if use_local_embeddings:
            mcp_config["env"]["SIMPLEMEM_LITE_USE_LOCAL_EMBEDDINGS"] = "true"

        # Add API keys from environment if available
        for key in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"]:
            val = os.environ.get(key)
            if val:
                mcp_config["env"][key] = val

        # Load or create ~/.claude.json
        if claude_json.exists():
            try:
                config = json.loads(claude_json.read_text())
            except json.JSONDecodeError:
                console.print("[yellow]Warning: ~/.claude.json is invalid, creating backup[/yellow]")
                shutil.copy2(claude_json, claude_json.with_suffix(".json.bak"))
                config = {}
        else:
            config = {}

        # Ensure mcpServers exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Check if already exists
        existing = config["mcpServers"].get("simplemem-lite")
        if existing and not force:
            console.print(f"  [yellow]○[/yellow] simplemem-lite MCP server already registered")
            console.print(f"    [dim]Use --force to overwrite[/dim]")
        else:
            if not dry_run:
                config["mcpServers"]["simplemem-lite"] = mcp_config
                claude_json.write_text(json.dumps(config, indent=2))
            action = "Would register" if dry_run else "Registered"
            console.print(f"  [green]✓[/green] {action} simplemem-lite MCP server")

    # ===== Skills Installation =====
    if not skip_skills:
        console.print("\n[bold cyan]Skills Installation[/bold cyan]")

        src_root = Path(__file__).parent.parent.parent  # simplemem repo root
        skills_src = src_root / ".claude" / "skills"
        skills_dst = home / ".claude" / "skills"

        if not skills_src.exists():
            console.print(f"  [yellow]○[/yellow] No skills found in {skills_src}")
        else:
            if not dry_run:
                skills_dst.mkdir(parents=True, exist_ok=True)

            for skill_dir in skills_src.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    dst = skills_dst / skill_dir.name
                    exists = dst.exists()

                    if exists and not force:
                        console.print(f"  [yellow]○[/yellow] {skill_dir.name} [dim](exists, use --force)[/dim]")
                    else:
                        if not dry_run:
                            if dst.exists():
                                shutil.rmtree(dst)
                            shutil.copytree(skill_dir, dst)
                        action = "Would install" if dry_run else "Installed"
                        console.print(f"  [green]✓[/green] {action} skill: {skill_dir.name}")

    # ===== Agents Installation =====
    if not skip_agents:
        console.print("\n[bold cyan]Agents Installation[/bold cyan]")

        src_root = Path(__file__).parent.parent.parent
        agents_src = src_root / ".claude" / "agents"
        agents_dst = home / ".claude" / "agents"

        if not agents_src.exists():
            console.print(f"  [yellow]○[/yellow] No agents found in {agents_src}")
        else:
            if not dry_run:
                agents_dst.mkdir(parents=True, exist_ok=True)

            for agent_file in agents_src.glob("*.md"):
                dst = agents_dst / agent_file.name
                exists = dst.exists()

                if exists and not force:
                    console.print(f"  [yellow]○[/yellow] {agent_file.stem} [dim](exists, use --force)[/dim]")
                else:
                    if not dry_run:
                        shutil.copy2(agent_file, dst)
                    action = "Would install" if dry_run else "Installed"
                    console.print(f"  [green]✓[/green] {action} agent: {agent_file.stem}")

    # ===== Permissions Setup =====
    if not skip_permissions:
        console.print("\n[bold cyan]Permissions Setup[/bold cyan]")

        # SimpleMem tools to auto-allow
        simplemem_permissions = [
            "mcp__simplemem-lite__store_memory",
            "mcp__simplemem-lite__search_memories",
            "mcp__simplemem-lite__ask_memories",
            "mcp__simplemem-lite__reason_memories",
            "mcp__simplemem-lite__relate_memories",
            "mcp__simplemem-lite__process_trace",
            "mcp__simplemem-lite__get_stats",
            "mcp__simplemem-lite__search_code",
            "mcp__simplemem-lite__index_directory",
            "mcp__simplemem-lite__code_stats",
            "mcp__simplemem-lite__check_code_staleness",
            "mcp__simplemem-lite__start_code_watching",
            "mcp__simplemem-lite__stop_code_watching",
            "mcp__simplemem-lite__get_watcher_status",
            "mcp__simplemem-lite__bootstrap_project",
            "mcp__simplemem-lite__get_project_status",
            "mcp__simplemem-lite__create_todo",
            "mcp__simplemem-lite__list_todos",
            "mcp__simplemem-lite__update_todo",
            "mcp__simplemem-lite__get_graph_schema",
            "mcp__simplemem-lite__run_cypher_query",
        ]

        if claude_settings.exists():
            try:
                settings = json.loads(claude_settings.read_text())
            except json.JSONDecodeError:
                settings = {}
        else:
            settings = {}

        # Ensure permissions.allow exists
        if "permissions" not in settings:
            settings["permissions"] = {}
        if "allow" not in settings["permissions"]:
            settings["permissions"]["allow"] = []

        # Add missing permissions
        existing = set(settings["permissions"]["allow"])
        added = []
        for perm in simplemem_permissions:
            if perm not in existing:
                if not dry_run:
                    settings["permissions"]["allow"].append(perm)
                added.append(perm)

        if added:
            if not dry_run:
                claude_settings.parent.mkdir(parents=True, exist_ok=True)
                claude_settings.write_text(json.dumps(settings, indent=2))
            console.print(f"  [green]✓[/green] Added {len(added)} tool permissions")
            if len(added) <= 5:
                for perm in added:
                    console.print(f"    [dim]{perm}[/dim]")
            else:
                console.print(f"    [dim]{added[0]}[/dim]")
                console.print(f"    [dim]... and {len(added)-1} more[/dim]")
        else:
            console.print(f"  [green]✓[/green] All permissions already configured")

    # ===== Summary =====
    console.print("\n" + "=" * 50)
    if dry_run:
        console.print("\n[yellow]Dry run complete - no changes made.[/yellow]")
        console.print("Run without --dry-run to apply changes.")
    else:
        console.print("\n[green bold]✓ Installation complete![/green bold]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Restart Claude Code to load the MCP server")
        console.print("  2. Run: simplemem verify")
        console.print("  3. In Claude Code, try: /memory-recall")


@app.command()
def backends():
    """List and compare available graph backends."""
    print_banner()

    table = Table(title="Graph Backend Comparison", box=box.ROUNDED)
    table.add_column("Feature", style="bold")
    table.add_column("FalkorDB", justify="center")
    table.add_column("KuzuDB", justify="center")

    features = [
        ("Requires Docker", "[yellow]Yes[/yellow]", "[green]No[/green]"),
        ("GLIBC Requirement", "2.35+", "2.17+"),
        ("Embedded Mode", "[red]No[/red]", "[green]Yes[/green]"),
        ("HPC Compatible", "[yellow]Via Apptainer[/yellow]", "[green]Yes[/green]"),
        ("Cypher Support", "[green]Full[/green]", "[green]Most[/green]"),
        ("shortestPath()", "[green]Yes[/green]", "[yellow]No*[/yellow]"),
        ("PageRank", "[green]Yes[/green]", "[red]No[/red]"),
        ("Schema Required", "[green]No[/green]", "[yellow]Yes[/yellow]"),
    ]

    for feature, falkor, kuzu in features:
        table.add_row(feature, falkor, kuzu)

    console.print(table)
    console.print("\n[dim]* Use variable-length paths with LIMIT 1 as alternative[/dim]")


@app.command()
def health():
    """Check health of all SimpleMem components."""
    print_banner()

    console.print("[bold]Checking components...[/bold]\n")

    checks = []

    # Check graph backend
    try:
        from simplemem_lite.db.graph_factory import get_backend_info

        info = get_backend_info()
        if info.get("active"):
            checks.append(("Graph Backend", True, info["active"]))
        else:
            checks.append(("Graph Backend", False, "No backend available"))
    except Exception as e:
        checks.append(("Graph Backend", False, str(e)))

    # Check LanceDB
    try:
        import lancedb

        checks.append(("LanceDB", True, "Available"))
    except ImportError:
        checks.append(("LanceDB", False, "Not installed"))

    # Check embedding model
    try:
        import sentence_transformers

        checks.append(("Embeddings", True, "sentence-transformers"))
    except ImportError:
        checks.append(("Embeddings", False, "sentence-transformers not installed"))

    # Check Claude skills
    skills_dir = Path.home() / ".claude" / "skills"
    if skills_dir.exists():
        skill_count = sum(1 for d in skills_dir.iterdir() if d.is_dir())
        checks.append(("Claude Skills", skill_count > 0, f"{skill_count} installed"))
    else:
        checks.append(("Claude Skills", False, "Not installed (run: simplemem install)"))

    # Display results
    table = Table(box=box.ROUNDED)
    table.add_column("Component", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    for name, ok, detail in checks:
        status = "[green]\u2713[/green]" if ok else "[red]\u2717[/red]"
        table.add_row(name, status, detail)

    console.print(table)

    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        console.print("\n[green]All components healthy![/green]")
    else:
        console.print("\n[yellow]Some components need attention.[/yellow]")

    return 0 if all_ok else 1


@app.command()
def verify(
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Show detailed output"),
):
    """Verify SimpleMem installation with detailed checks.

    Runs comprehensive verification of all components:
    - Core package imports
    - Optional dependencies
    - Database backends
    - Configuration
    - CLI entry points

    Use this after installation to ensure everything works.

    Example:
        simplemem verify
        simplemem verify --verbose
    """
    print_banner()
    console.print("[bold]Verifying SimpleMem Installation...[/bold]\n")

    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    def check_pass(name: str, detail: str = ""):
        nonlocal checks_passed
        checks_passed += 1
        if verbose:
            console.print(f"  [green]✓[/green] {name}" + (f" [dim]({detail})[/dim]" if detail else ""))

    def check_fail(name: str, detail: str = ""):
        nonlocal checks_failed
        checks_failed += 1
        console.print(f"  [red]✗[/red] {name}" + (f" [dim]({detail})[/dim]" if detail else ""))

    def check_warn(name: str, detail: str = ""):
        nonlocal checks_warned
        checks_warned += 1
        if verbose:
            console.print(f"  [yellow]○[/yellow] {name}" + (f" [dim]({detail})[/dim]" if detail else ""))

    # ===== Core Package =====
    console.print("[bold cyan]Core Package[/bold cyan]")

    try:
        import simplemem_lite
        check_pass("simplemem_lite package", f"v{getattr(simplemem_lite, '__version__', '0.1.0')}")
    except ImportError as e:
        check_fail("simplemem_lite package", str(e))

    try:
        from simplemem_lite.config import Config
        config = Config()
        check_pass("Config", f"data_dir={config.data_dir}")
    except Exception as e:
        check_fail("Config", str(e))

    try:
        from simplemem_lite.db import DatabaseManager
        check_pass("DatabaseManager")
    except Exception as e:
        check_fail("DatabaseManager", str(e))

    try:
        from simplemem_lite.memory import MemoryStore
        check_pass("MemoryStore")
    except Exception as e:
        check_fail("MemoryStore", str(e))

    try:
        from simplemem_lite.embeddings import embed, embed_batch
        check_pass("Embeddings module")
    except Exception as e:
        check_fail("Embeddings module", str(e))

    try:
        from simplemem_lite.traces import TraceParser, HierarchicalIndexer
        check_pass("Trace processing")
    except Exception as e:
        check_fail("Trace processing", str(e))

    # ===== Dependencies =====
    console.print("\n[bold cyan]Core Dependencies[/bold cyan]")

    deps = [
        ("mcp", "MCP protocol"),
        ("lancedb", "Vector database"),
        ("kuzu", "Embedded graph database"),
        ("litellm", "LLM API wrapper"),
        ("pyarrow", "Data serialization"),
        ("typer", "CLI framework"),
        ("rich", "Terminal output"),
        ("watchdog", "File watching"),
        ("loguru", "Logging"),
    ]

    for mod_name, desc in deps:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "")
            check_pass(f"{mod_name}", f"{desc}" + (f" v{ver}" if ver else ""))
        except ImportError:
            check_fail(f"{mod_name}", f"{desc} - NOT INSTALLED")

    # ===== Optional Dependencies =====
    console.print("\n[bold cyan]Optional Dependencies[/bold cyan]")

    try:
        import falkordb
        ver = getattr(falkordb, "__version__", "")
        check_pass("falkordb", f"FalkorDB client" + (f" v{ver}" if ver else ""))
    except ImportError:
        check_warn("falkordb", "Not installed (install with: uv pip install -e '.[falkordb]')")

    try:
        import sentence_transformers
        ver = getattr(sentence_transformers, "__version__", "")
        check_pass("sentence-transformers", f"Local embeddings" + (f" v{ver}" if ver else ""))
    except ImportError:
        check_warn("sentence-transformers", "Not installed (install with: uv pip install -e '.[local]')")

    # ===== Graph Backends =====
    console.print("\n[bold cyan]Graph Backends[/bold cyan]")

    try:
        from simplemem_lite.db.graph_factory import get_backend_info
        info = get_backend_info()

        for backend in ["kuzu", "falkordb"]:
            bi = info.get(backend, {})
            if bi.get("available"):
                check_pass(f"{backend.upper()}", "Available and working")
            elif bi.get("installed"):
                check_warn(f"{backend.upper()}", f"Installed but not available: {bi.get('error', 'unknown')}")
            else:
                check_warn(f"{backend.upper()}", "Not installed")

        active = info.get("active")
        if active:
            console.print(f"  [bold]Active backend:[/bold] [green]{active}[/green]")
        else:
            check_fail("No active graph backend", "At least one backend required")
    except Exception as e:
        check_fail("Graph backend check", str(e))

    # ===== Container Runtimes =====
    console.print("\n[bold cyan]Container Runtimes (for FalkorDB)[/bold cyan]")

    has_apptainer, apptainer_path = _check_apptainer()
    if has_apptainer:
        check_pass("Apptainer", apptainer_path)
    else:
        check_warn("Apptainer", "Not found (optional, for HPC)")

    has_docker, docker_path = _check_docker()
    if has_docker:
        check_pass("Docker", docker_path)
    elif docker_path:
        check_warn("Docker", "Installed but not running")
    else:
        check_warn("Docker", "Not found (optional)")

    # ===== FalkorDB Status =====
    is_running, method = _is_falkordb_running()
    if is_running:
        check_pass(f"FalkorDB server", f"Running via {method} on port {FALKORDB_PORT}")
    else:
        check_warn("FalkorDB server", "Not running (start with: simplemem start)")

    # ===== CLI Entry Points =====
    console.print("\n[bold cyan]CLI Entry Points[/bold cyan]")

    import shutil
    for cmd in ["simplemem", "simplemem-lite"]:
        path = shutil.which(cmd)
        if path:
            check_pass(cmd, path)
        else:
            check_fail(cmd, "Not found in PATH")

    # ===== Directories =====
    console.print("\n[bold cyan]Directories[/bold cyan]")

    from simplemem_lite.config import Config
    config = Config()

    dirs_to_check = [
        ("Data directory", Path(config.data_dir)),
        ("Claude traces", Path(config.claude_traces_dir)),
        ("User config", Path.home() / ".simplemem"),
    ]

    for name, path in dirs_to_check:
        if path.exists():
            check_pass(name, str(path))
        else:
            check_warn(name, f"{path} (will be created on first use)")

    # ===== Summary =====
    console.print("\n" + "=" * 50)
    total = checks_passed + checks_failed + checks_warned
    console.print(f"\n[bold]Verification Complete[/bold]")
    console.print(f"  [green]Passed:[/green]  {checks_passed}")
    console.print(f"  [red]Failed:[/red]   {checks_failed}")
    console.print(f"  [yellow]Warnings:[/yellow] {checks_warned}")

    if checks_failed == 0:
        console.print("\n[green bold]✓ Installation verified successfully![/green bold]")
        if checks_warned > 0:
            console.print("[dim]Some optional components are not installed.[/dim]")
        return 0
    else:
        console.print("\n[red bold]✗ Installation has issues that need attention.[/red bold]")
        console.print("\n[bold]Troubleshooting:[/bold]")
        console.print("  1. Reinstall: uv pip install -e . -v")
        console.print("  2. Check Python version: python --version (need 3.11+)")
        console.print("  3. Check logs: ~/.simplemem/logs/")
        return 1


@app.command()
def version():
    """Show SimpleMem version."""
    try:
        from simplemem_lite import __version__
        version_str = __version__
    except (ImportError, AttributeError):
        version_str = "0.1.0-dev"

    console.print(f"SimpleMem Lite [cyan]{version_str}[/cyan]")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
