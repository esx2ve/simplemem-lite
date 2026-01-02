"""SimpleMem Lite CLI with beautiful Rich output.

Provides commands for:
- Backend status and info
- Memory search and storage
- Project management
- User-space installation

Usage:
    simplemem status           # Show backend status
    simplemem search "query"   # Search memories
    simplemem install          # Install to ~/.claude
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Initialize Typer app and Rich console
app = typer.Typer(
    name="simplemem",
    help="SimpleMem Lite - Long-term memory for Claude Code",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


def print_banner():
    """Print SimpleMem banner."""
    banner = Text()
    banner.append("SimpleMem", style="bold cyan")
    banner.append(" Lite", style="cyan")
    console.print(Panel(banner, border_style="cyan", box=box.ROUNDED))


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
):
    """Install SimpleMem skills and agents to user space.

    Copies skills to ~/.claude/skills/ and agents to ~/.claude/agents/
    for Claude Code integration.
    """
    import shutil

    print_banner()

    # Find source directories
    src_root = Path(__file__).parent.parent.parent  # simplemem repo root
    skills_src = src_root / ".claude" / "skills"
    agents_src = src_root / ".claude" / "agents"

    # Target directories
    home = Path.home()
    skills_dst = home / ".claude" / "skills"
    agents_dst = home / ".claude" / "agents"

    items_to_install = []

    # Collect skills
    if skills_src.exists():
        for skill_dir in skills_src.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                items_to_install.append({
                    "type": "skill",
                    "name": skill_dir.name,
                    "src": skill_dir,
                    "dst": skills_dst / skill_dir.name,
                })

    # Collect agents
    if agents_src.exists():
        for agent_file in agents_src.glob("*.md"):
            items_to_install.append({
                "type": "agent",
                "name": agent_file.stem,
                "src": agent_file,
                "dst": agents_dst / agent_file.name,
            })

    if not items_to_install:
        console.print("[yellow]No skills or agents found to install.[/yellow]")
        return

    # Show what will be installed
    table = Table(title="Items to Install", box=box.ROUNDED)
    table.add_column("Type", style="cyan")
    table.add_column("Name")
    table.add_column("Status")

    for item in items_to_install:
        exists = item["dst"].exists()
        if exists and not force:
            status = "[yellow]Exists (use --force)[/yellow]"
        elif exists and force:
            status = "[yellow]Will overwrite[/yellow]"
        else:
            status = "[green]Will install[/green]"
        table.add_row(item["type"].capitalize(), item["name"], status)

    console.print(table)

    if dry_run:
        console.print("\n[dim]Dry run - no changes made.[/dim]")
        return

    # Create directories
    skills_dst.mkdir(parents=True, exist_ok=True)
    agents_dst.mkdir(parents=True, exist_ok=True)

    # Install items
    installed = 0
    skipped = 0

    for item in items_to_install:
        if item["dst"].exists() and not force:
            skipped += 1
            continue

        try:
            if item["src"].is_dir():
                if item["dst"].exists():
                    shutil.rmtree(item["dst"])
                shutil.copytree(item["src"], item["dst"])
            else:
                shutil.copy2(item["src"], item["dst"])
            installed += 1
            console.print(f"  [green]\u2713[/green] {item['type']}: {item['name']}")
        except Exception as e:
            console.print(f"  [red]\u2717[/red] {item['type']}: {item['name']} - {e}")

    console.print(f"\n[green]Installed: {installed}[/green], [yellow]Skipped: {skipped}[/yellow]")


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
