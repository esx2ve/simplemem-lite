"""Project bootstrap and detection pipeline for SimpleMem Lite.

Detects project type and metadata, then bootstraps code indexing and watcher.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm

from simplemem_lite.code_index import CodeIndexer
from simplemem_lite.config import Config
from simplemem_lite.logging import get_logger
from simplemem_lite.projects import ProjectManager, ProjectState
from simplemem_lite.watcher import ProjectWatcherManager

log = get_logger("bootstrap")


@dataclass
class ProjectInfo:
    """Detected project information."""

    name: str | None = None
    project_type: str | None = None  # python, typescript, javascript, rust, go, etc.
    description: str | None = None
    frameworks: list[str] | None = None
    confidence: float = 0.0  # 0-1 confidence in detection
    source: str = "unknown"  # Which provider detected this


class ProjectInfoProvider(ABC):
    """Abstract base class for project info detection."""

    @abstractmethod
    def detect(self, project_root: Path) -> ProjectInfo | None:
        """Detect project info.

        Args:
            project_root: Path to project root

        Returns:
            ProjectInfo if detection succeeded, None otherwise
        """
        pass


class ProjectStructureProvider(ProjectInfoProvider):
    """Fast path: detect project from config files."""

    # Config file patterns and their handlers
    CONFIG_FILES = [
        ("pyproject.toml", "_parse_pyproject"),
        ("package.json", "_parse_package_json"),
        ("Cargo.toml", "_parse_cargo"),
        ("go.mod", "_parse_go_mod"),
        ("pom.xml", "_parse_pom"),
        ("build.gradle", "_parse_gradle"),
        ("setup.py", "_parse_setup_py"),
    ]

    def detect(self, project_root: Path) -> ProjectInfo | None:
        """Detect project from config files."""
        for filename, parser_method in self.CONFIG_FILES:
            config_path = project_root / filename
            if config_path.exists():
                parser = getattr(self, parser_method)
                try:
                    info = parser(config_path)
                    if info:
                        info.source = f"config:{filename}"
                        log.info(f"Detected project from {filename}: {info.name}")
                        return info
                except Exception as e:
                    log.debug(f"Failed to parse {filename}: {e}")
        return None

    def _parse_pyproject(self, path: Path) -> ProjectInfo | None:
        """Parse pyproject.toml."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        content = path.read_text()
        data = tomllib.loads(content)

        project = data.get("project", {})
        name = project.get("name")
        description = project.get("description")

        # Detect frameworks from dependencies
        deps = project.get("dependencies", [])
        frameworks = []
        if any("fastapi" in str(d).lower() for d in deps):
            frameworks.append("fastapi")
        if any("django" in str(d).lower() for d in deps):
            frameworks.append("django")
        if any("flask" in str(d).lower() for d in deps):
            frameworks.append("flask")
        if any("pytorch" in str(d).lower() or "torch" in str(d).lower() for d in deps):
            frameworks.append("pytorch")

        return ProjectInfo(
            name=name,
            project_type="python",
            description=description,
            frameworks=frameworks if frameworks else None,
            confidence=0.9 if name else 0.7,
        )

    def _parse_package_json(self, path: Path) -> ProjectInfo | None:
        """Parse package.json."""
        data = json.loads(path.read_text())
        name = data.get("name")
        description = data.get("description")

        # Detect frameworks
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        frameworks = []
        if "react" in deps:
            frameworks.append("react")
        if "vue" in deps:
            frameworks.append("vue")
        if "next" in deps:
            frameworks.append("nextjs")
        if "express" in deps:
            frameworks.append("express")
        if "typescript" in deps:
            frameworks.append("typescript")

        project_type = "typescript" if "typescript" in deps else "javascript"

        return ProjectInfo(
            name=name,
            project_type=project_type,
            description=description,
            frameworks=frameworks if frameworks else None,
            confidence=0.9 if name else 0.7,
        )

    def _parse_cargo(self, path: Path) -> ProjectInfo | None:
        """Parse Cargo.toml."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        content = path.read_text()
        data = tomllib.loads(content)

        package = data.get("package", {})
        name = package.get("name")
        description = package.get("description")

        return ProjectInfo(
            name=name,
            project_type="rust",
            description=description,
            confidence=0.9 if name else 0.7,
        )

    def _parse_go_mod(self, path: Path) -> ProjectInfo | None:
        """Parse go.mod."""
        content = path.read_text()
        match = re.search(r"^module\s+(\S+)", content, re.MULTILINE)
        if match:
            module_path = match.group(1)
            name = module_path.split("/")[-1]
            return ProjectInfo(
                name=name,
                project_type="go",
                confidence=0.8,
            )
        return None

    def _parse_pom(self, path: Path) -> ProjectInfo | None:
        """Parse pom.xml (basic)."""
        content = path.read_text()
        name_match = re.search(r"<artifactId>([^<]+)</artifactId>", content)
        name = name_match.group(1) if name_match else None

        return ProjectInfo(
            name=name,
            project_type="java",
            confidence=0.8 if name else 0.6,
        )

    def _parse_gradle(self, path: Path) -> ProjectInfo | None:
        """Parse build.gradle (basic)."""
        content = path.read_text()
        # Try to find project name
        name_match = re.search(r"rootProject\.name\s*=\s*['\"]([^'\"]+)['\"]", content)
        name = name_match.group(1) if name_match else None

        # Check for Kotlin
        project_type = "kotlin" if "kotlin" in content.lower() else "java"

        return ProjectInfo(
            name=name,
            project_type=project_type,
            confidence=0.7 if name else 0.5,
        )

    def _parse_setup_py(self, path: Path) -> ProjectInfo | None:
        """Parse setup.py (basic regex)."""
        content = path.read_text()
        name_match = re.search(r"name\s*=\s*['\"]([^'\"]+)['\"]", content)
        name = name_match.group(1) if name_match else None

        return ProjectInfo(
            name=name,
            project_type="python",
            confidence=0.7 if name else 0.5,
        )


class ReadmeProvider(ProjectInfoProvider):
    """Slow path: extract project info from README using LLM."""

    # README filenames to check
    README_FILES = ["README.md", "README.rst", "README.txt", "README"]

    def __init__(self, config: Config):
        """Initialize with config for LLM access."""
        self.config = config
        self._cache: dict[str, ProjectInfo] = {}

    def detect(self, project_root: Path) -> ProjectInfo | None:
        """Detect project from README using LLM summarization."""
        cache_key = str(project_root)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Find README
        readme_content = None
        readme_file = None
        for filename in self.README_FILES:
            readme_path = project_root / filename
            if readme_path.exists():
                try:
                    readme_content = readme_path.read_text()[:4000]  # Limit size
                    readme_file = filename
                    break
                except Exception as e:
                    log.debug(f"Failed to read {filename}: {e}")

        if not readme_content:
            return None

        # Use LLM to extract info
        try:
            info = self._extract_with_llm(readme_content, readme_file)
            if info:
                self._cache[cache_key] = info
            return info
        except Exception as e:
            log.warning(f"LLM extraction failed: {e}")
            return None

    def _extract_with_llm(self, readme_content: str, filename: str) -> ProjectInfo | None:
        """Extract project info using LLM."""
        prompt = f"""Extract project information from this README. Return JSON with these fields:
- name: project name (string or null)
- project_type: primary language/platform (python, typescript, javascript, rust, go, java, etc.) or null
- description: one-sentence description or null
- frameworks: list of frameworks/libraries mentioned (e.g., ["fastapi", "react"]) or null

README content:
{readme_content}

Respond with ONLY valid JSON, no markdown formatting."""

        response = litellm.completion(
            model=self.config.summary_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )

        result = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Try to extract JSON from response
            import json_repair

            data = json_repair.loads(result)

            return ProjectInfo(
                name=data.get("name"),
                project_type=data.get("project_type"),
                description=data.get("description"),
                frameworks=data.get("frameworks"),
                confidence=0.6,  # LLM extraction is less reliable
                source=f"readme:{filename}",
            )
        except Exception as e:
            log.debug(f"Failed to parse LLM response: {e}")
            return None


class Bootstrap:
    """Orchestrates project detection and bootstrap."""

    def __init__(
        self,
        config: Config,
        project_manager: ProjectManager,
        code_indexer: CodeIndexer,
        watcher_manager: ProjectWatcherManager,
    ):
        """Initialize bootstrap orchestrator.

        Args:
            config: SimpleMem configuration
            project_manager: Project state manager
            code_indexer: Code indexer
            watcher_manager: File watcher manager
        """
        self.config = config
        self.project_manager = project_manager
        self.code_indexer = code_indexer
        self.watcher_manager = watcher_manager

        # Detection pipeline (ordered by speed/reliability)
        self._providers: list[ProjectInfoProvider] = [
            ProjectStructureProvider(),
            ReadmeProvider(config),
        ]

        log.info("Bootstrap orchestrator initialized")

    def _get_hooks_dir(self) -> Path | None:
        """Get the hooks directory path.

        Returns:
            Path to hooks directory, or None if not found
        """
        # Hooks are in the parent of the package directory
        package_dir = Path(__file__).parent
        hooks_dir = package_dir.parent / "hooks"
        if hooks_dir.exists():
            return hooks_dir

        # Fallback: check if installed as package
        import importlib.resources

        try:
            # For installed packages, hooks might be in package data
            with importlib.resources.files("simplemem_lite").joinpath("../hooks") as p:
                if p.exists():
                    return Path(p)
        except Exception:
            pass

        return None

    def _check_hooks_installed(self) -> dict[str, Any]:
        """Check if hooks are installed in Claude Code settings.

        Returns:
            Dict with:
            - installed: bool
            - settings_file: str (path to settings file)
            - session_start_hook: bool
            - stop_hook: bool
        """
        settings_file = Path.home() / ".claude" / "settings.json"
        result = {
            "installed": False,
            "settings_file": str(settings_file),
            "session_start_hook": False,
            "stop_hook": False,
        }

        if not settings_file.exists():
            return result

        try:
            settings = json.loads(settings_file.read_text())
            hooks = settings.get("hooks", {})

            # Check for simplemem hooks
            session_start = hooks.get("SessionStart", [])
            stop = hooks.get("Stop", [])

            result["session_start_hook"] = any("simplemem" in str(h).lower() for h in session_start)
            result["stop_hook"] = any("simplemem" in str(h).lower() for h in stop)
            result["installed"] = result["session_start_hook"] and result["stop_hook"]

        except Exception as e:
            log.debug(f"Failed to read Claude settings: {e}")

        return result

    def _install_hooks(self) -> dict[str, Any]:
        """Install hooks to Claude Code settings.

        Returns:
            Dict with:
            - success: bool
            - message: str
            - hooks_installed: list[str]
            - restart_required: bool
        """
        hooks_dir = self._get_hooks_dir()
        if hooks_dir is None:
            return {
                "success": False,
                "message": "Hooks directory not found",
                "hooks_installed": [],
                "restart_required": False,
            }

        session_start_hook = hooks_dir / "session-start.sh"
        stop_hook = hooks_dir / "stop.sh"

        if not session_start_hook.exists() or not stop_hook.exists():
            return {
                "success": False,
                "message": "Hook scripts not found",
                "hooks_installed": [],
                "restart_required": False,
            }

        # Make hooks executable
        import stat
        for hook in [session_start_hook, stop_hook]:
            hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        settings_file = Path.home() / ".claude" / "settings.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or create settings
        if settings_file.exists():
            try:
                settings = json.loads(settings_file.read_text())
            except Exception:
                settings = {}
        else:
            settings = {}

        # Initialize hooks structure
        if "hooks" not in settings:
            settings["hooks"] = {}
        if "SessionStart" not in settings["hooks"]:
            settings["hooks"]["SessionStart"] = []
        if "Stop" not in settings["hooks"]:
            settings["hooks"]["Stop"] = []

        hooks_installed = []
        already_installed = []

        # Add session start hook if not present
        session_start_path = str(session_start_hook)
        if not any("simplemem" in str(h).lower() for h in settings["hooks"]["SessionStart"]):
            settings["hooks"]["SessionStart"].append(session_start_path)
            hooks_installed.append("SessionStart")
        else:
            already_installed.append("SessionStart")

        # Add stop hook if not present
        stop_path = str(stop_hook)
        if not any("simplemem" in str(h).lower() for h in settings["hooks"]["Stop"]):
            settings["hooks"]["Stop"].append(stop_path)
            hooks_installed.append("Stop")
        else:
            already_installed.append("Stop")

        # Write settings
        try:
            settings_file.write_text(json.dumps(settings, indent=2))
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to write settings: {e}",
                "hooks_installed": [],
                "restart_required": False,
            }

        if hooks_installed:
            return {
                "success": True,
                "message": f"Installed hooks: {', '.join(hooks_installed)}. Restart Claude Code to activate.",
                "hooks_installed": hooks_installed,
                "already_installed": already_installed,
                "restart_required": True,
            }
        else:
            return {
                "success": True,
                "message": "Hooks already installed",
                "hooks_installed": [],
                "already_installed": already_installed,
                "restart_required": False,
            }

    def detect_project_info(self, project_root: str) -> ProjectInfo:
        """Detect project info using the provider pipeline.

        Args:
            project_root: Absolute path to project root

        Returns:
            ProjectInfo (may have null fields if detection failed)
        """
        root = Path(project_root)

        for provider in self._providers:
            try:
                info = provider.detect(root)
                if info and info.confidence >= 0.5:
                    log.info(f"Project detected via {provider.__class__.__name__}: {info.name}")
                    return info
            except Exception as e:
                log.warning(f"Provider {provider.__class__.__name__} failed: {e}")

        # Return minimal info if all providers fail
        log.info(f"No project info detected for: {project_root}")
        return ProjectInfo(
            name=root.name,
            source="fallback:dirname",
            confidence=0.1,
        )

    def bootstrap_project(
        self,
        project_root: str,
        index_code: bool = True,
        start_watcher: bool = True,
    ) -> dict[str, Any]:
        """Bootstrap a project.

        Args:
            project_root: Absolute path to project root
            index_code: Whether to index code files
            start_watcher: Whether to start file watcher

        Returns:
            Dict with bootstrap results
        """
        log.info(f"Bootstrapping project: {project_root}")
        results: dict[str, Any] = {
            "project_root": project_root,
            "success": False,
        }

        # Detect project info
        info = self.detect_project_info(project_root)
        results["project_info"] = {
            "name": info.name,
            "type": info.project_type,
            "description": info.description,
            "frameworks": info.frameworks,
            "source": info.source,
        }

        # Index code
        if index_code:
            try:
                index_result = self.code_indexer.index_directory(project_root)
                results["index"] = {
                    "files_indexed": index_result.get("files_indexed", 0),
                    "chunks_created": index_result.get("chunks_created", 0),
                }
                log.info(f"Indexed {results['index']['files_indexed']} files")
            except Exception as e:
                log.error(f"Code indexing failed: {e}")
                results["index"] = {"error": str(e)}

        # Start watcher
        if start_watcher:
            try:
                watcher_result = self.watcher_manager.start_watching(project_root)
                results["watcher"] = watcher_result
                log.info(f"Watcher started: {watcher_result.get('status')}")
            except Exception as e:
                log.error(f"Watcher start failed: {e}")
                results["watcher"] = {"error": str(e)}

        # Install hooks if not already installed
        hook_status = self._check_hooks_installed()
        if not hook_status["installed"]:
            hook_result = self._install_hooks()
            results["hooks"] = hook_result
            if hook_result["restart_required"]:
                log.info("Hooks installed - Claude Code restart required")
        else:
            results["hooks"] = {
                "success": True,
                "message": "Hooks already installed",
                "hooks_installed": [],
                "restart_required": False,
            }

        # Mark project as bootstrapped
        state = self.project_manager.mark_bootstrapped(
            project_root=project_root,
            project_name=info.name,
            project_type=info.project_type,
            description=info.description,
        )
        results["project_state"] = {
            "is_bootstrapped": state.is_bootstrapped,
            "project_name": state.project_name,
        }
        results["success"] = True
        results["restart_required"] = results["hooks"].get("restart_required", False)

        log.info(f"Bootstrap complete for: {project_root}")
        return results

    def get_bootstrap_status(self, project_root: str) -> dict[str, Any]:
        """Get bootstrap status for a project.

        Args:
            project_root: Absolute path to project root

        Returns:
            Dict with status info
        """
        state = self.project_manager.get_project_state(project_root)
        is_watching = self.watcher_manager.is_watching(project_root)

        if state is None:
            return {
                "project_root": project_root,
                "is_known": False,
                "is_bootstrapped": False,
                "is_watching": is_watching,
                "should_ask": True,
            }

        return {
            "project_root": project_root,
            "is_known": True,
            "is_bootstrapped": state.is_bootstrapped,
            "never_ask": state.never_ask,
            "project_name": state.project_name,
            "project_type": state.project_type,
            "is_watching": is_watching,
            "should_ask": self.project_manager.should_ask_bootstrap(project_root),
            "updated_at": state.updated_at,
        }

    def generate_context_injection(self, project_root: str) -> str | None:
        """Generate context to inject into Claude session.

        Called by session-start hook to provide project context.

        Args:
            project_root: Absolute path to project root

        Returns:
            Context string for Claude, or None if no context
        """
        state = self.project_manager.get_project_state(project_root)

        if state is None or not state.is_bootstrapped:
            # Not bootstrapped - suggest bootstrapping
            if state is None or not state.never_ask:
                return (
                    f"[SimpleMem] Project at {project_root} has not been bootstrapped. "
                    "Use `bootstrap_project` to enable code search and memory features."
                )
            return None

        # Bootstrapped - provide context
        parts = [f"[SimpleMem] Project: {state.project_name or project_root}"]
        if state.project_type:
            parts.append(f"Type: {state.project_type}")
        if state.description:
            parts.append(f"Description: {state.description}")
        parts.append("Code search and file watching are active.")

        return " | ".join(parts)
