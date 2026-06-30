"""Docker environment adapted from upstream ``src/minisweagent/environments/docker.py``.

Local modifications relative to the upstream mini-swe-agent implementation:
- uses ``dataclass`` configuration helpers instead of the upstream Pydantic config model;
- adds explicit startup error reporting for ``docker run`` failures and pull timeouts;
- keeps the execution shell fixed to ``bash -lc`` instead of exposing a configurable interpreter;
- returns synchronous cleanup behavior with timeout handling to reduce leaked containers.
"""

import logging
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.pull_timeout,  # docker pull might take a while
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or "no stderr/stdout captured"
            msg = (
                f"Docker container startup failed (exit={exc.returncode}) for image '{self.config.image}'. "
                f"Command: {' '.join(cmd)}. Details: {details}"
            )
            raise RuntimeError(msg) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Docker container startup timed out after {self.config.pull_timeout}s for image "
                f"'{self.config.image}'. Command: {' '.join(cmd)}"
            ) from exc
        self.container_id = result.stdout.strip()

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop and remove the Docker container synchronously to prevent accumulation."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            try:
                # Synchronously stop and remove container with timeout
                # This ensures containers are properly cleaned up before processing next instance
                cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) 2>/dev/null"
                subprocess.run(cmd, shell=True, timeout=65, capture_output=True)
            except subprocess.TimeoutExpired:
                # If cleanup times out, force remove the container
                try:
                    subprocess.run(
                        f"{self.config.executable} rm -f {self.container_id} 2>/dev/null",
                        shell=True,
                        timeout=10,
                        capture_output=True
                    )
                except Exception as e:
                    self.logger.error(f"Failed to remove container {self.container_id[:8]}: {e}")
            except Exception as e:
                self.logger.error(f"Error cleaning up container {self.container_id[:8]}: {e}")

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
