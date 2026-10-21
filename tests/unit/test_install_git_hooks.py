"""安装仓库 hooks 时隔离并验证全局 Git 配置的所有权边界。"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "install-git-hooks.sh"
HOOKS = ROOT / "scripts" / "git-hooks"
os.environ.setdefault("FUNASR_NOTIFICATION_ENABLED", "false")


def _temporary_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """构造隔离 HOME/GIT_CONFIG_GLOBAL 的临时仓库和安装脚本现场。"""
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    shutil.copy2(INSTALLER, repo / "scripts" / "install-git-hooks.sh")
    shutil.copytree(HOOKS, repo / "scripts" / "git-hooks")
    global_config = tmp_path / "global.gitconfig"
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(global_config),
        "GIT_CONFIG_NOSYSTEM": "1",
    })
    for key in list(env):
        if key.startswith("GIT_CONFIG_") and key not in {
            "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM"
        }:
            env.pop(key)
    subprocess.run(
        ["git", "init", "--quiet", "--template="],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo, env


def _git_config(repo: Path, env: dict[str, str], scope: str, key: str) -> str | None:
    result = subprocess.run(
        ["git", "config", scope, "--get", key],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def test_global_hooks_path_refuses_to_override_repo_setting(tmp_path):
    """设置全局 hooksPath 时安装必须非零退出且不写本仓 hooksPath。"""
    repo, env = _temporary_repo(tmp_path)
    global_hooks = tmp_path / "global-hooks"
    global_hooks.mkdir()
    subprocess.run(
        ["git", "config", "--global", "core.hooksPath", str(global_hooks)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        ["bash", "scripts/install-git-hooks.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert _git_config(repo, env, "--local", "core.hooksPath") is None
    assert _git_config(repo, env, "--global", "core.hooksPath") == str(global_hooks)
    assert "全局" in result.stderr
    assert "scripts/git-hooks" in result.stderr


def test_unset_global_hooks_path_installs_repo_hooks(tmp_path):
    """未设置全局 hooksPath 时安装仍写入本仓配置并成功。"""
    repo, env = _temporary_repo(tmp_path)

    result = subprocess.run(
        ["bash", "scripts/install-git-hooks.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert _git_config(repo, env, "--local", "core.hooksPath") == "scripts/git-hooks"
    assert _git_config(repo, env, "--global", "core.hooksPath") is None
