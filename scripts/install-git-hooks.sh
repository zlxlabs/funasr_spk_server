#!/usr/bin/env bash
# 安装版本控制的 git hooks (P4 A4 / P3 质量闸门)
# 用 core.hooksPath 指向 repo 内的 scripts/git-hooks, 让 hook 进版本控制、团队共享。
#
# 用法:  bash scripts/install-git-hooks.sh
# 卸载:  git config --unset core.hooksPath
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if global_hooks_path="$(git config --global --get core.hooksPath)"; then
  echo "检测到全局 core.hooksPath 已设置: ${global_hooks_path}" >&2
  echo "为避免停用全局 pre-push 与公开仓扫描，拒绝写入本仓 hooksPath。" >&2
  echo '如需同时运行，可在全局 hook 中显式委托：repo_root=$(git rev-parse --show-toplevel); exec "$repo_root/scripts/git-hooks/pre-push" "$@"' >&2
  exit 1
fi

chmod +x scripts/git-hooks/* 2>/dev/null || true
git config core.hooksPath scripts/git-hooks

echo "✓ 已设 core.hooksPath = scripts/git-hooks"
echo "  pre-push 现在会在 push 前跑 unit (碰 FunASR 路径再追加 parity)。"
echo "  紧急放行: FUNASR_SKIP_PREPUSH=1 git push"
echo "  卸载:     git config --unset core.hooksPath"
