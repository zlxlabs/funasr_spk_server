"""进程级启动失败契约：异常必须传播为非零退出，且不能发正常关闭通知。"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_original_notification_enabled = os.environ.get("FUNASR_NOTIFICATION_ENABLED")
os.environ["FUNASR_NOTIFICATION_ENABLED"] = "false"
import src.core.config  # noqa: F401 - 在 unit fixture 导入前以隔离配置初始化全局单例
if _original_notification_enabled is None:
    os.environ.pop("FUNASR_NOTIFICATION_ENABLED", None)
else:
    os.environ["FUNASR_NOTIFICATION_ENABLED"] = _original_notification_enabled


def test_startup_failure_exits_nonzero_without_normal_shutdown_notification():
    """经真实 Python 子进程和 main() 公开入口验证启动异常行为。"""
    project_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import asyncio
        import os
        import signal
        os.environ["FUNASR_NOTIFICATION_ENABLED"] = "false"
        import src.main as server_main

        notifications = []
        async def fail_start(self):
            raise RuntimeError("injected startup failure")
        async def stop_tasks():
            return None
        async def capture_notification(title, *args):
            notifications.append(title)

        server_main.signal.signal = lambda *args: None
        server_main.log_platform_info = lambda: None
        server_main.check_system_requirements = lambda: None
        server_main.FunASRServer.start = fail_start
        server_main.task_manager.stop = stop_tasks
        server_main.send_custom_notification = capture_notification

        try:
            asyncio.run(server_main.main())
        except RuntimeError:
            normal_shutdown_count = sum("已停止" in title for title in notifications)
            print(f"STARTUP_EXCEPTION_PROPAGATED=1 NORMAL_SHUTDOWN_COUNT={normal_shutdown_count}")
            raise
        else:
            normal_shutdown_count = sum("已停止" in title for title in notifications)
            print(f"STARTUP_EXCEPTION_PROPAGATED=0 NORMAL_SHUTDOWN_COUNT={normal_shutdown_count}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "STARTUP_EXCEPTION_PROPAGATED=1 NORMAL_SHUTDOWN_COUNT=0" in result.stdout
