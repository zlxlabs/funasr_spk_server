## 2026-10-21 调研与 Linux 复现完成

- 当前阶段：调研完成，正在执行范围/文档验收并准备提交。
- 本段结论：13 条均已按 `3fd1b2a` 重新定位并写入 `docs/sessions/261021-review-fixes-b4/research.md`；11 条成立、1 条不成立、R5 无法判定。R2/R3 的 P1 风险在 Linux 故障探针中复现；Mac/CUDA 与真实模型行为注明不可本机验证。
- 关键决策与已否决方案：严格只读，不改 `src/`、`tests/`；不把缺失 `venv`/依赖解释成测试通过，也不为达到测试绿在当前执行环境安装依赖或伪造真实硬件验证。
- 下一步唯一动作：执行指定 scope/narrow/full-check 判据，记录阻塞后提交并推送两份文档。

## 2026-10-21 文档验收与提交前范围复核

- 当前阶段：研究报告已提交；准备提交范围复核记录并推送。
- 本段结论：`test -s docs/sessions/261021-review-fixes-b4/research.md` 通过；`bash scripts/run_checks.sh` 因缺少 `venv/bin/python` 退出 127；pytest 另因系统 Python 缺 `loguru` 无法收集。首轮 scope checker 在 HEAD 等于主干时提示 fork point 歧义；提交研究文档后、最终进度提交前复跑通过，2 个已提交文件都在 Scope-Globs 内。
- 关键决策与已否决方案：不把环境缺依赖当作继承红或新代码红；不安装依赖、不改业务/测试文件、不扩大只读调研范围。
- 下一步唯一动作：提交本段进度，推送卡分支，并复核远端提交与干净工作树。
