# Stage1 实现冻结：stage1-20261102-r1

本版冻结路网、速度、building with units、depot、charging station 的构造代码、默认参数、方法文档和输入身份。11城完整CLE由服务器按本版生成，数据验收记录与本实现冻结分别保存；Stage2不在包内。

- [服务器代理任务书](../../docs/stage1/SERVER_AGENT_HANDOFF.zh-CN.md)
- [输入布局与搬运范围](INPUT_LAYOUT.zh-CN.md)
- [服务器配置模板](stage1.server.json)
- [输入文件及SHA-256](input_manifest.json)
- [本机验证范围](local_validation.json)
- [本机实际环境](environment.json)

冻结代码以 `MANIFEST.json` 的逐文件哈希为准，压缩包身份记录于本机交接目录的 `BUNDLE.json` 及 `dist/stage1-20261102-r1.sha256`。源码包含本轮工作区实现；Git HEAD仅作来源参考。

服务器解压代码包、按输入布局迁移数据、安装环境后，在项目根执行：

```bash
bash freeze/20261102-r1/run_stage1.sh
```

需要修改实际输入或输出路径时，先在同目录复制模板为 `stage1.local.json`，只编辑该副本，再把路径作为脚本参数。方法参数、输入字节和冻结源文件保持不变。脚本一次完成11城校准，逐城构造并验证CLE；`STAGE1_FROM` 支持阶段恢复。

本机建筑属性参考：3,922,741 / 4,012,462，覆盖率97.7639%。多户建筑均通过可评估的面积／层数代理筛查。该比例不是道路接入后的最终CLE地点比例；服务器按实际构造产物另行报告。
