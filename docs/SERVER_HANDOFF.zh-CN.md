# Stage 1 与 Stage 2 联合审阅

这份交接先供项目内部审阅。服务器执行时，先用 Stage 1 的独立冻结归档完成 11 城 CLE，再用审阅通过的 Stage 2 源码从已核验的 CLE 生成 Dallas Cus100。后续规模和批次依据这次的实际结果确定。

## 两阶段的输入和交付

| 阶段 | 使用的版本 | 输入与交付 |
| --- | --- | --- |
| Stage 1 | 独立归档 `stage1-20261102-r1.tar.gz` | 从指定的 116 个外部源文件生成 11 城 CLE。一次联合校准，再逐城组装。交付全部城市 manifest、核验记录、覆盖统计及批次回执。 |
| Stage 2 | 联合审阅包的源码或对应 Git 提交 | 读取通过核验的城市 CLE、TIGER BG 和 Amazon 订单来源。交付实例、选区、订单、选站明细、两套有向道路成本以及验证和导出结果。 |

Stage 1 的独立归档身份由同名 `.sha256` 文件和归档内的逐文件 manifest 给出。它包含 Stage 1 专用安装配置。联合审阅源码包含两个 CLI，使用单独的包清单，不以 Stage 1 冻结清单验收这一层。实际原始数据和已生成的 CLE 通过数据清单独立准备。

## 服务器按顺序执行

1. 解压 Stage 1 原始归档。按[冻结交接说明](stage1/SERVER_AGENT_HANDOFF.zh-CN.md)迁移 116 项输入并核对 SHA-256。保持方法和参数，按服务器路径编辑配置副本。用 `freeze/20261102-r1/run_stage1.sh` 完成全体城市的统一校准与验收。
2. 保留 Stage 1 输出结构 `<stage1-output>/cities/<city>`。项目输出根目录的 `stage1_server_build_receipt.json` 必须通过全体城市核验，并绑定当前归档版本和每城 `cle_manifest.json`。每城的 `stage1_validation.json` 同样随结果保留。Stage 2 从这些记录核对输入身份。
3. 安装联合审阅源码。从[Stage 2 运行说明](stage2/REVIEW_PACKAGE.zh-CN.md)准备 Dallas 的 TIGER BG 和 Amazon GEN-TRAIN donor。使用 [Dallas Cus100 模板](../examples/stage2/server_review.json)执行 `plan → build → verify --check-sources → inspect → export`。模板显式启用 BG 等概率 depot 抽样、90.75 分钟与单程能量准入、sequential tract、恰好 N 个地点的采样及距离与满充时间联合选站。
4. 返回输入和代码身份、执行日志、耗时及峰值内存，列出不支持的场景和原因。Dallas Cus100 审阅通过后再扩至其他规模。多城市批次的 seed、出发地配额与训练和测试划分随批量计划单独验收。

## 当前证据范围

Stage 1 的 97.7639% 是本地建筑属性覆盖率。最新冻结版本的 11 城全量道路组装及核验由服务器完成。Stage 2 的跨规模选站数据来自归档 Dallas 的离线对照。联合审阅版的规则通过代码和本地测试核对，最新版 CLE 的服务器联跑单列为验收项。实例生成和独立核验描述构造与来源一致性，车队路线的可行性与充电利用率由路线实验另行评价。
