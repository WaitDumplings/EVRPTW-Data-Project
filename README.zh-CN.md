# EVRPTW Data Project

用同一套规则构造城市配送环境（CLE）：**路网 → 参考速度 → 带住宅户数的建筑地点 → depot → charging station**。

[English](README.md) · [完整 Stage 1 方法](docs/STAGE1_SPEC.zh-CN.md) · [本机与服务器运行](docs/stage1/SERVER_IMPLEMENTATION.zh-CN.md) · [当前建筑统计](docs/stage1/STATISTICS.zh-CN.md)

Customer 以 Microsoft–OSM 对应的建筑为实体，NSI 提供住宅属性与户数证据；每栋选择一条合适的 NSI 代表记录，缺失户数由经过校准的 RMS 邻居模型补全。原始记录、最终户数、选择方法与未解决原因分别保存。本机和服务器执行同一条确定性流程。

冻结版本的 11 城全量生成按[服务器 Agent 任务书](docs/stage1/SERVER_AGENT_HANDOFF.zh-CN.md)执行：全体城市统一校准一次，再逐城构造完整 CLE。文档区分本机已完成的建筑属性结果和服务器待完成的道路接入/组装验收。

## 安装

需要 Python 3.11+，支持 Linux/macOS。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

从本地 OSM PBF 提取数据时安装 `python -m pip install -e '.[osm]'`。

## 从一座城市开始

```bash
evrptw-cle stage1 init my-city
# 编辑 my-city/stage1.json，填写本地源文件路径。
evrptw-cle stage1 plan my-city/stage1.json
evrptw-cle stage1 run my-city/stage1.json --cities dallas
```

`evrptw-data stage1` 和 `evrptw-stage1` 提供相同命令。相对路径以配置文件目录为基准。仓库提供实现和配置，城市源文件按[输入说明](docs/INPUTS.md)准备。

`run` 依次完成建筑准备、服务域归属、户数模型校准、户数估计、属性定稿及五部分 CLE 构造。先检查建筑属性时可执行：

```bash
evrptw-cle stage1 run my-city/stage1.json --cities dallas --customer-only
```

各阶段也可以独立运行：`prepare`、`scope`、`calibrate-units`、`estimate-units`、`finalize`、`components` 和 `verify`。完整参数与文件说明见[运行文档](docs/stage1/SERVER_IMPLEMENTATION.zh-CN.md)。

## 数据与输出

路网取行政边界外扩 20 km，保留道路方向、车型访问和实际障碍；普通 private 道路按配送穿行假设纳入路由。州级 NEI/StreetLight 统计为每条道路限速配置参考运行系数。Customer 使用 canonical 建筑；明确物流园区整体归并为 depot；AFDC 提供充电地点和设备。三类地点接入同一套路网。

当前 11 城离线属性运行中，**4,012,462 个候选建筑有 3,922,741 个得到可用住宅户数，覆盖率 97.7639%**。其中 3,194,252 个来自 NSI 代表记录，728,489 个来自邻居估计。此比例是道路接入前的建筑属性覆盖率；逐城数量、未补全原因与最大户数见[统计文档](docs/stage1/STATISTICS.zh-CN.md)。

每城组装结果包括五个组件目录、`locations.parquet`、`location_access_costs.parquet` 及来源和构造 manifest。每个 customer 保存 `units_source`、`units_method`、原始 NSI 记录关联与最终户数。详细字段见 [CLE Schema](docs/CLE_SCHEMA.md)。

```python
from evrptw_cle.cle_reader import CityLogisticsEnvironment

cle = CityLogisticsEnvironment("my-city/outputs/stage1-building-first/cities/dallas")
locations = cle.locations()
```

## Stage 2 配送实例

Stage 2 在核验通过的 CLE 上抽取 depot 和固定数量的 customer，生成订单、选取充电站并保存有向道路成本。先用[审阅配置](examples/stage2/server_review.json)跑 Dallas Cus100，命令和验收见 [Stage 2 运行说明](docs/stage2/REVIEW_PACKAGE.zh-CN.md)。两阶段的服务器顺序和交付物见[联合交接](docs/SERVER_HANDOFF.zh-CN.md)。历史数据与入口见[历史说明](docs/HISTORICAL_CLE.md)。
