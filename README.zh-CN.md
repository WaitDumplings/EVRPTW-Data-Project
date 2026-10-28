# EVRPTW Data Project

用统一的全国数据源和规则，构造城市级配送环境（City Logistics Environment，CLE）。

[English](README.md) · [输入格式](docs/INPUTS.md) · [建模方法](docs/METHODS.md) · [输出与读取](docs/CLE_SCHEMA.md)

Stage 1 生成城市路网、潜在 customer location、depot location、charging location，以及同一路网上的参考速度和接入成本。Customer 是地点，house 单个、可识别的 apartment 小区整体一个；residential units 仅为辅助信息。

**本仓库提供实现和配置，不附带城市原始数据或已生成的 CLE。** 完整数据包与 Stage 2 instance activation 留到后续阶段。已经有本地来源包时，按下面的流程运行。

## 安装

需要 Python 3.11+。建议在 Linux 或 macOS 上使用独立环境；Windows 可使用 WSL。大城市按城市串行构造，建议 32 GB 内存并预留足够磁盘空间。

```bash
git clone https://github.com/WaitDumplings/EVRPTW-Data-Project.git
cd EVRPTW-Data-Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

需要从本地 OSM PBF 提取道路时，安装 `python -m pip install -e '.[osm]'`。

## 从一座城市开始

```bash
evrptw-data init my-city
```

将本地来源包放入 `my-city/inputs/`。默认项目选择 Dallas，可以编辑 `my-city/project.json`：

```json
{
  "schema": "evrptw_stage1_project_v1",
  "sources": {
    "inventory": "inputs/inventory.json",
    "root": "inputs",
    "roads": "inputs/roads",
    "customers": null,
    "osm_cohort": null
  },
  "output": "outputs/cle",
  "cities": ["dallas"]
}
```

所有相对路径以 project.json 所在目录为准，与启动命令的位置无关。sources.customers 为空时自动从原始 Microsoft / NSI 输入准备住宅来源；已有匹配结果时可指定 customer registry。道路已提取时直接使用 sources.roads；需要从 PBF 提取时指定 sources.osm_cohort。具体文件格式见[输入说明](docs/INPUTS.md)。

```bash
# 先查看缺少哪些输入，不开始大规模计算。
evrptw-data plan my-city/project.json

# 构造并进行一次最终数据验证。
evrptw-data build my-city/project.json

# 查看三类地点数量和保存的证书状态。
evrptw-data inspect my-city/outputs/cle
```

缺少来源包时，plan 会明确列出缺少的输入；它不会自动下载 latest 数据替代固定版本。

## 多城、续跑与日志

在配置里填写多个城市，或临时覆盖选择：

```bash
evrptw-data build my-city/project.json --cities dallas fort-worth
```

支持的参考城市为 New York、Los Angeles、Chicago、Houston、Phoenix、Philadelphia、San Antonio、San Diego、Dallas、Jacksonville 和 Fort Worth。城市输入必须在来源包中存在；当前速度配置覆盖 AZ、CA、FL、IL、NY、PA、TX。

同一命令再次执行时，仅复用输入、代码和输出一致的已完成组件。输入或实现改变后请指定新的 output，避免混用版本。详细日志在 output/logs/，城市组件日志在 output/cities/&lt;city&gt;/。

正常构造流程为：

```text
来源准备（如需要） → 路网 → customer / depot / charging / speed → CLE 组装 → 数据验证
```

正常构造不会运行 pytest、论文基线对比或历史个案审计。独立数据验证也可以稍后单独执行：

```bash
evrptw-data build my-city/project.json --no-verify
evrptw-data verify my-city/outputs/cle
```

读取器要求数据具有匹配 manifest 的独立证书；跳过验证的中间输出需要先执行 verify。

## 在 Python 中读取

```python
from evrptw_cle.cle_reader import CityLogisticsEnvironment

cle = CityLogisticsEnvironment("my-city/outputs/cle/cities/dallas")
locations = cle.locations()
eligible = locations.loc[locations.candidate_eligible]
public_roads = cle.graph("public")
delivery_terminal = cle.graph("delivery_terminal")
charging_terminal = cle.graph("charging_terminal")
access_costs = cle.access_costs()
```

三种图分别读取。Private terminal 不能直接合并成公共穿行捷径。潜在地点的保留数、默认可用数和待核数分开保存，不能把它们当作订单数或已激活客户数。

## 项目结构

```text
src/evrptw_cle/   构造、接入、参数、组装与读取实现
configs/          全国统一规则与七州速度统计
scripts/          命令内部使用的分阶段 worker
examples/         项目配置与读取示例
docs/             输入格式、建模方法、CLE 数据结构
tests/            回归测试，供开发时运行
```

开发检查：`python -m pip install -e '.[dev,osm]'`，然后运行 `pytest` 和 `ruff check src scripts tests`。源数据和第三方来源的使用说明见 [NOTICE.md](NOTICE.md)。
