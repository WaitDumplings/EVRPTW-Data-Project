# Stage 1 服务器输入布局

本交接对应 `stage1-20261102-r1`。代码包与数据包分开迁移；冻结模板和inventory保持原字节，需要改路径时复制同目录的 `stage1.local.json` 运行配置；[stage1.server.json](stage1.server.json) 使用相对路径，可直接放在解压后的同一目录结构中运行。输入目录均相对于代码仓库根目录。

```text
EVRPTW-Stage1-stage1-20261102-r1/
├── configs/phase1_sources.template.json
├── freeze/20261102-r1/
│   ├── stage1.server.json
│   ├── input_manifest.json
│   └── INPUT_LAYOUT.zh-CN.md
├── inputs/stage1_20261102/
│   ├── customer/<city>/
│   │   ├── microsoft_footprints.parquet
│   │   ├── osm_buildings.parquet
│   │   └── nsi_native.parquet
│   ├── archive/<inventory 中记录的相对路径>
│   └── roads/<city>/
│       ├── osm_ways.parquet
│       ├── osm_access_nodes.parquet
│       └── extraction.json
└── outputs/stage1-20261102-r1/   # 服务器运行后新建
```

`<city>` 为配置中的11个城市：new-york、los-angeles、chicago、houston、phoenix、philadelphia、san-antonio、san-diego、dallas、jacksonville、fort-worth。

## 必须迁移的数据

[input_manifest.json](input_manifest.json) 精确列出116个数据文件，共 **2,807,236,223字节，约2.614 GiB**。每条记录包含用途、城市、目标路径、大小和SHA-256。

| 数据组 | 文件数 | 字节数 | 运行时用途 |
|---|---:|---:|---|
| Microsoft、OSM、native NSI | 33 | 1,774,598,488 | 重建建筑、用途、零户证据、代表户数与邻居参考 |
| 行政边界、服务边界 | 22 | 15,989,706 | 路网外扩范围、建筑和设施服务范围 |
| Depot原始记录及提取来源 | 22 | 8,092,013 | 物流设施筛选、明确园区合并及源快照绑定 |
| AFDC地点、设备、地址辅助表及三份来源记录 | 6 | 129,711,694 | 地点准入、同址合并、接口和功率选择 |
| 完整道路、原生限制节点、提取记录 | 33 | 878,844,322 | 有向路网、车辆禁行、路口／屏障及真实城外道路 |

这里的Microsoft、OSM、NSI输入是当前生成器直接读取的冻结规范化文件。原Microsoft文件可能仍带有其他`source_kind`记录；保持输入字节不变，当前建筑构造只读取`global`记录。无需另外迁移legacy原始分片。

**NSI必须使用配置中指定的完整native点表。** 它保留零户、非住宅及正住宅记录，供零户证据判断；正住宅匹配台账由生成器另行构造。库存中的旧`nsi_crosswalk.parquet`不能替代`nsi_native.parquet`。

道路输入必须三份同时迁移。`osm_ways.parquet`含完整道路节点及坐标；`osm_access_nodes.parquet`保留同一PBF中的原生准入和屏障标签；`extraction.json`绑定两者的哈希、行政边界和`2026-10-25T21:21:21Z`快照。当前构造入口直接读取这些冻结提取文件，不需要再读取PBF本体。

## 随代码包保留的配置

`configs/phase1_sources.template.json` 必须随代码包保留，SHA-256为：

```text
d206978b7b22324546d258e6981d5cdabe620d2c75c211e69f36026f9b860fb2
```

服务器配置的`components.inventory`指向该文件，`components.source_root`指向`inputs/stage1_20261102/archive`。因此，边界、depot和AFDC必须保留库存记录中的目录层级。库存还包含未被本版Stage 1使用的历史条目；本版入口只读取输入清单列出的运行时子集。

代码包中的速度州级profile、registry、充电参数及其他`configs/`方法文件也应完整保留。它们属于代码／方法资源，已与116个外部数据文件分开；无需额外安装MOVES数据库或重跑速度统计提取。

## 不需要迁移的已有运行结果

服务器会重新构造`customer_inputs`、建筑匹配、参考池、校准门槛、户数、道路接入和最终CLE。已有的`outputs/`、中间匹配缓存、AI响应、人工评审文件、旧CLE、checkpoint和Stage 2实例不作为这次重建输入。

已有路网提取文件即使在本机存放于某个`outputs`目录中，仍是本次明确冻结的输入，必须按清单迁移到`inputs/stage1_20261102/roads`。不要依据原目录名称决定是否丢弃文件；以`input_manifest.json`为准。

## 重新生成清单与本机迁移映射

在已有本机输入的仓库根目录中执行：

```bash
python scripts/prepare_stage1_server_inputs.py \
  --project <本机项目配置.json> \
  --output freeze/20261102-r1/input_manifest.json \
  --server-config freeze/20261102-r1/stage1.server.json \
  --local-map private/stage1-server-input-map.json
```

脚本只读输入并计算哈希，写出配置和小清单，不复制数据、不下载。`--data-root`可调整服务器输入根目录，默认`inputs/stage1_20261102`；`--freeze-id`默认`stage1-20261102-r1`。

公开清单的`data_root`相对于代码仓库根目录，`files[*].target_path`相对于`data_root`。本机绝对源路径仅保存在`private/stage1-server-input-map.json`的`source_path`中，`repo_target_path`给出相对于仓库根目录的迁移目标。该private映射及原本机项目配置不放入代码包。

`stage1.server.json`已写入当前统一`units`默认参数、`unrestricted`户数范围、7州速度所需的11城标识及源快照日期。服务器可以修改输出路径和已公开的建模参数；保留冻结输入字节与输入清单的对应关系。
