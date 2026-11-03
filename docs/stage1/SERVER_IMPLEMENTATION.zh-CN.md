# Stage 1：本机与服务器执行

入口为 `evrptw-cle stage1`；`evrptw-data stage1` 和 `evrptw-stage1` 使用同一个 `src/evrptw_cle/stage1.py`。流程为 **prepare → scope → calibrate-units → estimate-units → finalize → 五部分组件与组装**。

正式 11 城冻结版本交接见[服务器 Agent 任务书](SERVER_AGENT_HANDOFF.zh-CN.md)。本文单城示例用于了解命令；全量交接必须对全 11 城联合校准一次，再估计与构造，避免单城调用覆盖全局门槛。

## 1. 安装和最短运行路径

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[osm]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
evrptw-cle stage1 init /data/evrptw/project
```

编辑生成的 `/data/evrptw/project/stage1.json`，填写服务器实际路径，然后运行：

```bash
evrptw-cle stage1 plan /data/evrptw/project/stage1.json
evrptw-cle stage1 run /data/evrptw/project/stage1.json --cities dallas
```

先处理建筑属性时用 `run ... --customer-only`。`build` 是 `run` 的别名。默认 `customer.active_profile=unrestricted`，保留所有通过当前规则的正整数户数；neighbor 来源继续单独标记。

## 2. 配置

以下为一城示例；相对路径以配置文件目录为基准。初始化模板中的字段和默认值可直接编辑。

```json
{
  "schema": "evrptw_building_first_stage1_v1",
  "output": "outputs/stage1-building-first",
  "axis_r": 0.2853432473555575,
  "cities": {
    "dallas": {
      "microsoft": "inputs/customer/dallas/microsoft_footprints.parquet",
      "osm": "inputs/customer/dallas/osm_buildings.parquet",
      "nsi": "inputs/customer/dallas/nsi.parquet",
      "area_crs": "EPSG:32614",
      "service_boundary": "inputs/boundaries/dallas/service_boundary.geojson",
      "admin_boundary": "inputs/boundaries/dallas/admin_boundary.geojson"
    }
  },
  "components": {
    "inventory": "inputs/inventory.json",
    "source_root": "inputs",
    "road_sources": "inputs/roads"
  },
  "source_snapshots": {
    "microsoft_index": "2026-08-13",
    "osm": "2026-10-25",
    "nsi_archive": "2026-10-26"
  },
  "units": {
    "seed": 20261102,
    "block_m": 2000,
    "train_pct": 80,
    "cal_pct": 10,
    "test_pct": 10,
    "k": 8,
    "scale_probe_count": 2000,
    "calibration_per_city": 5000,
    "quality_target_retention": 0.95,
    "quality_min_group_size": 30,
    "quality_min_neighbors": 8,
    "quality_dispersion_column": "neighbor_log_spread_90",
    "min_unit_area_m2": 10.0,
    "floor_to_floor_m": 3.048
  },
  "customer": {
    "active_profile": "unrestricted",
    "include_model_imputed": true
  }
}
```

`k` 决定参考邻居数量；`scale_probe_count` 控制每个城市/类型/分支的 RMS 拟合样本；`quality_target_retention` 控制校准参考支持份额。`min_unit_area_m2` 和 `floor_to_floor_m` 只决定宽松建筑规模筛查。位置/面积/高度标度由训练比较对拟合，不另配固定数值权重。参数解释与公式见 [Customer 算法](CUSTOMER_METHOD.zh-CN.md)。

## 3. 准备原始建筑和服务域

```bash
evrptw-cle stage1 prepare /data/evrptw/project/stage1.json --cities dallas
evrptw-cle stage1 scope /data/evrptw/project/stage1.json --cities dallas
```

`prepare` 读取 Global Microsoft、完整 OSM building 和原始 NSI，构造整栋别名、复合轮廓的 OSM 子栋及初始空间匹配。保存 population、原始 NSI、候选证据及几何 registry。NSI 零户数用于用途未知建筑的资格/冲突判断，正户数住宅池与其分开。

`scope` 用最终 canonical polygon 的 `point_on_surface` 决定服务域成员：MS 实体用自身轮廓，OSM 子栋用自身完整子栋。`scope.parquet` 保存稳定 ID、范围资格和 canonical 经纬度；manifest 绑定边界及源文件哈希。原始几何和来源记录不因道路接入失败而删除。

`geometry_cache` 是可选的显式几何/初始 owner 导入模式。它记录缓存来源和可核查的原输入绑定；原始重建不需要该字段。改变原始输入或几何规则时使用新输出目录，使来源和生成参数保持一致。

## 4. 校准并估计 units

```bash
evrptw-cle stage1 calibrate-units /data/evrptw/project/stage1.json --cities dallas
evrptw-cle stage1 estimate-units /data/evrptw/project/stage1.json --cities dallas
evrptw-cle stage1 finalize /data/evrptw/project/stage1.json --cities dallas
```

`calibrate-units` 固定清楚的原始 NSI 住宅参考池及空间分组，拟合 RMS 标度，再用留出的校准参考确定相似度/户数分散度联合边界。测试标签只作留出诊断。

`estimate-units` 使用同一训练参考，补充缺失高度的宽松规模代理，重评空间候选，选择全局不重复的 NSI 代表；最终多户建筑必须有可评估且通过的宽松规模筛查，未评估标 `scale_unassessed_multiunit`；只有仍无代表且参考支持成立的建筑才接受 neighbor 户数。原始异常、零/正冲突和邻居不支持保留原因。`finalize` 将结果写成标准 building/customer 属性与覆盖统计。

每栋输出 `units_source=nsi|neighbor|unresolved`。前两者的方法为 `nsi_representative` 和 `neighbor_estimated`；所选 NSI 用 `selected_nsi_fd_id` 关联原始记录，不相加。独立原高度、补高标记、标度、门槛和未解决原因分别保存。

## 5. 构造路网和地点组件

道路源包布局：

```text
inputs/roads/<city>/
  osm_ways.parquet
  osm_access_nodes.parquet
  extraction.json
```

采用同一 2026-10-25T21:21:21Z OSM 快照、行政边界外扩 20 km。跨 PBF 边界时，先统一原生对象再提取完整 ways/引用节点。原生单行、车辆/配送禁令、锁闭和障碍继续作用；普通 private 参与配送穿行。组件 inventory 提供 depot、AFDC、边界及来源元数据。

完整 `run` 自动执行以下组件；也可逐项运行：

```bash
evrptw-cle stage1 components /data/evrptw/project/stage1.json --component all
evrptw-cle stage1 verify /data/evrptw/project/stage1.json
```

`all` 顺序为 roads、customers、depots、charging、speeds、assemble。三类地点引用相同原生道路；customer 使用完整建筑 polygon 投影。速度绑定共同图的有向源边，明确物流园区整体合并，AFDC 原生坐标与设备保留独立来源。细节见[主规格](../STAGE1_SPEC.zh-CN.md)与[设施规格](DEPOT_CHARGING.zh-CN.md)。

## 6. 产物与资源安排

```text
<output>/
  customer_coverage.csv
  customer_uncovered_reasons.csv
  units/
    metric_scales.csv
    quality_thresholds.json
    calibration_predictions.parquet
    calibration.json
  stage1_method.json
  build_progress.json
  cities/<city>/
    customer_inputs/
      population.parquet
      scope.parquet
      scope_manifest.json
      source_records.parquet
      unit_features.parquet
      unit_candidate_edges.parquet
      unit_reference_roster.parquet
      nsi_representatives.parquet
      nsi_assignment_ledger.parquet
      unit_results.parquet
      unit_uncovered_reasons.csv
      unit_coverage.json
      final_buildings.parquet
      finalization_manifest.json
      geometry/
    roads/
    customers/
    depots/
    charging/
    speeds/
    locations.parquet
    location_access_costs.parquet
    cle_manifest.json
    stage1_validation.json
```

户数模块在 `customer_inputs/` 下保存每城参考划分、代表记录和未解决账本；项目 `units/` 保存 RMS 标度与校准门槛。下游读取标准 `final_buildings.parquet`，完整来源通过 manifest 和原始记录关联。

本机可先单城 `--customer-only` 查看交接与属性统计。大规模原始几何与道路重建放到服务器逐城运行；单位参考查询按块执行。迁移时保留代码版本、数据字节、配置和来源日期，修改实际文件路径。建筑覆盖率在道路接入前计算，最终 CLE 可用数另由 customer 组件与组装结果报告。

`verify` 校验已保存数据及共同路网/来源绑定。Stage 2 激活使用独立入口，Stage 1 构造结果保留其运行状态和 manifest。
