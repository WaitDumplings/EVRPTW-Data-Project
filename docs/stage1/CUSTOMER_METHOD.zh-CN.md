# Customer：建筑、NSI 代表记录与户数补全

本方法把 customer location 定义为 **canonical building with residential units**。Microsoft/OSM 确定建筑身份与空间位置，NSI 提供原始住宅属性，邻居模型只在既有来源不能提供可用户数时补全。构造入口为 `evrptw-cle stage1`，完整运行步骤见[运行文档](SERVER_IMPLEMENTATION.zh-CN.md)。

## 1. 固定建筑几何和候选范围

1. 读取 Microsoft Global 建筑及完整 OSM building，保留原始 ID、几何和标签；`building:part` 不单独作为完整建筑。完全相同的归一化 polygon 去重，原始别名保留。
2. 在全部用途之间建立正面积相交图。双方相交 degree 都为 1 且 IoU≥0.5，才形成可信整栋别名。
3. 一个 OSM 子栋至少 50% 面积位于 Microsoft 父轮廓内，且子栋内部代表点被父轮廓覆盖时，建立实质父子关系。父轮廓包含两个及以上实质子栋时，以完整 OSM 子栋替代复合父轮廓。相关 Microsoft 父 ID 保留，父轮廓退出地点计数与障碍物，不把子栋裁成碎片。
4. Customer 几何范围为可信 MS–OSM 别名及上述细分子栋。城市归属采用 canonical polygon 的 `point_on_surface` 与服务面的 `covers` 关系；保留完整轮廓。
5. OSM 明确非住宅、混合用途分别列出。用途未知且只有清楚的 NSI 零户数证据的建筑排除；零/正冲突与归属未决单列，不据此猜成一户。清楚的 NSI 一户住宅可提供住宅支持。其余住宅和用途未知对象形成固定候选分母，未补全者继续留在分母。

独立 OSM 建筑和无可信 OSM 对应的 Microsoft 建筑保留空间证据/障碍信息，不作为额外 customer。OSM `house`、`apartments`、`generic` 仅用于相似建筑分组；最终地点都按建筑计数，不做小区级合并。

代码：`customer_geometry.py`、`building_inputs.py`；服务域和范围在 `scope.parquet` 及其 manifest 中记录。

## 2. 建立 NSI 空间候选

NSI 2026 的普通住宅 RES1/RES2/RES3、正整数 `resunits` 构成正户数来源池。保留 `fd_id`、原坐标、occupancy、BID、来源 footprint 标识、面积和原户数；没有候选的记录也保留。

空间候选使用以下证据：

- 完整 canonical polygon 是否覆盖 NSI 点，同实体的多个 alias 路径只算一次。
- BID 可解码的来源边框、来源 footprint 面积，以及目标 polygon 的最小旋转矩形。
- 来源 bbox IoU≥0.25，来源/目标面积的对称比≤4。
- 对建筑外点，结合凸包/院落、来源框和面积支持，并分别按长、短轴计算归一化偏移，避免长条建筑的横向误差被长对角线稀释。
- 轴向偏移阈值 `axis_r=0.2853432473555575`，来自固定 silver 参考记录的城市/真实来源组分层等权 Q99。

候选协调保留统一证据维度的 Pareto 前沿和原 owner 的反证。完整 OSM containment 会限制来源只能竞争真正覆盖该点的 canonical 候选；多个连排住宅边界共同覆盖点时，在这些候选内部解决，而不是整批放弃或转到无关大轮廓。

BID 框是上游线索，不是 NSI 提供的原始 polygon；NSI 户数也是模型属性。坐标匹配与户数是否符合当前建筑规模分开处理。代码：`customer_matching.py`、`bid_geometry.py`。

## 3. 独立建筑规模筛查

令建筑面积为 $A$，独立高度为 $H$，OSM 层数为 $L$。可用时取较大的楼层上界代理：

\[
f=\max\{\lceil H/3.048\rceil,\lceil L\rceil\},\qquad
U_{\mathrm{scale}}=\max\{1,\lfloor Af/10\rfloor\}.
\]

只对有值的项取最大值。3.048 m 和 10 m² 是可配置的宽松楼层/每户面积建模尺度；总建筑面积全部计入，保留屋顶和非住宅面积的余量。因此该式用于排除明显不合建筑规模的大数，不等同于实测住房容量。一户来源支持单独保留。高度/层数都缺失时先保留“未评估”，并尝试第 6 节的独立邻居补高。最终 U>1 必须有可评估且通过的规模代理；仍无法评估则标 `scale_unassessed_multiunit`，留在分母但不准入。一户来源支持保留既定例外。

独立高度来自同栋 Microsoft/OSM；NSI 自己的 modeled storeys 不作为验证该条 NSI 户数的独立输入。缺失高度的邻居补全见第 6 节，始终记录是否估计。

## 4. 固定原始参考池

参考建筑必须有清楚的单条、单栋 NSI 住宅归属，所选来源与原来源一致，无原生冲突，有有效位置与面积，且原始户数为一户或通过可用的独立规模筛查。多个 NSI 先不相加；邻居估计也不回流参考池。

参考池按 2 km 空间块，结合跨块实体/真实来源组关系，做稳定的 train/calibration/test 划分；默认份额 80/10/10。相关来源留在同一组。训练参考一旦固定，同一个模型用于标度拟合、校准目标预测和最终待补建筑；不在最终推断时悄悄换成全体参考。

校准默认每城抽至多 5,000 条，记录人口/样本权重。测试样本只用于留出诊断。待预测建筑的户数、NSI RES3 户数档位和由其推算的层数不作为特征。

## 5. 从数据拟合三组 RMS 标度

按城市、OSM 类型（house/apartments/generic）、可用垂直信息建立分组。有有效 OSM 层数则使用 levels 分支；否则有独立高度则使用 height 分支；两者缺失则使用 area 分支。

每个训练分组最多抽 2,000 个探针，仅按地理距离找同分组最近的 8 个独立参考，排除同一实体/来源组。该比较对不使用预设的位置、面积或高度系数。对比较对 $P$ 计算：

\[
s_r^2=\frac1{|P|}\sum_{(i,j)\in P}\|x_i-x_j\|^2,\quad
s_A^2=\frac1{|P|}\sum_{(i,j)\in P}(\log A_i-\log A_j)^2,\quad
s_V^2=\frac1{|P|}\sum_{(i,j)\in P}(\log V_i-\log V_j)^2.
\]

$V$ 为该分支的独立高度或 OSM 层数。设可用、非恒定的特征组数为 $m$，距离定义为：

\[
d(i,j)=\sqrt{\frac1m\left[
\frac{\|x_i-x_j\|^2}{s_r^2}+
\frac{(\log A_i-\log A_j)^2}{s_A^2}+
\frac{(\log V_i-\log V_j)^2}{s_V^2}\right]}.
\]

方括号只对可用且非恒定的组求和；area 分支省去垂直项。位置的 x/y 合成一个径向组。每个有效组在拟合比较对上的平均平方贡献为 $1/m$；这是标度定义，不要求每个地点或最终挑出的 KNN 都恰好等贡献。

若局部比较对无差异但完整分组仍有变化，采用该分组独立随机对的 RMS（$\sqrt{2\operatorname{Var}}$）并记录回退。真正恒定的组不制造微小 epsilon；新目标若偏离该常量则标为不在支持范围内。每组记录参考数量、比较对数、标度与回退来源。当前 11 城运行包含 99 个分组，均使用地理近邻对标度；含垂直信息的 66 组贡献各为 1/3，其余 33 组各为 1/2。

## 6. 邻居估计、相似度和分散度

在拟合后的距离中找最近 8 个独立参考。取权重 $w_j=(1+d_{ij})^{-2}$，预测为 log 户数的加权中位数，再指数还原、四舍五入为至少一户的整数。保留参考 ID、距离及原户数范围。

相似度成本为加权平均距离；分散度为参考 log 户数的加权 Q90−Q10。校准使用这两个量的经验边界：分别取相同边际分位 $q$，选择能达到目标联合保留份额的最小 $q$。默认目标为 95%，至少 8 个有效邻居。

阈值首先按 city/type/branch 拟合，样本不足 30 条时依次合并为 type/branch、type、全部参考。它表达参考数据的支持范围，95% 是可配置的保留目标；不使用目标建筑户数误差选择阈值，也不表示 95% 实测户数准确率。

缺失高度时，在同类型、原始独立高度非空的训练参考中，以 area 分支同一 RMS 距离取邻居高度加权 P90。只有相似度通过才把它用于宽松规模筛查；不再把补出的高度当作原始垂直特征重新预测户数。原高度和 `height_neighbor_estimated` 分别保存。

## 7. 全局选择 NSI 代表记录

先筛掉不合格空间边、原生来源冲突、明确非住宅/混合 containment、非正整数户数和可评估的规模异常。对剩余“来源组—建筑”边：

1. 按点包含直接匹配优先，再按可用邻居户数差异、来源/目标面积与高度差异、几何 gap 和稳定 ID 排序。
2. 同一真实来源组/建筑只保留一个最优代表边。
3. 在冲突连通分量内求二分匹配：优先最大化直接匹配数，再最大化推断匹配数，最后按证据顺序打破并列。大分量使用稀疏求解。
4. 每栋、每条 NSI、每个真实 footprint 来源组最多选一次。来源标识为空或为占位符时按 fd 单例，不误并为共同组。

最终 NSI 户数就是所选一条 `resunits`。多条来源不相加，小区总数不按面积分摊。clear 原始参考使用自身已知 NSI 值协助代表选择；其他建筑只在支持条件通过时使用邻居预测。

## 8. 补全、未解决原因与道路接入

最终先执行统一规模准入：U>1 必须 规模代理可用（`capacity_units` 非空） 且不超过宽松规模代理；未评估多户建筑保留空最终户数与 `scale_unassessed_multiunit`。通过后的 NSI 代表记 `units_source=nsi`、`units_method=nsi_representative`。无代表时，只有有效建筑、无原生冲突、邻居支持通过、预测正整数、U>1 时规模可评估且未超代理者，才记 `units_source=neighbor`、`units_method=neighbor_estimated`。若原归属的所有 NSI 都已因建筑规模异常被排除，不用邻居估计覆盖该异常。

其余保留 `units_source=unresolved`、空最终户数和互斥原因：原生冲突、建筑特征缺失、多户规模未评估、原始规模异常、独立邻居不足、相似度不支持、分散度不支持、两项均不支持或预测户数超规模。先后顺序写入结果，统计时每栋只计一次。

所有原始 NSI、候选边、所选来源和未选原因均保留。构造仅依赖固定输入、配置和随机种子。最终完整 canonical footprint 接入共同有向路网；属性准入与道路接入分别计数。当前[11 城结果](STATISTICS.zh-CN.md)是属性层统计，后续组装另报道路可用地点。

## 9. 包内实现与保存文件

| 职责 | 文件与主要入口 |
|---|---|
| Canonical 几何与原始空间候选 | `customer_geometry.py`、`customer_matching.py`、`building_inputs.py` |
| 统一候选、零户数/冲突与原始特征 | `unit_features.py` |
| 空间/来源组参考划分 | `unit_reference.make_reference_splits` |
| RMS 标度及户数/补高查询 | `unit_neighbors.EmpiricalMetricNeighborModel` |
| 校准支持门槛 | `unit_quality.fit_quality_thresholds`、`apply_quality` |
| 宽松建筑规模代理 | `unit_scale.annotate_capacity` |
| NSI 代表边及全局选择 | `unit_matching.select_representatives` |
| 准备、校准、估计与最终属性 | `building_units.prepare_units_city`、`calibrate_units`、`estimate_units`、`finalize` |
| CLI 与完整串联 | `stage1.py` |

文件位于 `src/evrptw_cle/`。每城 `customer_inputs/` 保存 `unit_features.parquet`、`unit_candidate_edges.parquet`、`unit_reference_roster.parquet`、`nsi_representatives.parquet`、`nsi_assignment_ledger.parquet`、`unit_results.parquet`、`unit_uncovered_reasons.csv` 和标准 `final_buildings.parquet`。项目级 `units/` 保存 RMS 标度、校准预测、门槛与配置 manifest。原始 source ledger 始终单独保留。
