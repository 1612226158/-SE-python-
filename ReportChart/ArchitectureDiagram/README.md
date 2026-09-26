# 层叠架构图与消融参数差分台

> 单文件、离线、零外链的交互式架构图。双击 `index.html` 即可打开。
> 版本：**v0.5.1**（合并组名定名）｜最后更新：**2026-09-26 10:40**
> 交付物：`index.html`（1,291 行 / 71,924 B，SHA256 `0A0CF700…F18DC85`）、本 README、`_review_screenshots/`
> ⚠️ **v0.5 口径变更**：`merged_dict` 由 7 组 18 子类（254 类）扩为 **9 组 22 子类（252 类）**，
> 分类头 130,302 → **129,276**，各配置总参数 −1,026。详见第二节表下说明与第十二节变更记录。

---

## 一、它解决什么问题

`garbage265` 的消融实验有 10 个配置，差异集中在四处：**SE 区域数**、**Transformer 层数**、**头版本（v1/v2）**、**解冻策略**。
靠读代码或读表格很难一眼看出「这个配置比那个配置多了什么、多了多少参数」。

这个页面把它变成一张图：**左侧点一个配置 → 中间 13 层的方块堆叠图就地变形换色 → 右侧给出与基准的逐模块参数差分**。

三条硬性要求（用户在三次返工中明确提出，也是本页的设计约束）：

1. **必须是一层层的方块堆叠图**（对齐经典 Transformer 论文图的语汇），不能只是"文字行 + 说明"；
2. **不能有文字重叠或超出边框**（前两版都栽在这里）；
3. **换配置要让图自动以动画变形**，而不是"写好几个页面只是切换一下"；
4. 原有文字说明**一条都不能删**。

---

## 二、三个结构性设计决定（这是"不会再重叠"的根本原因）

前两版之所以修不好文字重叠，是因为方法本身不可靠：用 `label.length * 14.2` 猜文字宽度来定位 SVG `<text>`，估算一错就叠字，而且没有任何测量校验。

本版换了三条做法：

| # | 决定 | 为什么它能根除问题 |
|---|---|---|
| 1 | **槽位固定**：13 个层框永远存在于 DOM，位置由常量 `HGT` / `GAP` 计算，**永不移动** | 没有重排就没有错位；同时让"动画"成为可能（位置不变才能做过渡） |
| 2 | **框内是 HTML 而不是 SVG 文本**：`<foreignObject>` 里放 `<div class="cb">`，flex 布局 + `min-width:0` + `text-overflow:ellipsis` + `overflow:hidden` | 内容放不下时被省略号截断，**在结构上不可能溢出**。这是浏览器排版引擎保证的，不依赖任何估算 |
| 3 | **就地更新，不重建 DOM**：`buildDiagramDOM()` 只执行一次；换配置走 `updateDiagram()` 改 class 与文本 | 颜色/位置变化交给 CSS `transition`，产生真正的"变形"而不是"换页" |

> 验证方式不是目测。页面内置了 `window.__qa()`（见第六节），它直接读 `scrollWidth/scrollHeight/clientWidth/clientHeight/offsetWidth` 做机器断言。

---

## 三、参数量推导与交叉校验

口径：骨干取 `children()[:-2]`（去掉原 `avgpool` 与 `fc`），即 `resnext50_32x4d` 的 25,028,904 − `fc`(2048×1000+1000) = **22,979,904**。
页头「总参数」= 骨干 + 本项目头部，**已扣除骨干原 fc 的 2,049,000**。

| 模块 | 公式 | 参数量 |
|---|---|---|
| ResNeXt-50 32×4d 骨干 | 25,028,904 − 2,049,000 | 22,979,904 |
| 投影 `Linear(2048 → 512)` | 2048×512 + 512 | 1,049,088 |
| 位置编码 `pos_embed`（**仅 v2**） | 49×512 | 25,088 |
| 单层 Transformer（v1 与 v2 **相同**） | MHA 1,050,624 + FFN 2,099,712 + 2×LN 2,048 | 3,152,384 |
| 分类头 `Linear(512 → N_CLS)` | 512×252 + 252（`N_CLS=254` 时为 130,302） | **129,276** |
| 简化解耦层 | `Linear(512→256)` 131,328 + `BN1d(256)` 512 + `Linear(256→512)` 131,584 | 263,424 |
| SE `regions=[1]` | 2048×128 + 128×2048（`bias=False`） | 524,288 |
| SE `regions=[1,2]` | `mid = max(2048×5//16, 128) = 640` → 10240×640 + 640×2048 | 7,864,320 |

| 配置 | 头版本 | SE | Trans. | 解冻 | 总参数 |
|---|---|---|---|---:|---:|
| `S-NoSE` | v1 | 无 | 3 | none | 33,878,844 |
| `S-StdSE` | v1 | `[1]` | 3 | none | 34,403,132 |
| `S-NoProg` | v1 | `[1,2]` | 3 | none | 41,743,164 |
| `V2-Full` | v2 | `[1,2]` | 3 | none | 41,768,252 |
| `V2-NoTF` | v2 | `[1,2]` | **0** | none | 32,311,100 |
| `G-Full-CAWR` | v1 | `[1,2]` | 3 | progressive | 41,743,164 |
| `G-Full-CAWR-Long` | v1 | `[1,2]` | 3 | progressive | 41,743,164 |
| `G-V2` | v2 | `[1,2]` | 3 | progressive | 41,768,252 |
| `G-Full` / `G-NoSE` | v1 | `[1,2]` / 无 | 3 | progressive（已退场，RLRP） | 41,743,164 / 33,878,844 |

> **2026-09-26 合并类数变更**：`merged_dict` 由 7 组 18 子类（254 类）改为 **9 组 22 子类（252 类）**——
> 并入「蔬菜 + 菜根菜叶 → 菜叶」与「插头电线 + 充电线 → **插头充电线**」。
> 组名取「插头充电线」而非「充电器」，避免与既有类别「充电头」(id 68) 语义重叠。
> 分类头参数量 130,302 → 129,276，**全部配置总参数各 −1,026**（上表已是变更后值）。
> 页面常量 `N_CLS = 252`，标题/分类头/输出/协议带均由此常量插值，不再硬编码。
> 交叉校验改为：`V2-Full` 较 `S-NoProg` 恰多 **25,088**（位置编码，与类别数无关，不受本次变更影响）✓。

**四项独立交叉校验**（推导值 vs 项目自身记录）——**合并类数变更后仍全部成立**（两位小数口径不变）：
头部 **18,763,260**（18.76M ✓） · `G-Full`/`S-NoProg` **41,743,164**（41.74M ✓） · `G-NoSE`/`S-NoSE` **33,878,844**（33.88M ✓） · `V2-Full` 较 `S-NoProg` 恰多 **25,088** ✓（位置编码，与类别数无关）。
差分自洽：`V2-NoTF` 相对 `S-NoProg` 逐层 Δ 之和 = 表尾合计 = **−9,432,064** ✓（41,743,164 − 32,311,100）。

---

## 四、状态语义：**外观 ≠ 差异**（本版关键设计）

页面同时回答两个不同的问题，**必须分开**：

| 问题 | 函数 | 决定什么 |
|---|---|---|
| 这个模块**在本配置里存在吗**？ | `presentOf(id, m)` | 方框**长相**（实心 / 虚线直通） |
| 这个模块**相对基准变了吗**？ | `statusOf(id, m, b)` | **颜色**与是否进「变化摘要」 |

为什么要分开：v1 头**完全没有**位置编码。如果在 v1 配置下只按"相对基准"判定，当基准也是 v1 时两边同为缺失 → 判成「= 一致」，于是画出一个**实心块**写着「可学习位置编码 / `pos_embed (1, 49, 512)`」——
**展示了一个源码里根本不存在的参数**。这是第三轮一致性审查查出的**实质对立**。

本版的解法是 `visOf()`：`statusOf()==='same' && !presentOf(id,m)` 时外观改判为 `∅ 直通`（虚线玫色），但**不计入差异**。

六种状态（每种都同时有字形 + 文字标签，不单独依赖颜色）：

| 字形 | 状态 | 含义 | 色值 |
|---|---|---|---|
| `=` | 一致 | 与基准相同 | `#8b95a3` |
| `Δ` | 参数变化 | 参数量改变 | `#ffd166` |
| `≡` | 结构变化 | 结构改变、参数量不变（如 v1↔v2 的分词/读出） | `#5cb8f0` |
| `∅` | 直通 | 本配置中该模块被 `nn.Identity()` 替换 | `#c98b8b` |
| `＋` | 新增 | 基准没有、本配置有 | `#5fd39a` |
| `❄` | 冻结 | 渐进式的 state 1/2 下骨干不可训练 | `#a4b0c8` |

> `--st-changed:#ffd166` 刻意与选中态强调色 `--accent:#f0a020` 拉开色相，避免"选中"与"数据状态"混淆。
> `--st-same` 由 `#6b7686` 提亮为 `#8b95a3`：前者在块底 `#171c23` 上只有 **3.72:1**（UI 审查实测），不达正文 4.5:1。

---

## 五、动画机制（为什么是"变形"而不是"切页"）

| 手段 | 实现 | 效果 |
|---|---|---|
| 逐块错峰入场 | `@keyframes cbPop`，`animationDelay = 层序号 × 30ms` | 13 个方块自上而下依次亮起，像"重新计算了一遍" |
| 扫描光带 | `#scan` 的 `<rect>` 沿 `translateY(var(--scan-to))` 走一次（1.25s） | 一道琥珀色光带自上而下扫过整列 |
| 箭头流动 | 换配置时给 `.conns` 加 `.flow`，触发 `stroke-dashoffset` 动画 | 连接箭头出现短暂的"数据流动" |
| 换色过渡 | `.cb` 上 `transition: background-color / border-color 460ms` | 状态色平滑过渡而非跳变 |
| 文本微动画 | `setTxt()` 只在文本**真的变了**时加 `.swapped`（420ms 淡入） | 变化的那几个字单独"闪"一下 |

**动效总开关**：`prefers-reduced-motion: reduce` 下全部关闭（CSS 与 JS `reduceMotion` 双重判断）。
**动画结束即清理**：`.cb` 上有 `animationend` 监听移除 `.tick`，因此 `.tick` 可被用作"本次正在动画"的判据。

> ⚠️ 无头 Chrome 实测 `matchMedia('(prefers-reduced-motion: reduce)')` 返回 **true**，动画会被全关。
> 若要在无头环境验证动画，必须先覆写 `window.matchMedia` 返回 `{matches:false}` 再加载页面脚本。

---

## 六、自检器 `__qa()`（怎么证明"没有重叠/溢出"）

页面尾部内置一个自检函数，控制台或自动化脚本都可直接调用：

```js
window.__qa()
// → {"overflow":[], "ellipsis":[], "H":1256, "svgW":719}
```

判定口径：

- `overflow` —— ① 任一 `.cb` 的 `scrollWidth/scrollHeight > clientWidth/clientHeight + 1`（内容被裁切）；
  ② `offsetWidth/offsetHeight` 与 `foreignObject` 声明尺寸不符（外框几何错误）。
  **非空即代表有内容超出边框。**
- `ellipsis` —— 任一文字元素的 `scrollWidth > clientWidth + 1`，即被省略号截断。用于发现"框太窄装不下"。
- `svgW` —— SVG 实际渲染宽度。`svgW / 700` 即缩放系数，乘框内字号可算出**有效像素字高**。

**实测结果（本轮定稿）**：

| 项 | 结果 |
|---|---|
| 19 个视口宽度（1920 → 560）× 默认态 | `overflow: []`、`ellipsis: []` **全部为空** |
| 12 个状态（10 配置 + 3 个 stage 变体，含 2 个退场项） | `overflow: []`、`ellipsis: []` **全部为空** |
| 最差有效字高 | **12.4px**（560px 视口，`svgW=600`）；1440px 视口 13.65px；≥1500px 为 15.3px |
| console 报错 / `undefined` / `NaN` / `function` 文本 | 30 组（10 配置 × 3 stage）全部为 **0** |

---

## 七、如何自己复现渲染验证

本机环境有三处坑，配方已按它们写好：

- **PowerShell 的 stdout 不回传** → 所有输出必须写临时文件再用 Read 读；
- **PowerShell 5.1 的 `Get-Content` 不带 `-Encoding UTF8` 会把中文读成乱码**，中文匹配一律用 ripgrep；
- **Pillow 装不上**（pip 走代理 502）→ 要让细节看得清，就把 SVG 渲到原生 700px、窗口宽度跟着走，一次截全图。

```powershell
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$base = Get-Content "index.html" -Raw -Encoding UTF8
# 状态注入 + QA 断言（探针只写 $env:TEMP，不动交付文件）
$probe = "<script>state.baseline='S-NoProg';selectModel('V2-NoTF');</script>" + @'
<script>(function(){var e=[];window.onerror=function(m,s,l){e.push('E:'+m)};
setTimeout(function(){var q={};try{q=window.__qa()}catch(x){q={threw:x.message}};
document.title='QA||'+JSON.stringify({q:q,err:e});
var d=document.createElement('div');d.id='qa-out';d.textContent=document.title;document.body.appendChild(d)},500)})();</script>
'@
$p = "$env:TEMP\arch_case.html"
[System.IO.File]::WriteAllText($p, $base + $probe, (New-Object System.Text.UTF8Encoding($false)))
$url = "file:///" + ($p -replace '\\','/')

# ① 取 QA 断言（抓 id="qa-out"）
& $chrome --headless=new --disable-gpu --enable-unsafe-swiftshader --no-first-run --hide-scrollbars `
  --user-data-dir="$env:TEMP\arch_prof" --window-size=1500,1400 --virtual-time-budget=6000 `
  --dump-dom $url 2>$null | Out-String

# ② 取截图；让图放大到可读就加 --window-size=740,1400 并注入一段隐藏其他区块的探针 CSS
& $chrome --headless=new --disable-gpu --enable-unsafe-swiftshader --no-first-run --hide-scrollbars `
  --user-data-dir="$env:TEMP\arch_prof" --window-size=740,1400 --virtual-time-budget=7000 `
  --screenshot="$env:TEMP\arch.png" $url
```

---

## 八、审查与修复记录（三轮，均为独立子代理）

按项目规范"面向人的视觉产出定稿前必审"，本题共走了 **4 路独立审查**（3 路首审 + 1 路修复验证）。

### 第 3 次重做后的首审（3 路）

| 审查路 | 评分 | 关键发现 |
|---|---|---|
| **UI/UX** | 8.0/10 | 24 组合自检 `overflow/ellipsis` 全空；方块堆叠与动画机制均成立；但查出 3 条 P1（焦点环被裁、数字键劫持阶段键、`--st-same` 对比度 3.72:1） |
| **功能逻辑** | — | `buildDiagramDOM()` 只调用一次、13 块节点身份与几何零变化（"就地动画"有硬证据）；10 配置 × 3 stage 无 `NaN`/`undefined`/报错；参数量逐字符正确；查出 2 条 P2（`.tick` 残留、切 stage 全量重放）+ 3 条 P3 |
| **架构一致性** | — | 22 条断言中 19 条成立；查出 **1 条实质对立**（v1 下显示了源码里不存在的 `pos_embed`）+ 2 条次要措辞不一致 |

### 本轮据审查修复的清单

| 编号 | 问题 | 修法 |
|---|---|---|
| **P1-1** | `.seg{overflow:hidden}` 把阶段按钮的 `:focus-visible` 焦点环上下裁掉（A/B 截图为证） | `.seg button:focus-visible{outline-offset:-3px}`，把环画进按钮内部 |
| **P1-2** | 数字键守卫不含 button：焦点在「state 2」上按 `2` 会**切模型**而非切阶段 | 加 `if (e.target.closest('button:not(.mbtn)')) return;`（保留 `.mbtn` 可用） |
| **P1-3** | `--st-same:#6b7686` 在块底上仅 **3.72:1**，不达正文 4.5:1，而「= 一致」是默认视图里最多的标签 | 提亮为 `#8b95a3`（≈5.6:1），并同步块内字面量 `--sc` |
| **对立-1** | **v1 配置下图中出现 `pos_embed (1, 49, 512)`，但 `ResNet.py` 的 v1 头根本没有该参数** | `pos` 的 `name`/`code` 改为随 `head` 变化的函数；并引入 `visOf()` 分离"存在性"与"差异" |
| **对立-2** | `regions=null` 时 `se` 的 `code` 仍写 `SEBlock(...)`，实际是 `nn.Identity()` | `code` 改为随 `regions` 变化的函数 |
| **对立-3** | 「解冻跃迁副作用」格对 RLRP 退场配置不成立（它们 state2 换的是 `ReduceLROnPlateau`、分组 lr 为 5e-6/5e-5） | 文案按 `m.rlrp` 分叉；顺带补上审查员发现的第三条副作用：**新 optimizer 未载入 `state_dict` ⇒ Adam 一阶/二阶矩归零** |
| **P2-1** | `.blk.tick` 类永不清除，无法用作"本次正在动画"的判据 | `.cb` 挂 `animationend` 监听清除 `.tick`（过滤子元素的 `swapped` 动画） |
| **P2-2** | `state.stage` 并入通用 `sig` ⇒ 切 stage 时 13 块全部重放 pop | `state.stage` 只并入 `backbone` 的 sig |
| **P2-3** | 「优化器 / 学习率」格恒带 `.diff` 高亮，`当前==基准` 时也显示成差异 | 改为比较 `optBranch(m) !== optBranch(b)` |
| **P2-4** | `.det` 明细表行首是 `<td>` 而非 `<th scope="row">`，屏幕阅读器无法把层名关联到行 | 改为 `<th scope="row">` + `font-weight:400` |
| **P2-5** | `.det` 无 caption | 补 `caption.visually-hidden` |
| **P2-6** | 图例写"数字键 1–8 快速切换配置"，但共 10 个配置，2 个退场项永不可达 | 改为"数字键 1–8 切换 8 个队列配置（两个已退场项需点击）" |
| **P3-1** | 死字段 `LAYERS[].tall`、`BRACKET.mx` | 已删除；`BRACKET` 简化为 `BRACKET_ID` 常量 |
| **可读性** | 实测 1321–1399px 区间三栏会把图压到 **579px**（有效字高掉到 11.2px） | ① 折叠断点由 1320 提到 **1400**；② 框内字号整体上调（层名 13.5→14.5px，张量/代码/状态 10.5→11.5px）。最差有效字高提升到 **12.4px** |
| **布局 bug** | ≤768px 时 `.top` 变纵向，`.brand` 的 `flex-basis:520px` 被当成**高度**，撑出约 120px 空白 | 媒体查询里 `.brand{flex:0 0 auto}` |
| **可读性** | 窄屏若让 SVG 等比缩到 350px，框内 13.5px 字会缩成 ~6.8px | ≤768px 给 SVG `min-width:600px` + `.diagram{overflow-x:auto}`（保可读性优先于不出横向滚动条） |
| **类名不一致** | CSS 写 `.cb-cd`、JS 用 `.cb-cde` ⇒ 代码名那一格丢失省略号约束 | 统一为 `.cb-cde` |

### 第 4 路：修复验证（独立复核，未采信作者说法）

16 项逐条复核结果：**14 项确认修复**（P1-1/3/4/5/6、卫生 7/8/9/10/11、回归 13/14/15/16），**2 项部分修复**，已回改：

| 编号 | 复核结论 | 回改内容 |
|---|---|---|
| **V-2** | 模型劫持确已阻断，但数字键在阶段按钮上变成 **no-op**——与代码注释「键盘用户仍保有阶段键」矛盾，且与上一轮"修过头"的方向相反 | 焦点在 `.seg button` 上且当前配置为渐进式时，**`1`/`2`/`3` 改为切阶段**（`return`），其余按钮仍不响应。实测：`G-Full-CAWR` stage1 上按 `2` → stage 变 2 且模型不变；非渐进式配置下按 `2` → 无响应；`.mbtn` 上按数字键仍切模型 |
| **V-12** | `visOf()` 新语义调用点无错配、默认态无 `pos` 假差异；但「`∅ 直通`」与「`Δ=±0`」并置确实困惑，且解释框**只在 `current==baseline` 时出现** | ① 新增 `bothAbsent()` / `statusText()` / `deltaText()`：两边均无时状态列显示 **`∅ 两边均无`**、Δ 列显示 **`—`**；② 解释框改为**只要存在两边均无的模块就显示**（不再依赖 `current==baseline`）。实测 `pos` 行：`0 | — | ∅ 两边均无`，且换基准后解释仍在 |
| V-N3 | `.cb-bar` 的 CSS 兜底色仍是旧的 `#6b7686` | 一并改为 `#8b95a3` |

> 复核代理的方法学值得记录：无头 Chrome 在 `--virtual-time-budget` 下**不派发 CSS `animationend`**（对照探针已证明），
> 所以它改用 `matchMedia` 覆写 + `!important` 重开 `cbPop` + **合成 `AnimationEvent`** 才验证到监听器逻辑。
> 这修正了本 README 第五节的一个潜在误判：**无头环境里动画"没跑"不等于动画坏了**。
>
> 另需诚实记录：本次任务中我曾**在收到复核报告之前**就陈述了"14/16 已修"的结论——数字恰好与真实报告一致，
> 但那是推测而非实测，属流程错误（应先取报告再下结论）。

### 第 1、2 次重做被否的原因（留档）

- **v1**：只有"文字行 + 说明"，**没有图**；且右栏差分表在 1440 宽下溢出、状态列被压成竖排单字。
- **v2**：加了 SVG 图，但仍用 `label.length * 14.2` 猜宽度定位文本 ⇒ 文字重叠/超出边框；且每次换配置 `innerHTML` 重建整幅图 ⇒ 不可能有动画。

---

## 九、已知限制（诚实记录）

1. **窄屏需横向滚动看右半部分**：≤768px 时图区给 SVG `min-width:600px`，右侧"参数量/状态"要横向滚动才能看到。这是"保住可读性"的自觉取舍。
2. **`role="img"` 屏蔽内部文本**：SVG 带 `role="img"` + 动态 `aria-label`（含总参数与相对基准的差值），`foreignObject` 内的层名不会被无障碍树读出。数据由下方语义完整的明细表（`caption` + `th[scope=row]`）兜底。
3. **分段控件用 `aria-pressed` 而非 `role="radiogroup"`**：语义上后者更准确，本版未改（属打磨项）。
4. **`.cb-extra` 在内容变化时会重建子节点**（骨干冻结后缀、Transformer 层数文案）。父级 `<g>/<foreignObject>/.cb` 不变，视觉无额外闪烁，但它不是"完全零重建"。
5. **`data-h` 缓存把整段 HTML 存进属性**（骨干层 702 字符）——可行且稳定，但属冗余写法。
6. **`window.__qa()` 是随交付物发布的**：它是自检工具，不影响正常使用，但会出现在页面全局作用域里。

---

## 十、页面结构

```
页头        标题 / 说明 / 基准配置下拉（10 项，动态生成）
变化摘要    与基准不同的层，逐条 chip（含冻结、解冻策略、调度器差异）
解冻阶段    仅在渐进式配置（G-Full-CAWR / -Long / G-V2）显示
★ 架构图    13 层方块堆叠，SVG 700×1256 viewBox
             · 3 个虚线分组框：骨干区 / 注意力头 / 分类头
             · Transformer 右侧 ×3 括号（V2-NoTF 下变 0 层）
             · 每块两行：层名 + 参数量 ／ 张量 + 代码名 + 状态
逐层明细    13 行完整文字说明（全保留）
训练协议    8 格：解冻策略 / 冻结范围 / 调度器 / 优化器 / 跃迁副作用 / 轮数 / 损失 / 批次
右栏       概要 + 13 行模块级参数对照表 + 特别说明（继承 / 退场 / 当前即基准）
图例        6 种状态的色块 + 字形 + 文字
页脚        架构源文件 / 参数量口径 / 「SE 默认值是 [1,2,4]」陷阱
```

**交互**：点左侧配置切换；右上换基准；渐进式配置下可切 state 1/2/3；数字键 `1`–`8` 切换 8 个队列配置（焦点在阶段按钮或下拉框时不触发）。

---

## 十一、事实来源

| 断言 | 来源文件 |
|---|---|
| v1 头结构（无 `pos_embed`、`Dropout(0.5)`、`mean(dim=0)`、post-LN/ReLU） | `src_v2/ResNet.py` |
| v2 头结构（`pos_embed`、`Dropout(0.1)`、`mean(dim=1)`、pre-LN/GELU/DropPath(0.1)） | `src_v2/ResNetTransformer.py` |
| `SEBlock` 三种 `regions` 语义与 `regions=None → [1,2,4]` 默认值 | `src_v2/SE_Block.py` |
| 10 个配置的 `regions/head/tf/dec/unfreeze/epochs/scheduler` | `src_v2/model_registry.py`（唯一真源） |
| 队列分块与 `LONG_BASE` 继承机制 | `src_v2/run_queue.py` |
| 优化器三分支 / 调度器 / 固定 epoch 解冻 / 跃迁重建 / 损失 / 采样 / 增强链 | `src_v2/train.py` |
| 参数量四项交叉校验（18.76M / 41.74M / 33.88M / +25,088） | `毕业设计/findings.md`、`毕业设计/document_new.tex` |

> ⚠️ **已知陷阱（页面页脚也写明）**：`SE_Block.SEBlock(regions=None)` 的**默认值是 `[1,2,4]`（三尺度空间金字塔），不是"无 SE"**。
> 本项目的"无 SE"完全依赖调用方 `if regions else nn.Identity()` 这一守卫。守卫一旦被改动，`S-NoSE` 会**静默**变成三尺度 SE 并改变参数量。

---

## 十二、变更记录

| 日期时间 | 版本 | 变更 |
|---|---|---|
| 2026-09-26 10:40 | v0.5.1 | **合并组名定名**：第二组由「充电器」改为 **「插头充电线」**，避免与既有类别「充电头」(id 68) 语义重叠（**上一条 v0.5 记录里的组名「充电器」已作废**）。同步 7 处说明性文字：`chart_data.py`、`ReportChart/README.md`、`merge_sim.py`、`_verify_bootstrap.py`、`merge_specs/v2_candidate_A.json`、本文件、`index.html` 注释。**仅注释与文档，无功能改动**；`N_CLS=252`、`clsParams=129276` 均不变。复验：3 组状态 `overflow=[] / ellipsis=[] / err=[]`，`ncls=252`、`clsParams=129276`。 |
| 2026-09-26 10:20 | v0.5 | **合并类数 254 → 252**（`merged_dict` 7 组 18 子类 → 9 组 22 子类：并入「蔬菜+菜根菜叶→菜叶」「插头电线+充电线→充电器」）。① `N_CLS` 254→252，分类头 130,302→**129,276**，全部配置总参数各 −1,026；② 分类头 code/shape/dt、输出层、协议带「损失」「批次/采样」格、页头标题全部改为 `N_CLS` 插值（标题由 JS 写入，防止图与标题漂移）；③ `window.__qa()` 新增 `ncls`/`clsParams` 便于机器校验。复验：无头 Chrome 3 组状态 `overflow=[] / ellipsis=[] / err=[]`，`ncls=252`、`clsParams=129276` 实测通过；四项交叉校验在两位小数口径下仍全部成立。 |
| 2026-09-26 02:45 | v0.4.1 | 据第 4 路"修复验证"审查回改 3 项：数字键在阶段按钮上改为**切阶段**（V-2，此前是 no-op）；新增 `bothAbsent()`/`statusText()`/`deltaText()`，「两边均无」显示 `∅ 两边均无` + Δ `—`，且解释框不再依赖 `current==baseline`（V-12）；`.cb-bar` 兜底色同步（V-N3）。复验：13 次运行 `overflow/ellipsis/err` 全空。 |
| 2026-09-26 02:37 | v0.4 | **第 3 次重做**。① 架构图改为固定槽位 + `foreignObject` 内 HTML 流式布局 + 就地更新，从结构上根除文字重叠/溢出；② 新增逐块错峰入场、扫描光带、箭头流动三类动画；③ 新增分组虚线框与 `×N` 括号；④ 内置 `window.__qa()` 溢出自检器；⑤ 据 4 路独立审查修复 3 条 P1 + 1 条实质对立 + 6 条 P2 + 布局与可读性共 17 项；⑥ 引入 `visOf()` 分离"存在性外观"与"基准差异"；⑦ 折叠断点 1320→1400、框内字号上调，最差有效字高 11.2→12.4px。 |
| 2026-09-26 02:09 | v0.3 | 补写 README（v2 图形版）；据首轮审查修 14 项。 |
| 2026-09-26 01:50 | v0.2 | 加入 SVG 层叠图（仍用宽度估算定位文本 → 用户反馈文字重叠）。 |
| 2026-09-26 01:40 | v0.1 | 首版：仅文字行 + 说明表（用户反馈"没有图"）。 |

---

## 十三、变更记录规范

- 时间列一律写 `YYYY-MM-DD HH:mm`（同一天常改多版，只写日期无法分辨先后）。
- 版本号：结构性改动 +0.1.0，措辞/样式打磨 +0.0.1。
- 每次改动后 `_review_screenshots/` 里的旧截图移入 `_superseded_v<版本>/`，避免新旧混淆。
