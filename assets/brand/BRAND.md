# BladeX 品牌资产

本目录是 BladeX 标识的**唯一真相源**。所有 SVG/PNG 由 `scripts/gen_brand_assets.py`
从同一份符号几何与同一份字标轮廓生成——**不要手改这里的文件**，改脚本再重跑：

```bash
python3 scripts/gen_brand_assets.py          # 只出 SVG，无依赖
python3 scripts/gen_brand_assets.py --png    # 连 PNG 一起出，需 pip install cairosvg
```

PNG 已提交进仓库，日常使用不需要装 `cairosvg`。

归档（**不要在产品里引用**）：

- `_source-sheet-20260905.svg` — 设计方交付的 1600×900 展示板，一张图含全部变体
- `_source-wordmark-outlined-20260905.svg` — 设计方交付的轮廓化字标原件

---

## 文件清单

### SVG

| 文件 | viewBox | 用在哪 |
|---|---|---|
| `bladex-symbol.svg` | 640×640 | 纯符号，渐变色，透明底。默认符号资产 |
| `bladex-symbol-mono-navy.svg` | 640×640 | 单色 `#0B1220`，浅底 / 印刷 / 单色场景 |
| `bladex-symbol-mono-white.svg` | 640×640 | 单色白，深底 |
| `bladex-app-icon.svg` | 1024×1024 | 圆角方形应用图标（浅） |
| `bladex-app-icon-dark.svg` | 1024×1024 | 圆角方形应用图标（深） |
| `favicon.svg` | 32×32 | 浏览器标签页，透明底 |
| `bladex-wordmark.svg` | 575×166 | 纯字标 |
| `bladex-wordmark-dark.svg` | 575×166 | 纯字标（深底） |
| `bladex-lockup-horizontal.svg` | 622×203 | 符号 + 分隔线 + 字标。**README 用的就是这个** |
| `bladex-lockup-horizontal-dark.svg` | 622×203 | 同上（深底，带背景块） |
| `bladex-lockup-primary.svg` | 1355×349 | 主标：横版 lockup + 两行 tagline |
| `bladex-lockup-primary-dark.svg` | 1355×349 | 同上（深底，带背景块） |
| `bladex-og.svg` | 1200×630 | Open Graph / 社交卡片 |

### PNG（`png/`）

`favicon-16` · `favicon-32` · `apple-touch-icon-180` · `icon-512` ·
`icon-512-dark` · `symbol-512` · `lockup-horizontal-1244` ·
`lockup-horizontal-dark-1244` · `og-image-1200x630`

## 色板

| 名称 | HEX | RGB |
|---|---|---|
| Primary Blue | `#2563FF` | 37 99 255 |
| Secondary Blue | `#4F7CFF` | 79 124 255 |
| Light Blue | `#8AB4FF` | 138 180 255 |
| Deep Navy | `#0B1220` | 11 18 32 |
| Slate Gray | `#687280` | 104 114 128 |
| Divider（辅助） | `#D7DCE6` / 深底 `#243044` | — |

符号使用两条渐变：外弧 `#2563FF → #4F7CFF`，内弧 `#8AB4FF → #4F7CFF`。
深色版字标：`Blade` 用白、`X` 用 Secondary Blue（`#2563FF` 在深底上对比不足）。

## Lockup 比例

全部以**大写高（cap height）**为单位，任何尺寸下比例一致：

```
符号高 1.846 cap | 符号→分隔线 0.862 cap | 分隔线→字标 0.677 cap
分隔线长 2.75 cap | 四周留白 0.485 cap
tagline 1: 0.340 cap，基线下移 0.728 cap
tagline 2: 0.226 cap，基线下移 1.253 cap
```

改尺寸只需改 `lockup()` 的 `cap` 参数。

## 字体

字标已轮廓化，**不依赖任何字体**。其余文字（tagline）用 Inter Medium/Regular，
字体栈 `Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif`。

⚠️ **字标是自有字形，不等同 Inter。** 实测度量与 Inter 有差：x-height/大写高
0.777（Inter 0.711）、`l`/`d` 的上伸部与 `B` 齐平（Inter 应高 3%）。所以：

- 不要用 Inter 去补字、造新字标、或重排字标里的任何字母；
- 需要新字样（比如产品名变体）时回设计方源文件，不要在代码里拼。

## Tagline

与 `README.md` 保持一字不差：

> **Build your private data assets.**
> **Connect all your agents and LLMs like a blade.**

tagline 在 SVG 里仍是活字 `<text>`——**这是有意的**：它是会改的市场文案，不是标识
本体；正文级文字回退到 Helvetica/Arial 可以接受。标识本体（符号 + 字标）已全部
是路径，零字体依赖。

## 使用规则

- **最小保护区** = 符号轨道厚度（约符号高度的 16%），四边等距。
- **最小尺寸**：符号数字端 16px、印刷端 20mm；横版 lockup 数字端最小宽度 120px。
- **不要**改变符号比例、重上色、加描边/阴影/外发光。深色场景用 `*-dark` 变体，
  不要给浅色版垫白色块。
- 渐变 `id` 已按文件加前缀（`bx-sym-*`、`bx-app-*`、`bx-lh-*`……），多个 SVG
  内联进同一个 HTML 页面不会互相覆盖。

---

## 已知偏差与遗留

### 1. e→X 间距（已修）

设计方交付的字标里 `e` 与 `X` 的间距是 90.33，其余字对是 13–15——源于更早那版
活字 SVG 里 `<text x="1010">` 的绝对定位被一起转成了轮廓。生成脚本以纯
`translate` 收敛到 13（`WM_EX_GAP`），**字形几何未做任何修改**。改这个数即可调整。

### 2. 微尺寸下内弧偏淡（未处理）

16px favicon 下 `#8AB4FF` 的内弧在浅色背景上几乎看不见，只剩外弧可读。若在意，
可请设计方出一版微尺寸专用标记：内弧提到 `#4F7CFF`，或按设计板 #15 MICRO MARK
的简化几何。

### 3. 轨迹描摹的小瑕疵（未处理）

外弧路径在刀刃末端（`379,682 → 384,681 → 382,676` 一段）有一处描摹回折，是位图
转矢量留下的自相交小尖。48px 以上肉眼不可见，未擅自修改——几何以设计方
"approved" 为准。做印刷放大件（海报、展板）前请设计方在源文件里清掉这个锚点。
