# VanityFair SRT v3

《大亨》（1977 TVB）字幕工程的第三版架构。当前里程碑只解决一个风险最高、也最适合离线反复实验的问题：把已有豆包 `result.json` 的逐词时间戳切成可审查的字幕段。

## 当前闭环

```text
cache/doubao/{episode}_result.json
  -> normalize_doubao_words
  -> adaptive word-gap profile
  -> speech islands
  -> subtitle cut candidates + scoring
  -> greedy cut + repair + quality flags
  -> cache/segments + cache/reports + lab debug views
```

已实现：

- 读取并校验 `body.result.utterances` 与 `words[]`
- 毫秒转秒、空 word 清理、utterance 标点回贴到 `WordToken.trailing_punct`
- 全集 word gap 分位数与自适应 weak/soft/strong gap
- 长停顿 speech island 切分与粤语语气词保护
- 逐 boundary 候选生成、可解释评分、软/硬上限切分
- 过短/语气词碎片修复、字幕尾部安全延长、时间防重叠
- `too_long_*`、`too_short_duration`、`too_fast_reading`、`particle_fragment`、`cross_long_silence` 等 flags
- 原子 JSON 写入，候选 CSV、字幕预览 CSV、报告 JSON；lab 模式在 artifact-tool 运行环境可额外导出 XLSX

当前明确未实现：

- TS -> WAV
- Cloudflare R2 upload
- Doubao submit/query
- DeepSeek / LLM diagnosis
- Yue draft
- Human review import
- Traditional/Simplified translation
- final SRT output

`vf_srt/ingest/` 只有会明确抛出 `NotImplementedError` 的占位接口。本阶段不会读取密钥、访问网络、调用 ASR 或上传文件。v1/v2 只作为后续迁移参考，不由 v3 修改。

## 使用

项目切分入口：

```powershell
python -m vf_srt --episodes 09-12 --run-until segmented
python -m vf_srt --episodes 09,10,11 --config config.example.yaml --run-until segmented --overwrite
```

前 8 集参考画像与本地诊断：

```powershell
# 已有 profile 时直接复用，不扫描原始 SRT
python -m vf_srt --run-until reference-profile

# 显式从本地 01–08 人工字幕重建
python -m vf_srt --run-until reference-profile --rebuild

# 先切分，再生成纯本地、只给提示且不修改 ASR 的诊断
python -m vf_srt --episodes 09-12 --run-until local-diagnosis --overwrite
```

内置参考画像位于：

- `reference/profile/reference_srt_profile.json`
- `reference/profile/reference_srt_profile.md`

它们由 01–08 人工精校 SRT、官方演员表和剧情资料离线提取，用于本地诊断的人名/称谓证据、术语与风格参考、前 8 集已确认表达习惯，以及后续 prompt 的压缩上下文来源。**不得用来补写当前集不存在的对白。**

完整的 `reference/simplified_human/*.srt` 默认由 `.gitignore` 排除，避免误提交到 public 仓库。profile 本身包含少量人工字幕例句；公开仓库维护者应自行评估版权风险，但工具不会自动删除 profile。

实验入口（额外尝试生成调试 XLSX）：

```powershell
python -m vf_srt.lab.segmentation_lab --episodes 09-12 --overwrite
```

核心切分只使用 Python 标准库，不需要安装依赖。YAML 读取器支持当前示例所用的嵌套键/标量子集。调试 XLSX 是可选视图：设置 `VF_SRT_NODE` 和 `VF_SRT_NODE_MODULES` 指向已提供 artifact-tool 的 Node 运行环境即可生成；缺少该运行环境时 JSON/CSV 主报告仍完整输出，不影响切分。

每集输出：

- `cache/normalized/{episode}_utterances.json`
- `cache/segments/{episode}_segments_raw.json`
- `cache/reports/{episode}_gap_profile.json`
- `cache/reports/{episode}_segmentation_report.json`
- `cache/local_diagnosis/{episode}_local_pre_review_diagnosis.json`（运行 local-diagnosis 时）
- `lab/{episode}_cut_candidates.csv`
- `lab/{episode}_segments_preview.csv`
- `lab/{episode}_segmentation_debug.xlsx`（lab 模式、可选）

未指定 `--overwrite` 时会复用已有 normalized/segments cache。需要比较新算法时务必加上 `--overwrite`。

## 最终主流程（后续里程碑）

```text
Doubao ASR
-> normalize_doubao_words
-> subtitle_segmentation
-> local_review_flags

-> pre_review_diagnosis
-> yue_draft_auto_lines
-> yue_review_xlsx
-> import_yue_review
-> yue_master.json

-> traditional_context
-> traditional_viewer_lines
-> simplified_context
-> simplified_viewer_lines
-> viewer_review_xlsx
-> import_viewer_review
-> viewer_master.json

-> build_srt
```

前置 ingest 链路后续再从已跑通的旧版迁移。这样可避免在切分实验阶段同时引入环境变量、R2 配置、API key 与命名冲突。
