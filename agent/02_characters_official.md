# 官方角色名

本文件说明官方角色名资料的来源和使用方式。

## 文件用途

- `agent/characters_official.txt`：人工识别片尾演员表的原始备份，防丢失、可追溯。
- `agent/characters_official.json`：由 txt 结构化而来，未改写原始内容，是代码和模型实际读取的角色名入口。

ASR 校正、DeepSeek 校正和字幕翻译阶段，需要角色名、演员名或别名时，均以 `agent/characters_official.json` 为准。

## JSON 字段

每条数据结构示例：

```json
{
  "actor_raw": "鄭少秋",
  "role_raw": "徐绍良",
  "aliases_raw": ["阿良", "良仔"],
  "actor_traditional": "鄭少秋",
  "role_traditional": "徐紹良",
  "aliases_traditional": ["阿良", "良仔"],
  "actor_simplified": "郑少秋",
  "role_simplified": "徐绍良",
  "aliases_simplified": ["阿良", "良仔"]
}
```

- `*_raw`：保留人工片尾表原样
- `*_traditional`：供粤语底稿和港式中文校正使用
- `*_simplified`：供最终简体中文字幕使用
- `aliases_*`：片尾斜线异名或用户人工确认称谓

## 使用原则

- 不要因为 wiki、网友剧情简介或模型猜测覆盖 `characters_official.json` 中的角色名。
- 片尾演员表自身存在的同音字、近音字、前后异写，应保留为别名或待人工确认，不要让模型自行创造新角色名。

