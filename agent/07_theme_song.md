# 主题曲规则

主题曲固定歌词以 `agent/theme_song.json` 为准。

## 使用原则

- Corrected 阶段：高置信命中主题曲时，使用匹配歌词行的 `traditional`。
- Translation 阶段：高置信命中主题曲时，使用匹配歌词行的 `simplified`。
- 主题曲不是普通对白，不要意译、润色或改写歌词含义。
- 只修正已经被 ASR 识别出的歌声片段；不要凭空补充 ASR 没有识别出来的歌词行。
- 如果主题曲漏唱、漏识别或跳过某几句，不要自动插入缺失歌词。
- 如果主题曲重复唱某一句，可允许多个字幕片段匹配同一 `theme_song.json` 歌词行。
- 低置信命中只能作为参考，不得强制替换为歌词。

## 高置信匹配含义

高置信主题曲匹配应由本地程序根据 ASR 文本、歌词相似度、片头/片尾位置和相邻片段共同判断。模型只应信任已经明确提供的 `theme_song_index`、`theme_song_traditional`、`theme_song_simplified` 等匹配结果。

