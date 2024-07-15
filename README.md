# Educational-Big-Data-Projects
論文閱讀器

介紹影片
https://www.youtube.com/watch?v=3jrApIVDhws

下關鍵字搜尋論文，呈現論文摘要以及關鍵字出現段落，下載符合需求的論文
- 選擇要閱讀的論文
- 可回答問題
- 可摘要選取的段落
- 可透過選取的段落來搜尋相關論文

## 功能實作簡述

搜尋 : arxiv api

摘要 : gimini api

回答問題 : 同時給 chat-gpt3.5 該篇論文和問題，要求以該篇論文的內容回答

段落搜尋 : 串接 tavily 和 arxiv api

## files:

app2.py - 主程式，實作了搜尋、摘要和回答問題等功能

search_tavily.py - 實作了段落搜尋功能

integrated_search2.html - 呈現搜尋功能

upload.html - 呈現上傳論文、摘要、回答問題和段落搜尋等功能

其餘為練習和測試用途
