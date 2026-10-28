# yencast-web-v08 data branch

毎時バッチ (YenCast predictor / ForexLens) が CSV を push するデータ専用ブランチ。
Render は master だけをデプロイするので、ここへの push ではデプロイ (ビルド枠) を消費しない。
web (app.py) は GitHub raw の data ブランチからこの CSV を読む。
