# moe-on-a-phone

iPhone(16 Pro Max、メモリ 8 GB)で MoE モデルを動かし、Mac Studio 向けに作った仕組み(SSD 直読みのエキスパートキャッシュ、
ルーティングの先読み、MTP の深い下書きと停止規則、n-gram 表の並列読み)をスマートフォンに持ち込むプロジェクト。

## 対象

- モデル候補:Qwen3.6-35B-A3B(総 35B・1 語あたり約 3B、MTP ヘッドあり。unsloth/Qwen3.6-35B-A3B-MTP-GGUF の UD-IQ2_XXS 11.8 GB・UD-Q2_K_XL 12.6 GB)。
- 参考にする既存の取り組み:Edge0-35B-A3B-preview(Qwen3.6-35B-A3B の int4+LoRA+ルーティング予測ヘッド、SSD からエキスパートを読む、MLX、
  Mac mini M4 Pro 24 GB で 14.9〜17.7 tok/s、作業メモリ 2.9 GiB。Apache 2.0)。
- 端末:iPhone 16 Pro Max(8 GB、256 GB)。開発は Mac Studio M4 Max 64 GB(Xcode)。

## 進め方

1. Mac で iPhone と同じ制約(アプリが使えるメモリ 5〜6 GB)を再現し、1 語あたりの読み出し量と速さを測る。
2. llama.cpp(私たちのフォーク)を iOS 向けにビルドし、簡単なチャットアプリで実機に入れて測る。
3. 既存の手段(Edge0 など)を同じ端末・同じ計測で測り、比較元として固定する。

記録は `DESIGN.md`(事実と試行のログ)。
