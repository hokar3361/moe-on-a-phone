# 引き継ぎ(2026-10-14、Mac Studio のセッションから)

この作業を MacBook(Xcode と Apple Developer のアカウントがある端末)で続けるためのメモ。

## 目的と端末

- iPhone 16 Pro Max(メモリ 8 GB、ストレージ 256 GB)で MoE モデルを動かし、Mac Studio 向けに作った仕組みで速くする。
- アプリが使えるメモリは 8 GB 機で 5〜6 GB 程度と見込む [仮説]。`com.apple.developer.kernel.increased-memory-limit` と
  `com.apple.developer.kernel.extended-virtual-addressing`(12 GB のファイルを mmap するため)の entitlement が要るはず [要確認]。

## モデル

- 第一候補:Qwen3.6-35B-A3B(総 35B、1 語あたり約 3B、MTP ヘッドあり、llama.cpp 対応)。
  `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` の revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`:
  - `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` 11.82 GB、sha256 `627e2b05f83088448e27861ec392d56efc2b178783cae5adfa4ad2d988441203`
  - `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` 12.57 GB、sha256 `ed7cda7e38985b4fcff76475865135039641d2bfbac3c169df15ca770f37fb0c`
  - (`UD-IQ1_M` 11.37 GB もあるが、非エキスパートを高い精度で残しているため IQ2_XXS とほとんど変わらない)
- 比較元(既存の手段):Edge0-35B-A3B-preview(huggingface.co/Edge0/Edge0-35B-A3B-preview、Apache 2.0)。同じ Qwen3.6-35B-A3B を int4+LoRA にし、
  次のルーティングを予測するヘッドで SSD からエキスパートを読む。MLX、iOS 版エンジンあり(GitHub: Edge0-AI)。公開値は Mac mini M4 Pro 24 GB で 14.9〜17.7 tok/s、作業メモリ 2.9 GiB。
  1 語あたりのエキスパート数が 4(元は 8)と書かれている点は品質との兼ね合いで要確認。

## 計測(固定。Mac Studio の計測と同じ考え方)

- 思考なし、greedy 400 トークン、decode tok/s。プロンプトは
  "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."(コード)と
  "Write a short story about a lighthouse keeper who finds a message in a bottle."(文章)。
- 既存の手段(素の llama.cpp の iOS ビルド、Edge0 の iOS エンジン)を同じ端末・同じ方法で先に測り、比較元として固定する。
- 量子化を下げて速くしたことにはしない(同じファイル同士で比べる)。出力が変わる仕組みは成果にしない。

## 持ち込む仕組み(Mac Studio で実証済み)

1. **SSD 直読みのエキスパートキャッシュ**:非エキスパートは GPU、エキスパートは LRU のキャッシュ(Metal の共有バッファ)に置き、ミスはファイルを
   `F_NOCACHE` の `pread` で並列に読む(ページキャッシュとの二重持ちをやめる)。GLM-5.3(101 GB)を Mac 64 GB で動かしたときの律速は SSD の帯域だった。
2. **次の層のルーティング予測で先読み**:今の層の入力に次の層のルーターを当て、上位 6 個を別スレッドで読む(GLM で当たり 84.5%)。広げすぎると外れが帯域を食う。
3. **命令列の中での合図**:ルーティングの直後に GPU から CPU へ合図し(MTLSharedEvent)、共有エキスパートを計算している間に CPU がミスを読む。
4. **n-gram/行単位で読む表の並列フォールト**:llama.cpp の `llama_prefetch` は Apple 系で何もしない(Linux と Windows の実装だけ)。
   要求された行のページを GCD の `dispatch_apply` で並列に触ると、冷えた行の読み待ちが消えた(Qwen 125B で初めての文章 +11%)。iOS でも同じはず。
5. **MTP の深い下書きと停止規則**:メモリに収まる部分が多い場合に効く。ストレージ律速の場合は読み出しが増えて逆効果になりうる(GLM で確認)。

上の 1〜4 のコードは、ユーザーの非公開の作業リポジトリ(llama.cpp フォークのパッチ列)にある。必要ならユーザーに場所を確認すること。

## 最初の手順(案)

1. llama.cpp(上流の `build-xcframework.sh`)で iOS 用の xcframework を作り、上流の `examples/llama.swiftui` 相当の最小アプリで
   Qwen3.6-35B-A3B UD-IQ2_XXS を mmap で動かす(ファイルは Finder の「ファイル」共有でアプリの Documents へ)。動くか、何 tok/s か、メモリの上限で落ちないかを見る。
2. 同じ端末で Edge0 の iOS エンジンを測る。
3. 1 にエキスパートキャッシュ(上の 1〜4)を入れ、1 語あたりのストレージ読み出し量・ヒット率・tok/s を測る。

## MacBook に置いた資材(2026-10-14)

- `~/moe-on-a-phone`:このリポジトリ。
- `~/moe-on-a-phone-work`(リポジトリには入れない作業用の資材):
  - `patches/01-mac125b`(llama.cpp 781dbc5 の上の 52 本)、`patches/02-glm-ssd`(SSD 直読みのエキスパートキャッシュ・先読み・合図、5 本)、
    `patches/03-prefetch-apple`(Apple 系の行の並列フォールト、1 本)。
  - `llama.cpp`:上流を 781dbc5 から 3 系列すべて当てたブランチ `phone`(`apply.log`、全件きれいに当たった)。
  - `models`:Qwen3.6-35B-A3B の UD-IQ2_XXS と UD-Q2_K_XL(`dl.sh` が sha256 を照合、`dl.log` に結果)。
- この MacBook は M5 Max 128 GB・Xcode 入り。cmake と gh は入っていない(cmake は pip か Homebrew で入れる)。
