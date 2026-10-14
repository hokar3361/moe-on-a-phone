# DESIGN(事実と試行のログ)

## 0. 前提(2026-10-14)

- ユーザーの指示:iPhone でこの仕組みを動かしたい。大きすぎるなら Q1 などでよい。小さい MoE でもよい。別リポジトリ・別の作業ディレクトリで進める。
- 端末:iPhone 16 Pro Max、8 GB、256 GB。アプリの上限は 5〜6 GB 程度と見込む [仮説、要確認]。
- Mac の作業ディレクトリ:`~/moe-on-a-phone`。モデルは `~/models/qwen36-35b/`(リポジトリには入れない)。

## 1. MacBook での準備(2026-10-14)

- 開発機:MacBook(M5 Max、128 GB、macOS 26.6.1、Xcode 27.0)。端末:iPhone 16 Pro Max(iPhone17,2、iOS 26.6.2、USB 接続、開発者モード有効)。
- 署名:有料の Apple Developer Program のチーム(Individual)。自動署名で `increased-memory-limit` と `extended-virtual-addressing` の両方が付いた [事実]。
- llama.cpp:上流 `fcac19079f3f2985fe74ed845f0880e7e9a8030a`(2026-10-14)を `third_party/llama.cpp` に置く(git には入れない)。
  `./build-xcframework.sh ios-device ios-sim` で xcframework を作る。上流は Qwen3.8-Flash-Next の arch(`qwen4exp`)に対応済み。
  上流の `lazy_mode`(行を必要時に読む)は arch が指定したテンソル向けで、エキスパートのキャッシュとは別物。
- 計測アプリ:`app/`(xcodegen の `project.yml`、bundle id は `app/Config/Local.xcconfig` で各自設定)。引数付きで起動すると自動で計測し、
  結果を Documents の `results.jsonl` に追記、ログを stderr に出す。
- 端末の値 [事実]:entitlement ありで `os_proc_available_memory()` = 5.99 GiB(読み込み前)。
  Metal の `recommendedMaxWorkingSetSize` = 5726 MB。

## 2. モデル(`download_models.sh` で取得、sha256 を確認)

| モデル | ファイル | 大きさ | 1 語あたりのエキスパート |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | UD-IQ2_XXS / UD-Q2_K_XL | 11.82 / 12.57 GB | 40 層 × 8/256、moe_ff 512、hidden 2048 |
| Qwen3.8-Flash-Next(unsloth rev `766911a6`) | UD-IQ1_S(3 分割) | 72.55 GB | 48 層 × 10/512、moe_ff 640、hidden 2560、共有エキスパート 1 |

- Qwen3.8-Flash-Next のエキスパートは 48 × 512 × 3 × 2560 × 640 ≈ 121B パラメータ。1 語で使うのは 48 × 10 本 ≈ 2.4B パラメータ [計算]。
  メモリに置けるのは 5 GB 程度なので、ほとんどが SSD 読みになる。1 語あたりの読み出し量はファイル内のエキスパートの量子化の型で決まるので、
  ダウンロード後に GGUF のテンソル型から計算する。

## 3. 比較元:素の llama.cpp(上流 `fcac1907`)、Qwen3.6-35B-A3B UD-IQ2_XXS、iPhone 16 Pro Max

計測アプリ `app/` で、引数付きで起動して測った。n_ctx 2048、スレッド 4、mmap。チャットの書式は ChatML で、思考は空の `<think>` ブロックで止める。
greedy で最大 400 トークン、EOG で止める。decode tok/s は 1 トークンずつの `llama_decode` の合計時間から出す。

| 設定 | コード(生成トークン数) | 文章(400) | 備考 |
| --- | --- | --- | --- |
| 全層 GPU(既定) | 失敗 | 失敗 | `kIOGPUCommandBufferCallbackErrorOutOfMemory`。10.4 GB の mmap バッファを GPU が抱えきれない(上限 5.7 GB) |
| エキスパートだけ CPU(`--cpu-moe`) | 2.73 tok/s(198) | 3.79 tok/s | 非エキスパートは GPU |
| 全部 CPU(`--ngl 0`) | 3.64 tok/s(217) | 5.04 tok/s | |

- 3 つの設定は上の順に続けて走らせたので、後のものほどページキャッシュが温まっている [注意]。比較の手順を揃えるまで、この差は確定しない。
- CPU と GPU では計算の細かい差で greedy の出力が分かれる(コードの生成トークン数が違う)。
- アプリの footprint は 0.3〜0.4 GiB のまま。mmap したファイルのページは footprint に数えられない [事実]。
- エキスパートの量:エキスパート 1 個(1 層ぶんの gate/up/down)が 0.96 MB、1 語あたり 40 層 × 8 個 = 307 MB [GGUF から計算]。
  全体は 10.09 GB(MTP 層を含む 41 層)。非エキスパートは 1.72 GB。

## 4. iPhone 16 Pro Max のストレージ(`--iotest`、F_NOCACHE の pread、ファイル内のランダムな位置)

| ブロック | 1 スレッド | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| 256 KiB | 991 MB/s | 1662 | 1935 | 1948 | 1958 |
| 1 MiB | 1428 | 1986 | 2018 | 2009 | 1994 |
| 4 MiB | 1789 | 2001 | 1999 | 2001 | 2006 |

- 4 スレッドで約 2.0 GB/s に達して頭打ち。Qwen3.6 IQ2_XXS の 1 語ぶん(307 MB)を全部読んでも約 0.15 秒なので、I/O だけなら約 6.5 tok/s 出る。
  今の比較元(3.6〜5.0)は I/O の上限より下にいる。

## 5. Qwen3.8-Flash-Next UD-IQ1_S の中身(GGUF から)

- エキスパート:39.85 GB。1 個(1 層ぶん)1.62 MB、1 語あたり 48 層 × 10 個 = 778 MB。gate/up は IQ1_S(34 層)と IQ2_XXS(14 層)、down は IQ4_NL。
- `per_layer_token_embd.weight`:28.8 GB(IQ4_NL、3.2 億行 × 160)。行単位で引く表で、引き継ぎメモの仕組み 4 の対象。
- その他の密な重み:約 3.9 GB(attn_qkv Q5_K、ssm_out Q6_K、hc_* Q8_0 など)。
- 見積もり [計算]:エキスパートをほぼ全部ストレージから読むなら 778 MB ÷ 2.0 GB/s ≈ 0.39 秒で、I/O だけで上限は約 2.5 tok/s。
  密な 3.9 GB がメモリから追い出されると毎語読み直しになるので、これをメモリに置き続けられるかが最初の分かれ目。アプリの上限が 6 GB なので、
  エキスパートのキャッシュに回せるのは 1〜2 GB(全体の 3〜5%)しかない。

## 6. Edge0 の iOS エンジン(github.com/Edge0-AI/Edge0 `c7ee73d5`、`ios/`)

- MLX Swift 製。35B はチェックポイントを並べ替えて(repack)、エキスパート 1 個を 1 回の連続読みにし、層ごとに 1 ファイルにする。アプリは約 20 GB。
- 公開値:iPhone 16 Pro(iOS 26.6.2)で 35B が prefill 4.9 tok/s、decode 6.4 tok/s。
- 条件の違い:エキスパートは 1 語あたり 4 個(元は 8 個)、int4、2 語目以降はサンプリング(temperature 0.7、top-k 64、top-p 0.95、repetition penalty 1.1)。
  同じ方法で比べるには、greedy に直して同じプロンプトで測る必要がある。

## 7. iPhone 17e に移行(2026-10-14)と最初の成果

端末:iPhone 17e(iPhone18,5、iOS 26.6.2、256 GB)。アプリの上限 6.0 GiB、Metal の上限 5.7 GB は 16 Pro Max と同じ。
ストレージは F_NOCACHE のランダム読みで 1 MiB ブロック 3.8 GB/s(16 Pro Max の約 2 倍)。

Qwen3.6-35B-A3B UD-IQ2_XXS、思考なし greedy 400、decode tok/s。温度は各プロンプトの前に nominal/fair まで冷ましてから測る。

| 方式 | コード | 文章 | 備考 |
| --- | --- | --- | --- |
| 素の llama.cpp、全部 CPU | 3.93 | 5.19 | 温度未管理 |
| 上流の MoE キャッシュ 3 GB(`--cpu-moe --moe-cache-mib 3000`) | 3.14 | 5.86 | 温度未管理 |
| Edge0 iOS(int4、1 語 4 エキスパート) | 6.36 | 6.11 | nominal → serious |
| **上流キャッシュ 3 GB + 並列 F_NOCACHE pread(パッチ 0001)** | **11.80** | **9.66** | fair → serious |
| 上と同じ + 次の層の予測 6 個 | 10.43 | 8.18 | 外れの読み込み 10.9 GB で遅く、熱くなる |

- 出力は Mac で、上流キャッシュとパッチ版の greedy が一致(120 トークン)。
- キャッシュのヒット率は 89.8%(3 GB)。読み込みにかかる時間は上流 73.7 秒 → パッチ 9.1 秒(2 プロンプト合計)。
- キャッシュ 4.5 GB は GPU のメモリ不足で失敗する。
- 温度が最大の変動要因。冷えていない状態の同じ設定は 7.06 tok/s だった。
- Edge0 とは量子化と 1 語あたりのエキスパート数が違う(こちらは IQ2_XXS で 8 個、Edge0 は int4 で 4 個)。
- CPU 経路でページキャッシュへ先読みする方式(`llama-moe-prefetch.cpp`)は 4.81 tok/s(コード)で、GPU 経路に及ばない。

## 8. 効かなかった案(17e、Qwen3.6 IQ2_XXS、キャッシュ 3 GB + pread を基準に)

| 案 | コード | 文章 | 判定 |
| --- | --- | --- | --- |
| 基準(再測定) | 12.09〜12.69 | 7.74〜9.66 | 文章は温度で振れる |
| MTP 2 トークン下書き | 11.81 | 9.00 | 不採用。3 トークン検証でエキスパートの和集合が増え、ヒット率 94% → 75% |
| MTP 3 トークン下書き | 12.70 | 7.20 | 不採用 |
| GPU 完了をスピン待ち(moe-on-a-desk 0007) | 6.44 | 7.32 | 不採用。P コア 2 つの iPhone では pread と組み立ての CPU を奪う |
| 次の層の予測 6 個 | 10.43 | 8.18 | 不採用。外れの読み込みで遅く、熱くなる |

- Mac では MTP 2 が 37 → 70 tok/s(コード)。端末の制約(キャッシュとコア数)で結論が逆になる。
- 出力の一致:MTP は文章で完全一致、コードは同点付近で分かれる(3 トークン検証で GPU のカーネルが変わるため)。

## 9. デモ画面

引数なしで起動すると、チャット画面になる(`app/MoEBench/Chat.swift`)。モデルを選び、Code / Story を押して Send。
既定はキャッシュ 3000 MiB + 並列 pread。文字が流れ、decode tok/s・温度・メモリが出る。

## 10. Flash-Next を iPhone で読み込むために(2026-10-14)

- iOS でアプリが mmap できるアドレス空間は合計約 48 GiB(17e、`--vatest`:無名の予約 48 GiB まで、46.6 GiB のファイルの次の 21 GiB で失敗)[事実]。
  元の Flash-Next UD-IQ1_S(3 分割、計 67.6 GiB)は llama.cpp がファイル全体を mmap するので `mmap failed: Cannot allocate memory`。
- 対策:`tools/split_experts.py` で「密な重み + 行の表」(30.44 GiB)と「エキスパートだけ」(19.95 + 17.16 GiB)に分け直し、
  loader に `LLAMA_UNMAPPED_EXPERT_FILES=1` を足して、エキスパートだけのファイルは mmap しない(仮の番地を与え、MoE キャッシュが pread で読む)。
  mmap は 30.44 GiB に収まる。プロンプトも MoE キャッシュを通るよう ubatch を小さく(4〜8)する。CPU の repack は中身を読むので切る(`--no-repack`)。
- Mac で元のファイル(普通の mmap)と分け直したファイル(エキスパートは pread だけ)の greedy 60 トークンが一致 [事実]。

## 11. MoE ベル(moe-on-a-desk の 02-glm-ssd 0003 を移植、`LLAMA_MOE_CACHE_BELL=1`)

- 上流のキャッシュは MoE 層ごとにグラフを分け、CPU で slot map を作ってから続きを実行する(1 語 42 分割)。
  ベルでは slot map を GPU の共有メモリに置き、ルーティングの後に `MTLSharedEvent` で合図、CPU が読み込みと slot map の更新をして合図を返すまで GPU が待つ(1 語 2 分割)。
- 元の実装は CPU 側が合図をスピンで待つ。iPhone ではスピンが逆効果だった(§8)ので、`MTLSharedEventListener` の通知で受ける形にした。
- Mac(M5 Max)、Qwen3.6 IQ2_XXS、キャッシュ 3 GB + pread、コード 120 トークン:39.9 → **64.4 tok/s**、出力はベルなしと一致 [事実]。

## 12. iPhone 17e での Flash-Next と MoE ベルの結果

| モデル・構成(17e) | コード | 文章 | 温度 |
| --- | ---: | ---: | --- |
| Flash-Next 125B UD-IQ1_S、エキスパートは mmap しない + pread + ベル、キャッシュ 1.2 GB、ubatch 4 | **1.81** | **1.87** | nominal のまま |
| Qwen3.6 IQ2_XXS、キャッシュ 3 GB + pread(ベルなし、再測定) | 12.21 | 9.61 | nominal → serious |
| 同上 + ベル(通知で待つ)、3 回 | 4.99〜5.08 | 10.13〜11.41 | 文章は fair 止まり |
| 同上 + ベル(スピンで待つ、`GGML_METAL_BELL_SPIN=1`) | 7.51 | 7.97 | |

- Flash:1 語約 545 ms。読み込み約 188 ms(ヒット率 44%、1 語 434 MB、約 2.3 GB/s)。GPU に載るのは密な重み 3.37 GiB + キャッシュ。
  残りの約 355 ms は GPU の計算(1 語 7,200〜7,700 ノード)。Mac(M5 Max)では同じ部分が約 36 ms。
- ベル:1 語のグラフが 4,087 ノード・2 分割になり、組み立ては 5〜8 ms(スピンでは 1.7 ms)。文章は速く涼しいが、ミスの多いコードで GPU の時間が 1 語 200 ms に伸びる
  (読み込みは 35 ms)。原因は未特定(GPU が待ちから再開するときのクロックか、CPU が書いた共有バッファの扱いと推定)。既定にはしない。
- 計測:`GGML_METAL_TIMING=1` で 64 グラフごとに組み立て時間・GPU の時間・壁時計の平均を出す(moe-on-a-desk 0001 を移植)。
