# moe-on-a-phone

> [!CAUTION]
> **端末がかなり熱くなります。利用は自己責任でお願いします。**
> 35B の MoE モデルを iPhone で連続して動かすと、数十秒で iOS の温度状態が「serious」(高温)に達します。長時間続けて動かさないでください。
> 熱いと感じたら止めて冷ましてください。ケースを付けたままや、充電しながらの実行は避けてください。バッテリーを大きく消費し、ストレージからの読み出しも大量に発生します。
> 無保証です。llama.cpp への改造の副作用は、限られた機種でしか確かめていません。
> このリポジトリのコード・手順・数値を使ったことで生じた端末の故障、データの消失、その他の損害について、作者は一切の責任を負いません。

> [!CAUTION]
> **Your phone will get hot. Use at your own risk.**
> Running a 35B MoE model continuously on an iPhone drives iOS into the "serious" thermal state within tens of seconds. Do not run it
> for long stretches; stop and let the phone cool down if it feels hot; avoid running it in a case or while charging. It drains the battery fast and reads a lot from storage.
> No warranty. The modifications to llama.cpp have been checked on very few devices.
> The authors accept no responsibility for any damage to devices, loss of data, or other harm caused by using the code, steps or numbers in this repository.

**iPhone で 35B の MoE モデルを、量子化を落とさずにどこまで速く動かせるか。**
Mac Studio 向けの [moe-on-a-desk](https://github.com/hokar3361/moe-on-a-desk) の考え方をスマートフォンに持ち込み、llama.cpp のパッチと計測アプリとして公開しています。
**試したのは 1 機種・1 モデルだけ** で、出力の確認も軽いものです。他の機種での結果や改良は、Issue や PR で歓迎します。

[English below](#english)

---

## 試した環境(これだけです)

| | |
|---|---|
| 端末 | **iPhone 17e**(iPhone18,5)、iOS 26.6.2、ストレージ 256 GB |
| アプリが使えるメモリ | 約 6.0 GiB(Increased Memory Limit あり)、Metal の上限 約 5.7 GB |
| モデル | **Qwen3.6-35B-A3B** `UD-IQ2_XXS`(11.82 GB)、[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`、sha256 `627e2b05…441203` |
| ビルドした Mac | Apple Silicon の Mac、macOS 26.6.1、Xcode 27.0、xcodegen、CMake |
| llama.cpp | 上流 `fcac19079f3f2985fe74ed845f0880e7e9a8030a` + `patches/llama.cpp/` |

他の iPhone、他のモデル、他の量子化では試していません。8 GB 未満の端末や、アプリが使えるメモリが少ない場合は動かない可能性が高いです。
(iPhone 16 Pro Max では、パッチ前の比較元だけを測りました。)

## 結果

固定の計測:思考なし(空の `<think>` ブロック)、greedy、最大 400 トークン、decode tok/s(プロンプト処理を除く)。
プロンプトは次の 2 つで固定しています。各プロンプトの前に、iOS の温度状態が nominal か fair に戻るまで待ちます。

- コード:"Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."
- 文章:"Write a short story about a lighthouse keeper who finds a message in a bottle."

| 方式(iPhone 17e) | 量子化 | コード | 文章 |
|---|---|---:|---:|
| 素の llama.cpp、全部 CPU(`ngl 0`)| IQ2_XXS | 3.9 | 5.2 |
| 素の llama.cpp、全層 GPU | IQ2_XXS | GPU のメモリ不足で動かない | — |
| 上流の MoE キャッシュ 3 GB(`--cpu-moe --moe-cache-mib 3000`)| IQ2_XXS | 3.1 | 5.9 |
| [Edge0](https://github.com/Edge0-AI/Edge0) の iOS アプリ(参考)| int4、1 語 4 エキスパート | 6.4 | 6.1 |
| **このパッチ(上流キャッシュ 3 GB + 並列 F_NOCACHE pread)** | IQ2_XXS | **12.1〜12.7** | **7.7〜9.7** |

- 文章の値が振れるのは温度のためです。生成の途中で fair から serious に上がり、速度が落ちます。
- Edge0 は量子化も、1 語あたりのエキスパート数も違います。同じ端末・同じプロンプトで測った参考値で、順位付けではありません。
- 温度を管理していない状態の同じ設定は、7 tok/s 前後でした。

### 出力の確認(軽いものです)

- Mac で、上流の MoE キャッシュとこのパッチの greedy 出力が、120 トークンでバイト単位で一致しました。読み込み方を変えただけで、計算は変えていません。
- iPhone の出力は、コードも文章も内容として自然でした(目視)。perplexity などの定量評価はしていません。

## 何をしたか

- **ミスしたエキスパートを並列 pread で読む**:上流の MoE キャッシュは、ミスしたエキスパートを mmap のポインタから 1 つずつ同期で GPU のバンクにコピーします
  (ページフォールトが 16 KB ずつ順番に起きます)。このパッチは、ファイルから `F_NOCACHE` の `pread` を 512 KB 単位で並列に出し、Metal の共有バッファ(バンク)へ直接読みます。
  読み込みにかかる時間は 73.7 秒 → 9.1 秒(2 プロンプト合計)になりました。
- 計測アプリ(`app/`):引数で自動計測するモードと、動画を撮りやすいチャット画面があります。

試して効かなかった案(MTP の下書き、GPU 完了のスピン待ち、次の層の予測による先読みなど)は [DESIGN.md](DESIGN.md) の §8 にあります。

## 使い方

### 必要なもの

- Apple Silicon の Mac、Xcode(27.0 で確認)、[xcodegen](https://github.com/yonaskolb/XcodeGen)、CMake(`brew install xcodegen cmake`)
- iPhone(開発者モードを有効にしたもの)と USB ケーブル
- **Apple Developer のアカウント**(下記)
- iPhone の空き容量 約 12 GB(モデル)

アプリはソースで配布しています。署名済みのアプリは同梱していないので、各自の Apple Developer のアカウントで署名してインストールします。

#### Apple Developer のアカウントについて

- このリポジトリの数値は、**有料の Apple Developer Program** のチームで署名して測りました。アプリは
  `Increased Memory Limit`(`com.apple.developer.kernel.increased-memory-limit`)と
  `Extended Virtual Addressing`(`com.apple.developer.kernel.extended-virtual-addressing`)の 2 つの entitlement を使います。
- **無料の Apple ID でも** Xcode の Personal Team で自分の iPhone にインストールできます。手順は次のとおりです。
  1. Xcode > Settings > Accounts で Apple ID を追加すると、Personal Team ができます
  2. そのチーム ID を下の `Local.xcconfig` に書きます
  3. 初回は iPhone の「設定 > 一般 > VPN とデバイス管理」で開発元を信頼します
  4. 無料の署名は **7 日で切れる** ので、切れたら入れ直します
- ただし、**無料の Personal Team で上の 2 つの entitlement が付けられるかは確認していません**。付けられない場合は `app/project.yml` の `entitlements` から外せばビルドできます。
  その場合アプリが使えるメモリが減るので、画面のキャッシュの値を小さく(例:1000〜1500 MiB)してください。この構成は **未検証** です。

### 手順

```bash
git clone https://github.com/hokar3361/moe-on-a-phone
cd moe-on-a-phone
scripts/setup.sh                      # llama.cpp を取得し、パッチを当てて iOS 用 xcframework を作る(10 分ほど)
./download_models.sh                  # モデルを ~/models に取得して sha256 を確認(他のモデルも取得します。必要な行だけ残してください)
cp app/Config/Local.xcconfig.example app/Config/Local.xcconfig   # チーム ID と自分の bundle id を書く
cd app && xcodegen generate && open MoEBench.xcodeproj           # Xcode で実機を選んで Run
```

モデルを iPhone のアプリに入れます。

- Finder:iPhone を選び、「ファイル」タブで MoEBench に `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` をドラッグする(11.8 GB、数分かかります)
- またはコマンドで送る:

```bash
xcrun devicectl device copy to --device <UDID> --domain-type appDataContainer --domain-identifier <bundle id> --source ~/models/qwen36-35b/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --destination Documents/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
```

アプリを開き、モデルを選んで Code か Story を押し、Send を押します。最初の 1 回だけ、読み込みに約 15 秒かかります。
画面の上に tok/s、温度の状態、メモリが出ます。**serious になったら止めて冷ましてください。**

計測モード(Mac から起動):`DEV=<UDID> BUNDLE=<bundle id> OUT=/tmp app/run_bench.sh run1 --model Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --cpu-moe --moe-cache-mib 3000 --pread`

## まだできないこと

- **Qwen3.8-Flash-Next(125B、Q1 で 72.5 GB)**:iOS でアプリが mmap できるアドレス空間は合計約 48 GiB で、ファイル全体を mmap する llama.cpp の読み込みでは入りません。
  エキスパートを mmap せず pread だけで読む形に作り変えている途中です。

## ライセンス

このリポジトリ自体のライセンスはまだ決めていません。llama.cpp(MIT)、Edge0(Apache 2.0)、各モデル(それぞれの配布元のライセンス)は、それぞれの条件に従ってください。

---

<a id="english"></a>
## English

**How fast can a 35B MoE model run on an iPhone, without lowering its quantization?** This brings the ideas of
[moe-on-a-desk](https://github.com/hokar3361/moe-on-a-desk) (Mac Studio) to a phone, as a llama.cpp patch and a measurement app.
**Tested on one phone and one model only**, and output quality was checked only lightly. Reports from other devices and PRs are welcome.

### Tested on (only this)

| | |
|---|---|
| Phone | **iPhone 17e** (iPhone18,5), iOS 26.6.2, 256 GB |
| Memory for the app | about 6.0 GiB (with Increased Memory Limit); Metal working set about 5.7 GB |
| Model | **Qwen3.6-35B-A3B** `UD-IQ2_XXS` (11.82 GB), unsloth/Qwen3.6-35B-A3B-MTP-GGUF revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` |
| Build Mac | Apple silicon Mac, macOS 26.6.1, Xcode 27.0, xcodegen, CMake |
| llama.cpp | upstream `fcac19079f3f2985fe74ed845f0880e7e9a8030a` + `patches/llama.cpp/` |

No other iPhone, model or quantization has been tried.

### Result

Fixed metric: thinking off, greedy, up to 400 tokens, decode tok/s; the two fixed prompts above; before each prompt the app waits until the
thermal state is nominal or fair.

| Method (iPhone 17e) | quantization | code | prose |
|---|---|---:|---:|
| llama.cpp, all on CPU (`ngl 0`) | IQ2_XXS | 3.9 | 5.2 |
| llama.cpp, all layers on GPU | IQ2_XXS | out of GPU memory | — |
| upstream MoE cache 3 GB (`--cpu-moe --moe-cache-mib 3000`) | IQ2_XXS | 3.1 | 5.9 |
| [Edge0](https://github.com/Edge0-AI/Edge0) iOS app (reference) | int4, 4 experts per token | 6.4 | 6.1 |
| **this patch (upstream cache 3 GB + parallel F_NOCACHE pread)** | IQ2_XXS | **12.1–12.7** | **7.7–9.7** |

Prose varies with temperature: the phone goes from fair to serious during the run and slows down. Edge0 uses a different quantization and
fewer experts per token; it is a reference point on the same phone, not a ranking.

Quality checks (light): on a Mac, the greedy output of the upstream MoE cache and of this patch is byte-identical for 120 tokens (only how
experts are read changed, not the math). On the phone, the outputs read naturally; no perplexity evaluation was done.

### What it does

The upstream MoE cache copies each missing expert from the mmap into the GPU bank one at a time (faulting 16 KB pages in sequence). This patch
reads missing experts straight from the file into the Metal shared bank with parallel `F_NOCACHE` `pread`s in 512 KB chunks
(time spent filling the cache: 73.7 s → 9.1 s over the two prompts). Ideas that did not help on the phone (MTP drafting, spin-waiting on
the GPU, next-layer prediction) are logged in [DESIGN.md](DESIGN.md) §8.

### How to use

Requirements: an Apple silicon Mac with Xcode (27.0 tested), xcodegen and CMake (`brew install xcodegen cmake`); an iPhone with
Developer Mode on; about 12 GB free on the phone; **an Apple Developer account**. The app is distributed as source only: you sign it yourself.

- The numbers above were measured with a **paid Apple Developer Program** team. The app uses two entitlements, Increased Memory Limit and
  Extended Virtual Addressing.
- **A free Apple ID also works for installing on your own iPhone.** To set it up:
  1. Add the Apple ID in Xcode > Settings > Accounts; this creates a Personal Team.
  2. Put its team ID in `Local.xcconfig`.
  3. On first install, trust the developer in Settings > General > VPN & Device Management.
  4. The signature **expires after 7 days**, so reinstall the app when it does.
- **Whether a free Personal Team can use the two entitlements has not been checked.** If it cannot, remove them from `entitlements` in
  `app/project.yml`. The app then gets less memory, so lower the cache size on the screen (for example 1000–1500 MiB). This setup is **untested**.

```bash
git clone https://github.com/hokar3361/moe-on-a-phone
cd moe-on-a-phone
scripts/setup.sh                      # fetch llama.cpp, apply the patch, build the iOS xcframework (~10 min)
./download_models.sh                  # download to ~/models and check sha256 (it also fetches other models; keep only the lines you need)
cp app/Config/Local.xcconfig.example app/Config/Local.xcconfig   # your team ID and your own bundle id
cd app && xcodegen generate && open MoEBench.xcodeproj           # pick your iPhone and Run
```

Copy the model into the app:

- Finder: select the iPhone, open the Files tab, and drag `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` onto MoEBench.
- Or from the command line: `xcrun devicectl device copy to ... --destination Documents/<file>` (see the Japanese section for the full command).

Open the app, pick the model, tap Code or Story, then Send. The first run loads the model in about 15 s. The screen shows tok/s, the thermal
state and memory. **Stop and let the phone cool down when it reaches serious.**

### Not working yet

- **Qwen3.8-Flash-Next (125B, 72.5 GB at Q1)**: an iOS app can map about 48 GiB of address space in total, and llama.cpp maps whole files.
  We are reworking it to read the experts with `pread` only, without mapping them.

### License

A license for this repository itself has not been chosen yet. llama.cpp (MIT), Edge0 (Apache 2.0) and each model (its own license) keep their own terms.

## Files

- `DESIGN.md` — research log (Japanese): every measurement and trial, including what did not work
- `patches/llama.cpp/` — the llama.cpp patch (against `fcac1907`)
- `patches/edge0/` — the small change used to measure Edge0 with the same prompts
- `app/` — the iOS app (xcodegen); `app/run_bench.sh` runs a measurement from the Mac
- `scripts/setup.sh` — fetch llama.cpp, apply the patch, build the xcframework
- `download_models.sh` — pinned model downloads with sha256 checks
