# CMP 170HX ×2 で Qwen3.8-Flash-Next W4A16 を動かす

[English](README.md)

マイニング用カード **NVIDIA CMP 170HX** 2枚で、**Qwen3.8-Flash-Next**（W4A16）を vLLM で
動かす構成です。ホストのメインメモリは **92GiB** で、公開レシピの要求より少なめです。

**1本あたり散文で約100〜110 tok/s、コードで約170 tok/s・同時4リクエストで合計約390 tok/s・
コンテキスト262,144トークン（モデル本来の上限。20万トークンのニードルも正解）・
ツール呼び出し対応。**

このページの数値はすべてこのマシンで実測したものです。測り方は[計測方法](#計測方法)に、
生データは `results/` にあります。

> **既定のモデルは heretic-2（拒否応答を取り除いた版）に変わりました。**[結果](#結果)の表を
> 測った Vnimanie の W4A16 ではありません。既定値も `--max-num-batched-tokens 8192`、KV は
> 1枚あたり17.6GiBに変わっています。[モデル：heretic-2](#モデルheretic-2) と
> [バッチトークン数とKVサイズ](#バッチトークン数とkvサイズ) を参照してください。
> `MODEL_DIR=$BASE_MODEL_DIR` を指定すれば、今も Vnimanie のチェックポイントで動かせます。

---

## ここが違う

公開されているFlash-Next W4A16のレシピ（[loktar00](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe)、
[alesha-pro](https://github.com/alesha-pro/qwen38-flash-next-4x3090)）はRTX 3090 4枚向けで、
メインメモリを110〜128GB要求します。170HX 2枚・92GiBで動かすための工夫は2つです。

1. **PLEのn-gramテーブルを自前でFP8に変換する。**
   - W4A16チェックポイントは、51Bパラメータのper-layer embeddingのn-gramテーブルを
     BF16のまま（102.4GB / 95.4GiB）持っています。vLLMのPLEオフロードは、これをメインメモリに置きます。
   - `scripts/quantize_ple_fp8.py` で、テーブル全体で1つのスケール（`amax / 448`）を使う
     FP8 E4M3に書き換えます。alesha-proのオーバーレイが読める形式で、51.3GB / 47.7GiBになります。
   - 求まったスケール `0.00019931793212890625` は、RadixArkが公開しているFP8 PLEと
     **ビット単位で一致**しました。180GBを追加でダウンロードせずに、同じテーブルが作れます。
2. **64GBカードでTP2＋エキスパート並列。** エキスパート並列は必須です。MoEの中間次元640を
   2分割すると320になり、量子化のグループサイズ128で割り切れません。

## モデル：heretic-2

`run.sh` が動かすのは
[densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE](https://huggingface.co/densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE)
（revision `705a2ce`）です。拒否応答を取り除いた
[trohrbaugh/Qwen3.8-Flash-Next-heretic-2](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2)
を AutoRound で W4A16 に量子化したものです。アーキテクチャは同じなので、
`scripts/prepare-heretic2.sh` で次の書き換えをすれば、同じイメージとオーバーレイで動きます。

- **量子化設定。** イメージには `auto-round` 方式がありませんが、AutoGPTQ の経路で同じ
  パッキングを読めます。`config.json` をベースのチェックポイントのものに置き換え、GPTQ
  w4 g128 sym の設定を入れます。量子化されているのはルーティングされるエキスパートだけで
  （`qweight` 73,728個がすべて `.experts.` 配下）、`dynamic: {"-:.*mtp.*": {}}` で MTP ヘッドは
  BF16 のままにします。エキスパートの計算は Marlin です（ログに `Using MarlinExperts`）。
  このイメージは `gptq`・`gptq_marlin`・`marlin` をすべて `auto_gptq` に置き換えるので、
  `--quantization` の指定は関係ありません。
- **MTPヘッド。** チェックポイントに含まれていないので、ベースの BF16 MTP ヘッドを
  `model-mtp-graft.safetensors`（4.9GiB）として移植します。改変前のモデルに対して学習された
  ヘッドですが、検証で採用されるのは本体のトークンです。
- **トークナイザー・チャットテンプレート・生成設定。** ベースのチェックポイントのものを使います
  （書き換え後のトークナイザー一式はベースとバイト単位で一致）。heretic の
  `generation_config.json` は停止トークンが 248044 だけで、ベースは 248046 と 248044 です。
- **PLE。** ベースから作った FP8 テーブル（`build-ple-fp8.sh`）を使います。リポジトリ付属の
  FP8 PLE はダウンロードしません。

heretic-2 を以前の既定値（バッチトークン数1024、KV 1枚あたり22.7GiB）で測った結果は、
[bench-of-us の深度ラダーのレポート](https://github.com/jimoto-no-llm/bench-of-us/blob/main/report/2026-09-27_153558_measuring_qwen3.8_flash_next_w4a16_on_2x_cmp_170hx_with_vllm_tp2.md)にあります。

| 深度 | プリフィル（新規トークン/s） | デコード、合成テキスト | デコード、実運用相当の生成 |
|---|---|---|---|
| 32K | 3,323 | 186.2（MTP採択率0.94） | 97.1（0.39） |
| 128K | 2,994 | 184.6（0.95） | 89.1（0.35） |
| 258K | 2,575 | 178.3（0.93） | 69.7（0.21） |

2つのデコード列の2.6倍の差は、MTP の採択率の違いだけで説明できます。MTP 付きのモデルを
合成テキストのラダーだけで測ると、デコードを大きく見積もりすぎます。この heretic-2 の数値は、
ベースのチェックポイントを `bench/bench.py` で測った下の[結果](#結果)の表とは直接比べられません。

## ハードウェア（実測）

| | |
|---|---|
| GPU | CMP 170HX ×2：GA100、sm_80、70 SM、1枚あたり63.4GiB |
| 接続 | 1枚ずつPCIe Gen2 x16（`lspci` のLnkStaが5GT/s x16）、**P2Pなし**（`can_device_access_peer` は双方向ともfalse） |
| ホスト⇔GPU | **6.6GB/s**（ピン留めメモリ、256MBコピー）。Gen2 x16の上限の約83% |
| メインメモリ | 91.9GiB |

170HXについて「リンクがx4に絞られている」という記述もありますが、この個体は違いました。
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv` と
コピーのベンチマークで自分のカードを確認してください。x4なら転送量は1/4になります。

## 結果

**ベースの Vnimanie チェックポイント**を、当時の既定の構成で測った結果です：
`--max-model-len 262144`、MTP k=4、`--max-num-seqs 8`、`--max-num-batched-tokens 1024`、
BF16 KV、`--kv-cache-memory 24383208960`（カードを使い切る値）。
「64K」と付いた行は、その既定を変える前に `--max-model-len 65536` で測ったものです。
今の既定値は[バッチトークン数とKVサイズ](#バッチトークン数とkvサイズ)にあります。

### デコード、1本（tok/s、3回の中央値、最大512トークン）

| 構成 | 日本語の散文 | 英語の散文 | 日本語のリスト | コード |
|---|---|---|---|---|
| MTPなし | 65.8 | 65.9 | 64.9 | 66.5 |
| MTP k=2 | 105.3 | 102.2 | 101.2 | 133.6 |
| MTP k=3 | 105.9 | 104.6 | 111.1 | 156.5 |
| MTP k=4、`max-num-seqs 1`（64K） | 99.6 | 100.9 | 104.9 | 182.0 |
| MTP k=4、`max-num-seqs 8`（64K） | 104.5 | 101.1 | 114.0 | 173.0 |
| **MTP k=4、`max-num-seqs 8`、コンテキスト262K（既定）** | **101.3** | **97.8** | **107.7** | **170.0** |

MTPの効果は、散文で約1.6倍、コードで約2.6倍です。kを増やすとコード（予測しやすい
トークン）は速くなりますが、散文は変わりません。各セル内のばらつきはおおむね±5〜10%です。

### 同時リクエスト（合計tok/s、256トークンのコード回答）

| 同時数 | 1 | 4 | 8 |
|---|---|---|---|
| `max-num-seqs 1`（64K） | 127 | 126（順番待ち） | – |
| `max-num-seqs 8`（64K） | 127 | 333 | 302 |
| **`max-num-seqs 8`、コンテキスト262K（既定）** | 126 | **392** | 387 |

### プリフィル（入力tok/s、ランダムな文字列を先頭に付けてプレフィックスキャッシュを無効化）

| 入力 | 6,954トークン | 27,853トークン |
|---|---|---|
| 64K | 2,637 | 3,068 |
| コンテキスト262K（既定） | 3,191 | 3,261 |

### 長いコンテキスト

200,087トークンのチャット入力の24%の位置にランダムなコードを埋め込んだところ、正しく
答えました。最初のトークンまで70.5秒（プリフィル2,839 tok/s）、その深さでのデコードは
69.0 tok/sでした。同じ前置きの繰り返しはプレフィックスキャッシュが効くので、長い文書の
読み込みで時間がかかるのは初回だけです。

### 効果がなかったもの

| 試したこと | 結果 |
|---|---|
| `--max-num-batched-tokens 4096`、KVはカードを使い切る値（24383208960） | 7Kトークンのプリフィルで500エラー。デコードは変わらず。KVを減らせば8192でも動く（下記） |
| `VLLM_COMPILE`（inductor） | 問題なく起動する（古い資料にある「止まる」は起きない）。ただしデコードは速くならず（124 / 101 / 112 / 176）、起動が30秒遅くなる |
| `--pipeline-parallel-size 2` | 起動時に拒否：`VLLM_PLE_CPU_OFFLOAD does not support the requested configuration. Unsupported settings: PP=2`。upstream のビルドでは動く（[upstream のビルドのほうが速い](#upstream-のビルドのほうが速い)） |
| `cudagraph_mode: FULL` | vLLMが `FULL_DECODE_ONLY` に格下げする（QSAバックエンドが均一バッチにしか対応していない） |
| FP8 KV（`fp8_e4m3`、alesha-proのQSAオーバーレイ、この構成用にキャリブレーション） | 動くが、ここでは使う価値がない（下記） |

### バッチトークン数とKVサイズ

`--max-num-batched-tokens 8192`（以前の1024の8倍）にすると、プリフィルの中間データ用の
メモリが要ります。カードを使い切るKVでは、その余地がありません。しかも `--kv-cache-memory` を
明示すると、その分を確保するはずのプロファイリングが行われません。8192・heretic-2 で、
1枚あたりの値ごとに実測した結果です。

| `--kv-cache-memory` | 結果 |
|---|---|
| 21213949952（19.76GiB）、vLLM自身が「GPUメモリを使い切る値」として出す推奨値 | 8.9Kトークンのプリフィルで実行中にOOM |
| **18897856102（17.6GiB）、既定** | KV 1,151,760トークン、262Kで同時4.39本分。33Kトークン×8本の並列が通る。ピーク 64,783 / 65,536MiB で、**余裕は753MiBだけ** |
| 空（`--gpu-memory-utilization 0.92` で決める。15.5GiB） | KV 1,016,505トークン。安全側 |

17.6GiBは余裕がわずかです。OOMになったら `KV_CACHE_MEMORY= ./scripts/run.sh` で割合指定に
戻せます。`MAX_NUM_BATCHED_TOKENS=1024 KV_CACHE_MEMORY=24383208960` にすれば、結果の表を
測ったときの構成に戻ります。

### FP8 KVキャッシュ

オーバーレイ付属の収集機能（`QSA_FP8_CALIBRATE_OUT`）で、このチェックポイント・TP2・MTP k=4
の構成でキャリブレーションし、層ごとに自前とalesha-proの値の大きい方を採用しました
（`results/fp8-kv-scales-w4a16-tp2-mtp4.json`）。MTPドラフターのアテンション層はalesha-proの
ファイルに含まれていないので、自前のキャリブレーションが必要でした。

| | BF16 KV（既定） | FP8 KV |
|---|---|---|
| KVトークン数 | 1,058,505 | 1,441,792（+36%。線形アテンションの状態はBF16のまま） |
| デコード 日本語 / 英語 / リスト / コード | 104.5 / 101.1 / 114.0 / 173.0 | 95.5 / 94.9 / 101.2 / 158.0（−6〜−11%） |
| プリフィル 7K / 28K | 2,637 / 3,068 | 2,668 / 2,603 |
| ニードル、2.9万〜4.3万トークン、チャット形式、12回 | 正解11、拒否1 | 正解10、拒否2 |
| 短い貪欲生成16件、64トークン | – | 12件がBF16と同一、4件は言い回しの違いのみで破綻なし |

検索能力はBF16と同等でした（見つけられなかった回はなく、失敗はすべて「金庫の解除コード」の
開示をモデルが拒否したもの）。ただしデコードが遅くなり、増える容量も不要です。BF16でも
262Kのコンテキストを5.67本分保持できます。既定はBF16のままです（FP8の計測は
`--max-model-len 65536` で行いました）。

### 長文の出力

`bench/report-prompt-ja.txt` の5セクション構成の日本語レポートを `reasoning_effort: medium`
で依頼したところ、6,813文字・4,080トークンを44秒で出力しました。途中切れなし、全セクション
あり、日本語の破綻なし。ただし、知らない略語をそれらしく展開したり、比較表に根拠のない
数値を入れたりもしていました。固有名詞や数値は検証してください。

### メモリの内訳

| | 1枚あたり | 合計 |
|---|---|---|
| 重み＋torch以外（MTPヘッド含む） | 38.7GiB | 77.4GiB |
| KVキャッシュ（BF16）、計測時 | 22.7GiB | 45.4GiB → コンテキスト262Kで1,486,412トークン |
| KVキャッシュ（BF16）、今の既定 | 17.6GiB | 35.2GiB → 1,151,760トークン |
| メインメモリ、コンテナ全体 | – | 61.2GiB（うちPLEワーカー48.4GiB） |

起動は約**5分50秒**です（本体の重み103秒、MTPドラフター9秒、FP8 PLE約40秒、
エンジン初期化とCUDAグラフのキャプチャ122秒）。

## `reasoning_effort` は medium（か low）を必ず指定する

チャットテンプレートの既定は `xhigh` です。コーディング・推論の3つの質問を、各レベル2回ずつ、
max_tokens 20,000で測りました（各回の思考トークン数）。

| 質問 | low | medium | xhigh |
|---|---|---|---|
| CSVの統計関数 | 176 / 205 | 196 / 117 | 1,114 / **20,000（打ち切り）** |
| asyncioサーバーの遅延調査 | 238 / 307 | 440 / 335 | **16,577 / 19,451（どちらも打ち切り）** |
| アルゴリズムの証明 | 557 / 693 | 1,302 / 451 | **19,403 / 20,000（どちらも打ち切り）** |

lowとmediumはすべて回答まで書き切りました。**xhighは6回中5回、上限まで思考だけで使い切り、
回答を出しませんでした**（毎回3分ほど何も返らない状態）。
`chat_template_kwargs: {"reasoning_effort": "medium"}` を送ってください（有効な値は `xhigh`・
`medium`・`low` で、`high` はエラー）。指定しないクライアントは `xhigh` になります。

## upstream のビルドのほうが速い

2026年11月10日から、このチェックポイントを動かす方法がもう一つあります。upstream の
vLLM 公式 wheel に [`vllm-cmp170hx`](https://github.com/AlexBrandes/vllm) プラットフォーム
プラグインを足したもので、[AlexBrandes](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next)
がパッケージしています。計測したすべての項目で本リポジトリのビルドを上回ったので、ここの
レシピは推奨ではなく、参照とロールバック用として残します。

両方とも**同じ** heretic-2 W4A16 チェックポイント、同じ FP8 PLE、同じプロンプト、
`bench/bench.py` と同じデコード定義（TTFT を除外）で測っています。どちらも
`num_requests_running == 0` を待ってから計測し、事前にウォームアップを8回回しました。
生データは [`results/2026-11-10-upstream-build-comparison.jsonl`](results/2026-11-10-upstream-build-comparison.jsonl)
にあります。

| | 本リポジトリ | upstream + プラグイン |
|---|---|---|
| vLLM | `0.1.dev20073+g8e685d198`（フォーク。このコミットは upstream に存在しません） | `0.33.0rc1.dev659+ga0d7356b9`（公式 wheel） |
| 170HX 対応 | パッチ群とオーバーレイ（`alesha-pro@9b36aae` に固定） | `vllm-cmp170hx 0.2.0` プラットフォームプラグイン |
| PLE | `VLLM_PLE_CPU_OFFLOAD=1`、47GiB をホストRAMに常駐 | `--engram-config {"disk_offload":true}`、ステップごとに `preadv`、32スレッドで読み出し |

### デコード、1本（tok/s、6回の中央値）

| プロンプト | 本リポジトリ | upstream | upstream + nvfp4 ドラフトヘッド |
|---|---|---|---|
| コード | 225 | 241 | **266** |
| 英語の散文 | 82 | 79 | **95** |
| `banana` ×80 | 281 | 895 | **696**（最高 2,151） |

受理長は両ビルドで同じなので、これはドラフトが良くなったのではなくエンジン側の差です。
`banana` の行がいちばん分かりやすく、1ステップが 5.7ms から **2.3ms** に下がります。

### 同時リクエスト（合計tok/s、3回の最良値、400トークンの回答）

| 同時数 | 本リポジトリ | ストール | upstream | ストール |
|---:|---:|---:|---:|---:|
| 1 | 213.6 | 0 | **310.9** | 0 |
| 3 | 511.3 | 0 | **776.4** | 0 |
| 4 | 449.8 | **6** | **673.8** | 0 |
| 6 | 455.7 | **16** | **848.4** | 0 |
| 8 | 863.8 | **10** | **992.9** | 0 |
| 9 | 1042.3 | 0 | **1378.3** | 0 |
| 11 | 945.5 | **6** | **1106.4** | 0 |

同時4本・6本・8本・11本で落ち込んでいるのは、このビルドがストールしている箇所とぴったり
一致します。upstream のビルドではどの同時数でもストールが起きず、`ITLmax` は 23〜64ms に
収まりました。最高は**同時9本で 1,378 tok/s** です。

### `--pipeline-parallel-size 2` は向こうでは動く。それでも遅い

[効果がなかったもの](#効果がなかったもの)に書いたとおり、ここでは PP は拒否されます。
`VLLM_PLE_CPU_OFFLOAD` が `pipeline_parallel_size != 1` を受け付けないからです。upstream の
`disk_offload` にはその制約がないので、TP1 + PP2（エキスパート並列は**切る**）で普通に
起動します。VRAM は 51,152 / 59,168 MiB に分かれて層分割が効いていることが分かり、TP2 で
EP が必須になる量子化の割り切れ問題も、PP は層を分けるだけで `moe_intermediate` を割らない
ので関係なくなります。ただし TP2 + EP より 5% 遅く（コード 252.5 対 266.4、英語の散文
87.9 対 93.2）、このカードでは TP2 + EP のままが正解です。

### `tool_choice` はここでは壊れていて、向こうでは直っている

| | 本リポジトリ | upstream |
|---|---|---|
| `tool_choice: "required"`、非ストリーム、12回 | **12回違反** | **0回違反** |
| ストリーム × thinking の4通り | 1通りだけ成功 | **4通りすべて成功** |
| xgrammar の `Failed to advance FSM` と matcher 警告 | **215件** | 55リクエストで**0件** |

このビルドでは `tool_choice: "required"` が黙って `auto` に落ち、それでも `finish_reason` は
`"tool_calls"` に書き換えられて `tool_calls` は空のまま返ります。`finish_reason` を信じる
クライアントは壊れます。原因は、`VLLM_ENFORCE_STRICT_TOOL_CALLING` の既定が `True` で、
`structural_tag_model` を持つパーサ（このモデルは `qwen_3_coder` で持っています）の
`supports_required_and_named` が `False` にされ、required と named が `auto` の経路に
フォールバックすること。そして本来それを強制すべき文法が `reasoning=False` 固定で作られ、
出力の先頭が `<tool_call>` であることを要求するため、thinking するモデルでは一致しない
ことです。upstream の `ToolStrictLevel` と `constraint_start` が両方を解決しています。

### メモリの内訳（向こうのビルド）

| | 本リポジトリ | upstream |
|---|---|---|
| ホストRAM使用 | 69〜70GiB | **20GiB** |
| ホストRAM空き | 53GiB | **103GiB** |
| VRAM（1枚あたり） | 61,545MiB | 59,536MiB |

`disk_offload` はテーブルを保持せずステップごとに必要な PLE 行を `preadv` で読むので、
47GiB がホストに返ってきます。実測の読み出しは**ピークで 2.4MB/s**（コードのプロンプトは
1リクエストで 0.0〜0.4MB）で、相手は PCIe 4.0 ×4、47.7GiB のテーブルのうちページキャッシュ
常駐は5〜21%でした。スループット上の競合リスクはありませんが、ここではテーブルを専用の
NVMe に移したので、蒸留や動画生成とホストを共有できます。

ひとつ罠があります。テーブルを別のディスクに移した直後は数字が悪く見えます（コード
40〜53 tok/s）が、これは単にページキャッシュが冷えているためです。数分で常駐率が5.1%から
21.3%に上がり、デコードは 268〜287 に戻りました。**キャッシュが温まるまで測らないこと。**

### `draft_lm_head_quantization`

upstream には、ドラフタが共有する `lm_head` のコピーを量子化する設定が1行あります。
このビルドに相当するものはありません。

```
--speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_lm_head_quantization":"nvfp4"}'
```

同じビルドでオンとオフを測ると、コード 240.8 → **266.4**（+10.6%）、英語の散文
78.8 → **94.9**（+20.5%）。受理長は動かず、これは upstream が書いているとおりです
（検証はターゲットが自分のヘッドで行うため）。

### どちらのビルドでも直らないもの

1本のリクエストが、どちらのビルドでもストールします。英語の散文はほぼ毎回ストールし
（3〜14回、`ITLmax` 332〜1,090ms）、その間も1ステップの中央値は 25〜31ms で安定しています。
10個の候補を測って、すべて否定しました。電力上限、スワップ、メジャーページフォルト、
PLE のホスト側読み出し、プリエンプション、受理率のばらつき、プレフィックスキャッシュ、
CUDA グラフの不一致、TP/EP の all-to-all、ディスクI/O です。

測って分かったこと。1リクエストあたりの仕事量は同じ（約35ステップ、1ステップ約4.8トークン、
プリエンプション0）なのに所要時間は6倍ばらつく。ストール中も GPU は 86〜88% で動いている。
遅い回はCPUを2〜3倍使う。コードのプロンプトはディスクを**0.0MB**しか読まずにストールする。
そして PP2（TPなし、EPなし、集団通信なし）でも同じようにストールする。プロンプトの内容に
依存し、`banana` ×80 はほとんどストールせず、英語の散文はほぼ必ずストールします。設定では
なくカーネルかルーティング側を示していますが、原因は分かっていません。

英語の散文が遅いのには、もう一つはっきりした理由があります。受理長がコードや `banana` の
4.5〜4.8 に対して 2.45〜2.75 しかありません。ここで使っている MTP ヘッドは、このモデル自身の
**日本語**の回答で蒸留したものなので、英語の散文ではドラフトが当たりません。これはビルドを
変えても直らず、英語データを混ぜて蒸留し直す必要があります。

## 必要なもの

- **64GBのsm_80カード2枚。** 確認済みなのはCMP 170HXのみです。
- **メインメモリの空き約64GiB。** コンテナの実測は61.2GiB。この92GiBのマシンでは残り約31GiBです。
- **準備中はディスク約335GB。** ベースのチェックポイント179.8GB＋heretic-2 74GB＋
  FP8 PLE 51.3GB＋イメージ28.8GB。`prepare-heretic2.sh` と `build-ple-fp8.sh` が終われば
  ベースは削除でき、約155GBになります。
- NVIDIAランタイム付きのDocker、CUDA 13.0のイメージが動くドライバ。
- 170HXの速度がここの数字に全然届かない場合は、まずマザーボードの `PWRBRK#` パワー
  ブレーキを疑ってください。
  [deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx#troubleshooting-cards-running-4-slow-pwrbrk--edge-pin-b30)
  に診断方法があります（このマシンでは2枚とも `HW Power Brake Slowdown` が `Not Active`）。

## クイックスタート

```bash
git clone https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx
cd qwen38-flash-next-vllm-2x170hx

./scripts/setup.sh              # 固定版オーバーレイ＋vLLMイメージ（28.8GB）
./scripts/download-model.sh     # ベースのW4A16、179.8GB、帯域制限あり（RATE=95M）、再開可能
./scripts/build-ple-fp8.sh      # BF16 PLE → FP8＋マニフェスト
./scripts/download-heretic2.sh  # heretic-2のW4A16、74GB、sha256検証つき、再開可能
./scripts/prepare-heretic2.sh   # GPTQ設定、ベースのトークナイザー、MTPヘッドの移植

GPUS=1,2 ./scripts/run.sh     # 170HX 2枚をインデックスかUUIDで指定（nvidia-smi -L）
until curl -sf localhost:18024/health; do sleep 10; done
python3 bench/bench.py --reps 3 --prefill 8192,32768 --conc 1,4,8
```

エンドポイントは `http://localhost:18024/v1`、モデル名は `flash-next-w4a16` です。ツール
呼び出しパーサ `qwen3_xml` と推論パーサ `qwen3` が有効です。`gateway/litellm-config.yaml`
はLiteLLMゲートウェイを置く場合の設定例です。パスや設定はすべて環境変数で変えられます。
`scripts/env.sh` と `scripts/run.sh` の冒頭を参照してください。

## 押さえておくべき設定

| 設定 | 値 | 理由 |
|---|---|---|
| `--tensor-parallel-size 2 --enable-expert-parallel` | 必須 | 640/2 がグループサイズ128で割り切れない |
| `--speculative-config` | MTP、k=4 | コードで最速、散文は同等（上の表） |
| `--max-model-len` | 262144 | モデル本来の上限。上げても短いリクエストは遅くならなかった |
| `--max-num-seqs` | 8 | 同時4で合計約3倍、1本の速度は変わらない |
| `--max-num-batched-tokens` | 8192 | 下のKVサイズなら動く。KVを使い切る値では4096でも500エラー |
| `--compilation-config` | `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` | inductorは効果なし、FULLは格下げされる |
| `--kv-cache-memory` | 18897856102 | 8192トークンのプリフィルに耐えた最大の実測値。vLLMの推奨値ではOOM |
| `--kv-cache-dtype` | `auto`（BF16） | FP8 KVを実測：容量+36%、デコード6〜11%低下 |
| `VLLM_PLE_EMBEDDING_DTYPE` | `float8_e4m3fn` | オーバーレイのFP8 PLE経路を選ぶ |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | 75 | 既定の5秒だと、LiteLLMゲートウェイが再利用しようとした待機中の接続をvLLMが閉じてしまう（"Server disconnected"） |
| `--cap-add SYS_PTRACE --security-opt seccomp=unconfined` | 元レシピのまま | PLEオフロードが `pidfd_getfd` を使う。外した場合は未検証 |

## 計測方法

- `bench/bench.py`：ストリーミング、`temperature 0.7`、`reasoning_effort: medium`。
  デコードtok/s ＝（生成トークン数 − 1）÷（最後のトークン − 最初のトークン）で、最初の
  トークンまでの時間は含みません。生成トークン数には思考トークンも含みます。プリフィル
  tok/s ＝ 入力トークン数 ÷ 最初のトークンまでの時間で、プロンプトの先頭に毎回ランダムな
  文字列を付けてプレフィックスキャッシュを効かなくしています。
- 各構成は新しいコンテナで、ほかのリクエストを流さずに測りました。例外は `results/` の
  `E` で、デコードの計測中に別のリクエストが重なったため上の表には使っていません。既定の
  行はそのやり直し（`FINAL`）です。
- `batched-tokens 4096` は、結果を書き出す前にエラーで止まりました。
- ニードルテストはチャットテンプレート経由で行っています。同じ文書を生の `/v1/completions`
  で送ると、文書の書き出し方次第でBF16でもFP8でも1トークンで止まりました。FP8の「失敗」に
  見えた初期の結果は、キャッシュではなくこれが原因でした。

## クレジット

本リポジトリが自らの比較対象にしている upstream wheel + プラグイン版と、その土台の
`vllm-cmp170hx` プラットフォームプラグインは
[AlexBrandes](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next) によるものです。

- モデル：Qwen。W4A16量子化：[VnimanieAI](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16)
- heretic-2：[trohrbaugh](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2)。
  そのW4A16量子化：[densohax](https://huggingface.co/densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE)
- PLEオフロードのオーバーレイとマニフェストツール：
  [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090)
  （固定コミットを参照。本リポジトリには同梱していません）
- 出発点にした設定とMTPの知見：
  [loktar00/qwen38-flash-next-vllm-3090-recipe](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe)
- 再現したFP8 PLEの形式：
  [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- vLLM

## ライセンス

本リポジトリのコードはApache-2.0です。モデルの重みは含みません（重みはQwen Community
Licenseに従います）。
