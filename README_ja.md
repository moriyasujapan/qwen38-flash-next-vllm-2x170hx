# CMP 170HX ×2 で Qwen3.8-Flash-Next W4A16 を動かす

[English](README.md)

マイニング用カード **NVIDIA CMP 170HX** 2枚で、**Qwen3.8-Flash-Next**（W4A16）を vLLM
で動かす構成です。カードは GA100（sm_80）、VRAMを64GBにアンロック済み、PCIe Gen2 x16、
P2Pなし。ホストのメインメモリは **91GBしかありません**。

**デコード 約170〜190 tok/s（コードなら最大277）・KV 1,058,505トークン・コンテキスト65K・
ツール呼び出し対応。** 同じモデルのGGUF版をllama.cppで170HX 1枚で動かした場合（49 tok/s）の
約3.5倍です。

---

## ここが違う

公開されているFlash-Next W4A16のレシピ（[loktar00](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe)、
[alesha-pro](https://github.com/alesha-pro/qwen38-flash-next-4x3090)）はRTX 3090 4枚向けで、
メインメモリを110〜128GB要求します。170HX 2枚・91GBで動かすために、2つの工夫をしています。

1. **PLEのn-gramテーブルを自前でFP8に変換する。**
   - W4A16チェックポイントは、51Bパラメータのper-layer embeddingテーブルをBF16のまま
     （約96GiB）持っています。vLLMのPLEオフロードは、これをメインメモリに置きます。
   - `scripts/quantize_ple_fp8.py` で、これをテーブル全体で1つのスケール（`amax / 448`）
     を使うFP8 E4M3に書き換えます。alesha-proのオーバーレイが読める形式で、サイズは
     約48GiBと半分になります。
   - 求まったスケール `0.00019931793212890625` は、RadixArkが公開しているFP8 PLEと
     **ビット単位で一致**しました。180GBを追加でダウンロードせずに同じテーブルが作れます。
2. **64GBカードでTP2＋エキスパート並列。**
   - 170HX 2枚なら、GPUに載る約77GiBの重みと、100万トークン超のKVキャッシュが入ります。
   - エキスパート並列は必須です。MoEの中間次元640を2分割すると320になり、量子化の
     グループサイズ128で割り切れません。

MTP投機的デコーディング（チェックポイントにドラフトヘッドがBF16で入っています）は
既定で有効で、約+60%の効果があります。

## 結果

3つのプロンプト（日本語の散文、Pythonコード、長めのリスト）をストリーミングで実行、
`temperature 0.7`、`reasoning_effort: medium`、各1回。実行ごとに±15%程度ぶれます。

| 構成 | 散文 | コード | 長文 | 中央値 |
|---|---|---|---|---|
| llama.cpp、GGUF Q4_K_M 4.27bpw、170HX 1枚 | – | – | – | 49 |
| vLLM W4A16、MTPなし | 145.3 | 92.1 | 108.5 | 108.5 |
| vLLM W4A16、MTP k=4 | 173.3 | 276.6 | 149.6 | 173.3 |
| vLLM W4A16、MTP k=4、KV最大（**既定**） | 188.3 | 253.0 | 143.0 | **188.3** |

長文：`bench/report-prompt-ja.txt` の5セクション構成の日本語レポートを依頼したところ、
6,813文字・4,080トークンを44秒で出力しました（102 tok/s、最初のトークンまで4.1秒）。
途中切れなし、全セクションあり、日本語の破綻なし。

`--max-num-batched-tokens` を1024から2048に上げても（MTP使用時にvLLMが警告を出します）
164対173で誤差の範囲だったので、1024のままです。

### メモリの内訳

| | 1枚あたり | 合計 |
|---|---|---|
| 重み＋torch以外 | 38.7GiB | 77.4GiB |
| KVキャッシュ（BF16） | 22.7GiB | 45.4GiB → 1,058,505トークン |
| メインメモリ（PLEオフロード） | – | 49.2GiB |

起動には約5分かかります（重み約100秒、FP8 PLE約15秒、エンジン初期化とCUDAグラフ
キャプチャ約120秒）。

このカードのリンクはGen2 x16でネゴシエートされ、ホスト⇔GPU間の転送は実測 **6.6GB/s**
（ピン留めメモリ、256MBコピー）でした。Gen2 x16の上限の約83%です。170HXについては
「リンクがx4に絞られている」という記述も見かけますが、この個体は違いました。
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv`
とコピーのベンチマークで自分のカードを確認してください。x4なら転送はこの1/4になります。

## 必要なもの

- **64GBのsm_80カード2枚。** CMP 170HXで確認済み。A100 80GBの2枚でも動くはずです。
- **メインメモリの空き約64GB。** PLEオフロード（49GiB）＋余裕。
- **ディスク約270GB。** チェックポイント180GB＋FP8 PLE 51GB＋イメージ29GB。
- NVIDIAランタイム付きのDocker、CUDA 13.0が使えるドライバ。
- 170HXの速度がここの数字に全然届かない場合は、まずマザーボードの `PWRBRK#`
  パワーブレーキを疑ってください。
  [deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx#troubleshooting-cards-running-4-slow-pwrbrk--edge-pin-b30)
  に診断方法があります。

## クイックスタート

```bash
git clone https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx
cd qwen38-flash-next-vllm-2x170hx

./scripts/setup.sh            # 固定版オーバーレイ＋vLLMイメージ（約29GB）
./scripts/download-model.sh   # 約180GB、帯域制限あり（RATE=95M）、再開可能
./scripts/build-ple-fp8.sh    # BF16 PLE → FP8＋マニフェスト、数分

GPUS=1,2 ./scripts/run.sh     # 170HX 2枚をインデックスかUUIDで指定（nvidia-smi -L）
until curl -sf localhost:18024/health; do sleep 10; done
python3 bench/bench.py
```

エンドポイントは `http://localhost:18024/v1`、モデル名は `flash-next-w4a16` です。
ツール呼び出しパーサ `qwen3_xml` と推論パーサ `qwen3` が有効です。
`gateway/litellm-config.yaml` はLiteLLMゲートウェイを置く場合の設定例です。

パスや設定はすべて環境変数で変えられます。`scripts/env.sh` と `scripts/run.sh` の冒頭を
参照してください。

## 押さえておくべき設定

| 設定 | 値 | 理由 |
|---|---|---|
| `--tensor-parallel-size 2 --enable-expert-parallel` | 必須 | 640/2 がグループサイズ128で割り切れない |
| `--compilation-config` | `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` | このモデルではAmpereでinductorが止まる。FULLはQSAバックエンドが非対応 |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":4}` | +60%。`SPEC=` で無効化 |
| `--kv-cache-memory` | 24383208960 | 64GBカードを使い切る値。割合指定だと約4.5GiB余る |
| `--kv-cache-dtype` | `auto`（BF16） | FP8 KVにはこの構成用のキャリブレーションが必要 |
| `VLLM_PLE_EMBEDDING_DTYPE` | `float8_e4m3fn` | オーバーレイのFP8 PLE経路を選ぶ |
| `--cap-add SYS_PTRACE --security-opt seccomp=unconfined` | 必須 | PLEオフロードが `pidfd_getfd` を使う |

## 既知の制限

- **長文を出させるときは `chat_template_kwargs: {"reasoning_effort": "medium"}` を
  付けてください。** テンプレートの既定は `xhigh` で、推論に数万トークン使って途中で
  切れることがあります。
- **パイプライン並列は使えません。** PLEオフロードがPPを拒否するので、PCIeのみのカードでも
  TP（PyNCCLのall-reduce）になります。P2Pがないのでカスタムall-reduceも無効です。
- **FP8 KVは未検証です。** alesha-proのFP8 QSAスケールは彼らのチェックポイント用で、
  この構成での妥当性は確認していません。
- **品質（長文テスト1回での所感）：** 日本語は自然で構成もよく、書式の指示もすべて
  守りました。ただし、知らない略語をそれらしく展開したり、比較表に根拠のない数値を
  入れたりしていました。固有名詞や数値は検証してください。
- 2枚のカードは使い切るので、他のモデルは同時に載りません。

## クレジット

- モデル：Qwen。W4A16量子化：[VnimanieAI](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16)
- PLEオフロードのオーバーレイとマニフェストツール：
  [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090)
  （固定コミットを参照。本リポジトリには同梱していません）
- 計測済みの設定とMTPの知見：
  [loktar00/qwen38-flash-next-vllm-3090-recipe](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe)
- 再現したFP8 PLEの形式：
  [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- vLLM

## ライセンス

本リポジトリのコードはApache-2.0です。モデルの重みは含みません（重みはQwen Community
Licenseに従います）。
