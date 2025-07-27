<div style="text-align: center;">

# ComfyUI Local Translator
ローカルSLM(Microsoft Phi-4)を用いたテキスト翻訳ノードです。<br/>
This is a text translation node using a local SLM (Microsoft Phi-4) for ComfyUI.

![Screenshot](./images/image000.png)

!!! ComfyUIの一定の知識を前提とし、今のところ初心者向けの案内はしていません。 !!!<br/>
!!! NOW, TARGET FOR DEVELOPERS. NOT RECOMMENDED FOR BEGINNERS. !!!

</div>

## 特徴 / Features
* ローカルSLMを用いた翻訳を行います。<br/>
  This is a translator using a local SLM.
  * Google Translatorなどを使う翻訳ノードと比べAPI回数制限などを受けません。<br/>
    There is no API limit like Google Translator.
  * 検閲を受けません。(笑)<br/>
    It is not subject to censorship.
* ローカルSLMもComfyUIのモデルマネジメント下に入ります。<br/>
  Local SLM also has VRAM control under ComfyUI's model management.
* トリガーワード指定が可能になっています。<br/>
  The keywords can be specified.

## インストール方法 / Installation
通常のcustom nodesと同様です。割愛します。<br/>
The installation process is similar to that of a typical custom nodes for ComfyUI. I will omit the explanation.

初回実行時にHugging Faceよりモデルをダウンロードします。<br/>
At first time you run this extension, it will download a model from Hugging Face.

## 使い方 / Usage
#### 基本的な使い方 / Basic Usage
上段に翻訳する文字列を、下段の文字列は原則としてそのまま出力されます。<br/>
The upper line contains the string to be translated, and the lower line is generally output as is.<br/>
下段を指定する場合、 **%TRANSLATE%** を含めて指定する必要があり、このキーワード部分に上段で指定した翻訳した後の文字列が埋め込まれて出力されます。<br/>
When specifying the lower line, you must include **%TRANSLATE%** at once and the translated string specified in the upper line will be embedded in this keyword and output.

![Screenshot](./images/image001.png)

#### 特殊ルール / Special rules
翻訳する文章には次の特殊ルールが適用されます。<br/>
The special rules are as follows:

1. _文_[_キーワード_] / _SENTENCE_[_KEYWORD_]

    前の文を角括弧内に指定したキーワードで表現するように指示します。<br/>
    Make sure the KEYWORD in the brackets expresses the previous SENTENCE.

    例 / Example:
    * Input: 大きなツインテール[pigtails hair]が彼女の特徴だ。<br/>
      Translated: Her big pigtails hair are her defining feature.

2. [_文_|_キーワード_] / [_SENTENCE_|_KEYWORD_]

    角括弧内の縦棒前の文を縦棒の後ろのキーワードで表現するように指示します。<br/>
    The SENTENCE before the vertical bars in square brackets must be expressed with the KEYWORD after the vertical bar.

    例 / Example:
    * Input: 彼は約20年にわたり[アイドルマスター|"THE IDOLM@STER"]に夢中だ。<br/>
      Translated: He has been crazy fun about "THE IDOLM@STER" for a long time about 20 years.

特定のトリガーキーワードを指定する時に便利です。<br/>
This is useful when you want to be explicit about a particular trigger keyword.
