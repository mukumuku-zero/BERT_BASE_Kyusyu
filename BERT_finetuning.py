from transformers import BertJapaneseTokenizer, BertForSequenceClassification, BertModel
from transformers.trainer_utils import set_seed
import pandas as pd
import numpy as np
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from torch.optim import AdamW
from datasets import Dataset, ClassLabel
from transformers import TrainingArguments
from transformers import Trainer

## データ読み込み ##
# Excelファイルのパス
file_path = '脳外科入院初診時記録_v2.xlsx'

# Excelファイルの全シートを読み込む
all_sheets = pd.read_excel(file_path, sheet_name=None)

df_dict = {}

# 各シート名とデータフレームを表示する
for sheet_name, df in all_sheets.items():
    print(f"シート名: {sheet_name}")
    print(f'データサイズ：{df.shape}')
    display(df.head())  # 各シートの最初の5行を表
    df_dict[sheet_name] = df


## 利用データフィルタ
use_col = ['obj(死亡)', 'obj(機能予後)', '性別', '主訴', '既往歴＞あり＞既往歴', '現病歴', '検査所見']

df_shoshin = df_dict['初診時記録']
df_shoshin1 = df_shoshin[use_col]
print(df_shoshin1.shape)


## データ加工 ##
df_shoshin1 = df_shoshin1.fillna('記載なし。')

# 改行コードを「。」に変換
df_shoshin1['既往歴＞あり＞既往歴'] = df_shoshin1['既往歴＞あり＞既往歴'].apply(lambda x: x.replace('\n', '。'))
df_shoshin1['現病歴'] = df_shoshin1['現病歴'].apply(lambda x: x.replace('\n', '。'))
df_shoshin1['検査所見'] = df_shoshin1['検査所見'].apply(lambda x: x.replace('\n', '。'))

# 上記で作成される「。。」を「。」に変換
df_shoshin1['既往歴＞あり＞既往歴'] = df_shoshin1['既往歴＞あり＞既往歴'].apply(lambda x: x.replace('。。', '。'))
df_shoshin1['現病歴'] = df_shoshin1['現病歴'].apply(lambda x: x.replace('。。', '。'))
df_shoshin1['検査所見'] = df_shoshin1['検査所見'].apply(lambda x: x.replace('。。', '。'))

# 空白を削除
df_shoshin1['既往歴＞あり＞既往歴'] = df_shoshin1['既往歴＞あり＞既往歴'].apply(lambda x: x.replace('　', ''))
df_shoshin1['現病歴'] = df_shoshin1['現病歴'].apply(lambda x: x.replace('　', ''))
df_shoshin1['検査所見'] = df_shoshin1['検査所見'].apply(lambda x: x.replace('　', ''))


# シードを固定
set_seed(42)

device = torch.device(cuda if torch.cuda.is_available() else cpu)

MODEL_NAME = 'cl-tohokubert-base-japanese-whole-word-masking'

tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
sc_model = BertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

col_list = ['text', 'label']
col_name = '現病歴'

def tokenize(row)
    return tokenizer(row['text'], max_length=512)

# 40個のデータをランダムに抽出
sample_40 = df_shoshin1.sample(n=40, random_state=42)

text_names = {col_name 'text', 'obj(機能予後)' 'label'}

# カラム名を変更
sample_40 = sample_40.rename(columns=text_names)

# 最初の30個をtrain_dataに、残りの10個をtest_dataに割り当て
train_data = sample_40.iloc[30][col_list]
test_data = sample_40.iloc[30][col_list]

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# バッチサイズ指定
train_batch_size = len(train_data)
test_batch_size = len(test_data)

# データの前処理
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=train_batch_size)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=test_batch_size)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 追加学習における評価関数
def compute_metrics(result)
    labels = result.label_ids
    preds = result.predictions.argmax(-1)
    roc_auc = roc_auc_score(labels, preds)
    return {'ROC_AUC' roc_auc,}

# 訓練
training_args = TrainingArguments(
    output_dir='results',  # 結果の保存フォルダ
    logging_dir='logs' # 途中経過のログを格納するフォルダ
    per_device_train_batch_size=8,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=8,  # 評価時のバッチサイズ
    learning_rate=2e-5,  # 学習率
    lr_scheduler_type='linear',  # 学習率スケジューラの種類
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=5,  # エポック数
    save_strategy='epoch',  # チェックポイントの保存タイミング
    logging_strategy='epoch',  # ロギングのタイミング
    evaluation_strategy='epoch',  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model='f1',  # 最良のモデルを決定する評価指標
)

# trainerの設定
trainer = Trainer(
    model=sc_model, # モデル
    train_dataset=train_dataset, # 学習用データセット
    eval_dataset=test_dataset, # 評価用データセット
    args=training_args,
    compute_metrics=compute_metrics,
)

# 訓練開始
trainer.train()

# モデルの評価
trainer.evaluate()

## 学習曲線の取得 ##
import matplotlib.pyplot as plt

# 学習曲線の保存

# ログファイルから学習曲線のデータを取得
logs = trainer.state.log_history

# 学習と検証の損失をプロット
train_loss = [log['loss'] for log in logs if 'loss' in log]
eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]

plt.plot(train_loss, label='Train Loss')
plt.plot(eval_loss, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Evaluation Loss')

# 図を表示
plt.show()

# 図を保存
plt.savefig('training_curve.png')

 
## 新しいデータへの推論
new_data = df_shoshin1.sample(n=10, random_state=42)
new_data = new_data.rename(columns=text_names)
# new_dataset = Dataset.from_pandas(new_data)

# # バッチサイズ指定
# new_batch_size = len(new_data)

# # データの前処理
# new_dataset = new_dataset.map(tokenize, batched=True, batch_size=new_batch_size)
# new_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# # DataLoaderの作成
# new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=new_batch_size)

new_data_list = new_data[col_name].to_list()
new_inputs = tokenizer(new_data_list, return_tensors='pt', max_length=512)

# モデルの推論モードを有効にする
sc_model.eval()

# アテンションウェイトの取得
with torch.no_grad():
    outputs = sc_model(**new_inputs, output_attentions=True)

# アテンションウェイトを表示
attention_weights = outputs.attentions
print(attention_weights)

# 結果を取得
predicted_labels = torch.argmax(outputs.logits, dim=-1)

print(predicted_labels)

## 推論時の最終層ベクトル取得 ##
# 最終層のベクトルを取得
with torch.no_grad():
    outputs = model(**new_inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # 全ての隠れ層の状態を取得
    final_layer_vectors = hidden_states[-1]  # 最終層のベクトルを取得

# 最終層のベクトルを表示
print(final_layer_vectors)


## SHAPによる可視化
import shap

# # SHAPのExplainerを作成
# class ShapModelWrapper:
#     def __init__(self, model):
#         self.model = model

#     def __call__(self, inputs):
#         inputs = {k: torch.tensor(v).to(self.model.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.logits.cpu().numpy()

# explainer = shap.Explainer(ShapModelWrapper(sc_model), new_inputs)

# # SHAP値を計算
# shap_values = explainer(new_inputs)

# # SHAP値を表示
# shap.summary_plot(shap_values, new_inputs['input_ids'])

explainer = shap.Explainer(predicted_labels, output_names=list(label_map.keys()))

input_text = [text[:512] for text in new_data['text'].tolist()]
shap_values = explainer(input_text)
shap.plots.text(shap_values)