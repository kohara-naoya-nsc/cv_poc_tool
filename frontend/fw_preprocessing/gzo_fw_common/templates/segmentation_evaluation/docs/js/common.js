var translation =
{
    "prediction_id": "推論ID",
    "dataset_id": "データセットID",
    "model_repository_id": "モデルリポジトリ",
    "start_datetime": "ジョブ開始日時",
    "end_datetime": "ジョブ終了日時",
    "evaluation_score_thresh": "スコア閾値",
    "detection_true_iou_thresh": "検出正解IoU閾値",
    "classification_true_score_thresh": "分類正解スコア閾値",
}

var glossary =
{
    "detection_evaluation": [
        {
            "word": "正解クラス",
            "brief_description": "正しい分類(アノテーションで指定した正解のクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論クラス",
            "brief_description": "モデルが推論した分類(モデルが判定したクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論スコア",
            "brief_description": "推論の確信の強さ",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        },
        {
            "word": "平均スコア",
            "brief_description": "モデルが推論にどの程度の確証を持っているかを表す値",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        },
        {
            "word": "検出成功率",
            "brief_description": "モデルによる検出(物体があるかどうかの判断)の成功率",
            "how_to_use": "検出に成功したか(=検出の見逃しが許されないケース)を重視する場合に重要となる指標。高いほど検出に成功していることを表す。",
            "range": "0～1"
        },
        {
            "word": "分類成功率",
            "brief_description": "モデルによる分類(物体が何かの判断)の成功率",
            "how_to_use": "分類が成功したかを重視する場合に重要となる指標。高いほど分類に成功していることを表す。",
            "range": "0～1"
        },
        {
            "word": "TP",
            "brief_description": "検出に成功した物体の中で分類結果も正しい物体の数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "FP",
            "brief_description": "検出に成功した物体の中で分類結果を誤った物体の数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "FN",
            "brief_description": "実際に存在する物体の中でモデルが検出および分類ができなかった物体の数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "適合率",
            "brief_description": "検出した物体を、正しく分類できている確率",
            "how_to_use": "誤分類の少なさを重視する場合に重要となる指標。高いほど誤分類が少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "再現率",
            "brief_description": "実際に存在する物体を、正しく検出・分類できた確率",
            "how_to_use": "検出漏れの少なさを重視する場合に重要となる指標。高いほど検出漏れが少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "正答率",
            "brief_description": "「物体が対象のクラスである/物体が対象のクラスではない/対象のクラスの物体が存在しない」の推論が正しい確率",
            "how_to_use": "誤分類と検出漏れの少なさを重視したい場合に重要となる指標。高いほど、誤分類と検出漏れの合計が少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "F値",
            "brief_description": "適合率と再現率が、どの程度両立できているかを表す指標",
            "how_to_use": "誤分類と検出漏れの少なさを両立したい場合に重要となる指標。高いほど誤分類と検出漏れの少なさが両立できていることを表す。",
            "range": "0～1"
        },
        {
            "word": "IoU平均",
            "brief_description": "物体の位置と範囲を、どの程度正しく推論できているかを表す指標",
            "how_to_use": "検出範囲の誤差の小ささを重視する場合に重要となる指標。高いほど検出範囲の誤差が小さいことを表す。",
            "range": "0～1"
        },
        {
            "word": "AP",
            "brief_description": "Detectionの総合的な精度の評価値。",
            "how_to_use": "誤検出と検出漏れの少なさを両立したい場合に重要となる指標。高いほど誤検出と検出漏れの少なさが両立できていることを表す。",
            "range": "0～1"
        }
    ],
    "detection_prediction_only": [
        {
            "word": "推論クラス",
            "brief_description": "モデルが推論した分類(モデルが判定したクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論スコア",
            "brief_description": "推論の確信の強さ",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        },
        {
            "word": "平均スコア",
            "brief_description": "モデルが推論にどの程度の確証を持っているかを表す値",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        }
    ],
    "classification_evaluation": [
        {
            "word": "正解クラス",
            "brief_description": "正しい分類(アノテーションで指定した正解のクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論クラス",
            "brief_description": "モデルが推論した分類(モデルが判定したクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論スコア",
            "brief_description": "推論の確信の強さ",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        },
        {
            "word": "平均スコア",
            "brief_description": "モデルが推論にどの程度の確証を持っているかを表す値",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        },
        {
            "word": "TP",
            "brief_description": "対象のクラスであると推論されたクラスの中で、実際に対象のクラスが正解である画像の数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "TN",
            "brief_description": "対象のクラスではないと推論されたクラスの中で、実際に対象のクラスが正解ではない画像の数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "FP",
            "brief_description": "対象のクラスであると推論されたクラスの中で、実際には対象のクラスが正解ではない画像の数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "FN",
            "brief_description": "対象のクラスではないと推論されたクラスの中で、実際には対象のクラスが正解である画像の数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "適合率",
            "brief_description": "分類した画像を、正しく推論できている確率",
            "how_to_use": "誤分類の少なさを重視する場合に重要となる指標。高いほど誤分類が少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "再現率",
            "brief_description": "正解を正しく推論できた確率",
            "how_to_use": "分類漏れの少なさを重視する場合に重要となる指標。高いほど分類漏れが少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "正答率",
            "brief_description": "「対象のクラスであるか、それ以外か」の推論が正しい確率",
            "how_to_use": "誤分類と分類漏れの少なさを重視したい場合に重要となる指標。高いほど、誤分類と分類漏れの合計が少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "特異度",
            "brief_description": "対象のクラスではない、という推論が正しい確率",
            "how_to_use": "モデルによる否定の判断の正しさを重視する場合に重要となる指標。高いほど、対象のクラスではない、という判断の正確さが高いことを表す。",
            "range": "0～1"
        },
        {
            "word": "F値",
            "brief_description": "適合率と再現率が、どの程度両立できているかを表す指標",
            "how_to_use": "誤分類と分類漏れの少なさを両立したい場合に重要となる指標。高いほど誤分類と分類漏れの少なさが両立できていることを表す。",
            "range": "0～1"
        }
    ],
    "classification_prediction_only": [
        {
            "word": "推論クラス",
            "brief_description": "モデルが推論した分類(モデルが判定したクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論スコア",
            "brief_description": "推論の確信の強さ",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        },
        {
            "word": "平均スコア",
            "brief_description": "モデルが推論にどの程度の確証を持っているかを表す値",
            "how_to_use": "モデルが推論にどのぐらい自信を持っているかを表す指標。1に近いほどモデルが確信を持って判断したことを表す。",
            "range": "0～1"
        }
    ],
    "segmentation_evaluation": [
        {
            "word": "正解クラス",
            "brief_description": "正しい分類(アノテーションで指定した正解のクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論結果",
            "brief_description": "入力画像の推論結果",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "推論クラス",
            "brief_description": "モデルが推論した分類(モデルが判定したクラス)",
            "how_to_use": "-",
            "range": "-"
        },
        {
            "word": "TP",
            "brief_description": "物体が存在するピクセルの内、正しくモデルが予測したピクセル数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "FP",
            "brief_description": "物体が存在しないピクセルの内、誤ってモデルが予測したピクセル数（過検知したピクセル数）",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "FN",
            "brief_description": "物体が存在するピクセルの内、正しくモデルが予測しなかったピクセル数（未検知のピクセル数）",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "適合率",
            "brief_description": "モデルが存在すると予測したピクセルの内、実際に物体が存在するピクセルの割合",
            "how_to_use": "誤分類の少なさを重視する場合に重要となる指標。高いほど誤分類が少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "再現率",
            "brief_description": "物体が存在するピクセルの内、正しくモデルが予測したピクセルの割合",
            "how_to_use": "分類漏れの少なさを重視する場合に重要となる指標。高いほど分類漏れが少ないことを表す。",
            "range": "0～1"
        },
        {
            "word": "IoU",
            "brief_description": "物体が存在するピクセル(𝑇𝑃+𝐹𝑁)と過検知したピクセルの合計の内、正しくモデルが予測したピクセルの割合",
            "how_to_use": "モデルがどの程度正しく物体を検出できるかを評価する指標。高いほど予測が正しいことを表す。",
            "range": "0～1"
        },
        {
            "word": "アノテーション画像枚数",
            "brief_description": "入力した画像に写っていた枚数",
            "how_to_use": "-",
            "range": "0～"
        },
        {
            "word": "アノテーション面積",
            "brief_description": "物体が存在するピクセル数（TP＋FN）",
            "how_to_use": "-",
            "range": "0～"
        }
    ]
}