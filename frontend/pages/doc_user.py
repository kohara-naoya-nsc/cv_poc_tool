import streamlit as st  

st.title("ユーザー登録・プロジェクト作成")
section = st.radio("セクションを選択してください", [
    "ユーザー登録",
    "プロジェクト作成",
])

if section == "ユーザー登録":
    st.header("ユーザー登録")
    st.image("assets/labelstudio_login.png")
    st.markdown('''
    新規登録（Sign up）画面より、日鉄ドメインのメールアドレスおよび任意のパスワードを用いてログインを行ってください。
    ''')

elif section == "プロジェクト作成":
    st.header("プロジェクト作成")
    st.video("assets/movie/プロジェクト作成.mp4")
    st.markdown('''
    1. 「Project Name」タブより、任意のプロジェクト名を入力してください。  
    2. 「Data Import」タブより、画像データをインポートしてください。  
       ※画像はドラッグ＆ドロップ、またはファイル選択ダイヤログからアップロードが可能です。  
    3. 「Labeling Setup」タブより、アノテーション方法の設定を行ってください。  
       ※プロジェクトで使用するモデルに応じて、アノテーション方法は異なります。それぞれのマニュアルページをご参照ください。
    ''')

# st.title("ユーザー登録・プロジェクト作成")


# section = st.radio("セクションを選択してください", [
#     "ユーザー登録",
#      "プロジェクト作成",
# ])

# if section == "ユーザー登録":
#     st.header("ユーザー登録")
#     st.image("assets/labelstudio_login.png")
#     st.markdown('''
#     Sign upで日鉄ドメインのメールアドレス及び任意のパスワードでログインしてください
#     ''')

# elif section == "プロジェクト作成":
#     st.header("プロジェクト作成")
#     st.video("assets/movie/プロジェクト作成.mp4")
#     st.markdown('''
#     1. Project Nameタブでプロジェクト名を設定してください
#     2. Data importタブで画像をインポート  \n
#     →ドラッグアンドドロップもしくはファイル選択で画像をアップロード  \n
#     3. Labeling Setupタブでアノテーション方法を設定  \n
#     →モデルごとにアノテーション方法が異なるので、それぞれのページで説明
#     ''')


