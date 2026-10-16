import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
import numpy as np
import cv2
import zipfile
import io


def sort_points(points):
    """
    4つの点を以下の順序で並び替える関数:
    1. y座標が低い2点の内、x座標が低い点
    2. y座標が低い2点の内、x座標が高い点
    3. y座標が高い2点の内、x座標が高い点
    4. y座標が高い2点の内、x座標が低い点
    Args:
        points (list of tuple): 4つの点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    Returns:
        list of tuple: 並び替えられた4つの点
    """
    # 点をy座標でソート
    points = sorted(points, key=lambda p: p[1])
    
    # y座標が低い2点と高い2点に分割
    top_points = points[:2]
    bottom_points = points[2:]
    
    # それぞれのグループをx座標でソート
    top_left = min(top_points, key=lambda p: p[0])
    top_right = max(top_points, key=lambda p: p[0])
    bottom_right = max(bottom_points, key=lambda p: p[0])
    bottom_left = min(bottom_points, key=lambda p: p[0])
    
    # 並び替えられた順序で返す
    return [top_left, top_right, bottom_right, bottom_left]


def perspective_transform(image, points, output_size=(200, 200)):
    """
    指定された4点の位置情報を基に画像を射影変換し、指定されたサイズにリサイズする関数
    Args:
        image (PIL.Image.Image): 変換対象の画像
        points (list of tuple): 変換前の4点の座標リスト [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        output_size (tuple): 出力画像のサイズ (幅, 高さ)。デフォルトは (200, 200)
    Returns:
        PIL.Image.Image: 射影変換後の画像
    """
    # 4点の座標を並び替える
    points = sort_points(points)
    # 変換前の4点
    src = np.array(points, dtype=np.float32)
    # 変換後の4点
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    # 射影変換行列を計算
    matrix = cv2.getPerspectiveTransform(src, dst)
    # 射影変換を適用
    transformed_image = cv2.warpPerspective(np.array(image), matrix, output_size)
    return Image.fromarray(transformed_image)


def tool_reshape_app():
    # タイトル
    title = '<p style="font-size: 40px; font-weight: bold;">【前処理ツール】射影変換(画角補正)</p>'
    st.markdown(title, unsafe_allow_html=True)
    st.write("アップロードした画像ファイルに対して、指定した４点で射影変換を行います。")

    
    try:
        # 画像ファイルのアップロード
        st.markdown("###### import image file(画像ファイルをアップロードしてください)")
        st.markdown('''*注：日本語を含むファイルはアップロードしないでください。
        *対応する形式：.png, .jpg, .jpeg
        ''')
        uploaded_files = st.file_uploader("画像ファイルをアップロードしてください", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    except Exception as e:
        print(e)
    
    # セッション状態の初期化
    if "points" not in st.session_state:
        st.session_state["points"] = []
    if "key_index" not in st.session_state:
        st.session_state["key_index"] = 0
    
    # キーのリスト
    keys = ["pil_1", "pil_2"]
    
    # pers_imgの初期化
    pers_img = None
    
    if uploaded_files:
        # 拡張子のマッピング作成
        format_map = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
        }

        # アップロードされた画像ファイルの名前リストを作成
        file_names = [file.name for file in uploaded_files]
    
        # ドロップダウンメニューで画像を選択
        selected_file_name = st.selectbox("表示する画像を選択してください", file_names)
    
        # 選択された画像ファイルを取得
        selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
    
        with Image.open(selected_file) as img:
            # 画像のサイズを変更
            img = img.resize((640, 480))
            # 元画像をコピー
            org_img = img.copy()
    
            # 画像描画用のオブジェクトを作成
            draw = ImageDraw.Draw(img)
    
            for point in st.session_state["points"]:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red', outline='red')
    
            # 画像上でクリックを待ち、クリックすると座標を返す
            value = streamlit_image_coordinates(img, key=keys[st.session_state["key_index"]])
    
            # クリックした点の座標を取得
            if value is not None:
                point = value["x"], value["y"]
    
                # クリックした点が5点を超えるとき リセット
                if len(st.session_state["points"]) == 5:
                    st.session_state["points"] = []
    
                # クリックしたポイントの座標を格納
                if point not in st.session_state["points"]:
                    st.session_state["points"].append(point)
                    st.rerun()  # 座標追加後に再実行
    
            # リセットボタンの追加
            if st.button("座標をリセット"):
                st.session_state["points"] = []
                st.session_state["key_index"] = (st.session_state["key_index"] + 1) % 2  # キーを切り替える
                st.rerun()  # リセット後に再実行
    
            # 出力サイズの入力（デフォルト値は200x200）
            output_width = st.number_input("出力画像の幅を入力してください", value=200)
            output_height = st.number_input("出力画像の高さを入力してください", value=200)
    
            # 射影変換を開始するボタン
            if st.button("射影変換を開始"):
                if len(st.session_state["points"]) == 4:
                    pers_img = perspective_transform(org_img, st.session_state["points"], output_size=(output_width, output_height))
    
                # 画像を表示
                if pers_img is not None:
                    st.image(pers_img)
    
            # 他の画像も射影変換するかを問うボタン
            if st.button("他の画像も同じ座標で射影変換"):
                transformed_images = []
                for file in uploaded_files:
                    with Image.open(file) as temp_img:
                        temp_img = temp_img.resize((640, 480))
                        transformed_img = perspective_transform(temp_img, st.session_state["points"], output_size=(output_width, output_height))
                        transformed_images.append(transformed_img)
    
                # ダウンロード用のZIPファイルを作成
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                    for i, img in enumerate(transformed_images):
                        img_buffer = io.BytesIO()
                        # img.save(img_buffer, format="JPEG")
                        # zip_file.writestr(f"transformed_image_{i+1}.jpg", img_buffer.getvalue())
                        img.save(img_buffer, format="JPEG")
                        zip_file.writestr(f"transformed_image_{i+1}.jpg", img_buffer.getvalue())
    
                st.download_button(
                    label="射影変換した画像をダウンロード",
                    data=zip_buffer.getvalue(),
                    file_name="transformed_images.zip",
                    mime="application/zip"
                )


