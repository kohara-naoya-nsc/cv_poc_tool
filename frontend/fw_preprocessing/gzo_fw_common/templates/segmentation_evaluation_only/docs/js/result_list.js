let dataTableSetting = {
    "paginate": {
        "first": "<<",
        "last": ">>",
        "next": ">",
        "previous": "<"
    }
}
var app = new Vue({
    el: "#result_list_table",
    data: {
        evaluation_score_thresh: result_data.index_data.segmentation_iou_thresh,
        item_list: result_data.result_list_data
    },
    computed: {
        computed_list: function () {
            for (let i = 0; i < this.item_list.length; i++) {

                // 画面項目の四捨五入（小数2桁表示）
                this.item_list[i]["segmentation_iou"] = Math.round(this.item_list[i]["segmentation_iou"] * 100) / 100;
                this.item_list[i]["presision"] = Math.round(this.item_list[i]["presision"] * 100) / 100;
                this.item_list[i]["recall"] = Math.round(this.item_list[i]["recall"] * 100) / 100;

                // インデックス番号を作成
                this.item_list[i]["index"] = String(i + 1)

                // パーセントエンコーディング
                this.item_list[i]["segmentation_image_path"] = this.item_list[i]["segmentation_image_path"].replace(/[^\/]+$/, match => encodeURIComponent(match))

                // // セル結合のために同一の値を設定
                NUM_ROWSPAN = result_data.index_data.class_count
                this.item_list[i]["segmentation_image_path"]
                    = this.item_list[i - (i % NUM_ROWSPAN)]["segmentation_image_path"];

                // 閾値による文字色変更
                if (result_data.index_data.segmentation_iou_thresh <= this.item_list[i]["segmentation_iou"]) {
                    this.item_list[i]["object_class_iou"] = ["text_black"];
                } else {
                    // IoUが閾値未満の場合に文字色を赤色にする
                    this.item_list[i]["object_class_iou"] = ["text_red"];
                };
                this.item_list[i]["class_color_square"] = "■"
                this.item_list[i]["object_class_color"] = this.item_list[i]["class_color_code"]
            }
            return this.item_list
        }
    }
});

// データテーブル関連
// 項目数/ページ初期化
let pageSize = 20;

// データテーブルを作成する関数
function make_datatable() {
    return $('#result_list_table table').DataTable({
        destroy: true,
        ordering: false,
        language: dataTableSetting,
        info: false,
        searching: true,
        pageSize: pageSize,
        pagingType: 'full_numbers',
        lengthChange: false,
        displayLength: pageSize,
        pagingDivSelector: "#paging_result_list_table",
        // 結合するカラムを指定
        rowsGroup: [1, 2],
        initComplete: function () {
            this.api()
                .column(3)
                .every(function () {
                    let column = this;

                    // 推論クラス名フィルターの作成
                    let select = document.createElement('select');
                    select.add(new Option(''));
                    let yesSelect = document.getElementById("select_class_list");
                    yesSelect.appendChild(document.createTextNode("推論クラス名："));
                    yesSelect.appendChild(select);

                    // Apply listener for user change in value
                    select.addEventListener('change', function () {
                        var val = select.value;

                        column
                            .search(val)
                            .draw();
                    });

                    // Add list of options
                    this.column(4)
                        .data()
                        .unique()
                        .sort()
                        .each(function (d, j) {
                            console.log(typeof d)
                            select.add(new Option(d));
                        });
                });
        }
    });
}

// 項目数/ページ更新時、データテーブルを再作成
$(function () {
    $('.button1').click(function () {
        inputPageSize = $('#page-size-input').val();
        if (!(inputPageSize.match(/^[1-9][0-9]*$/))) {
            // 型チェック
            window.alert("「項目数/ページ」には1以上の整数を指定してください。");
        } else if (inputPageSize > Number.MAX_SAFE_INTEGER) {
            // 最大値チェック
            window.alert("「項目数/ページ」には" + Number.MAX_SAFE_INTEGER + "以下の整数を指定してください。");
        } else {
            // データテーブル更新処理
            pageSize = inputPageSize;
            table_data.page.len(pageSize).draw();
        }
    });
});

// ページ読み込み時、データテーブルを作成
var table_data = make_datatable();

// 検索ボックスの非表示化
document.getElementById("DataTables_Table_0_filter").style.display = 'none';

