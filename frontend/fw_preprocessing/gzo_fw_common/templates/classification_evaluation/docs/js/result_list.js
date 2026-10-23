var app = new Vue({
    el: "#result_list_table",
    data: {
        evaluation_score_thresh: result_data.index_data.evaluation_score_thresh,
        item_list: result_data.result_list_data
    },
    computed: {
        computed_list: function () {
            for (let i = 0; i < this.item_list.length; i++) {

                // インデックス番号を作成
                this.item_list[i]["index"] = String(i + 1)

                // パーセントエンコーディング
                this.item_list[i]["classificated_image_path"] 
                    = this.item_list[i]["classificated_image_path"].replace(/[^\/]+$/, match => encodeURIComponent(match))
                
                // 推論結果による場合分け
                if (this.item_list[i]["true_class"] == this.item_list[i]["classification_class"]) {
                    if (Number(this.item_list[i]["classification_score"]) >= this.evaluation_score_thresh) {
                        // 分類が正しく、スコアが閾値以上の場合
                        this.item_list[i]["class_coincidence"] = "○";
                        this.item_list[i]["object_class"] = ["text_black"];
                    } else {
                        // 分類が正しく、スコアが閾値未満の場合
                        this.item_list[i]["class_coincidence"] = "○";
                        this.item_list[i]["object_class"] = ["text_green"];
                    }
                } else {
                    // 誤分類の場合
                    this.item_list[i]["class_coincidence"] = "×";
                    this.item_list[i]["object_class"] = ["text_red"];
                };
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
    $('#result_list_table table').datatable({
        pageSize: pageSize,
        pagingNumberOfPages: 100,
        sort: [sort_number_without_tag, false, sort_strings_without_tag, sort_strings_without_tag, sort_strings_without_tag, sort_number_without_tag, sort_strings_without_tag],
        filters: [false, false, 'select', 'select', 'select', false, true],
        filterText: 'フィルター',
        pagingDivSelector: "#paging_result_list_table"
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
            make_datatable();
        }
    });
});

// ページ読み込み時、データテーブルを作成
make_datatable();