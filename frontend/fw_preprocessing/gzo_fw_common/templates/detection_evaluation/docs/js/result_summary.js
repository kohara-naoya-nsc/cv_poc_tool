var app = new Vue({
    el: "#result_summary_table",
    data: result_data.result_summary_data,
    computed: {
        computed_list: function () {
            for (let i = 0; i < this.class_summary_list.length; i++) {
                // インデックス番号を作成
                this.class_summary_list[i]["index"] = String(i + 1)
            }
            return this.class_summary_list
        }
    }
});

// データテーブル関連
// 項目数/ページ初期化
let pageSize = 1000000;

// データテーブルを作成する関数
function make_datatable() {
    $('#result_summary_table table').datatable({
        pageSize: pageSize,
        sort: [sort_number_without_tag, true, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag],
    });
}

// ページ読み込み時、データテーブルを作成
make_datatable();