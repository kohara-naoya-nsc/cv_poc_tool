var app = new Vue({
    el: "#result_summary_table",
    data: {
        avg_row_iou: result_data.result_summary_data.average.avg_row_iou,
        avg_row_true_count: result_data.result_summary_data.average.avg_row_true_count,
        avg_row_true_area: result_data.result_summary_data.average.avg_row_true_area,
        avg_row_tp: result_data.result_summary_data.average.avg_row_tp,
        avg_row_fp: result_data.result_summary_data.average.avg_row_fp,
        avg_row_fn: result_data.result_summary_data.average.avg_row_fn,
        avg_row_presision: result_data.result_summary_data.average.avg_row_presision,
        avg_row_recall: result_data.result_summary_data.average.avg_row_recall,
        class_summary_list: result_data.result_summary_data.class_summary_list
    },
    computed: {
        computed_list: function () {
            for (let i = 0; i < this.class_summary_list.length; i++) {
                // 画面項目の四捨五入
                this.class_summary_list[i]["avg_iou_pred"] = Math.round(this.class_summary_list[i]["avg_iou_pred"] * 100) / 100;
                this.class_summary_list[i]["avg_presision"] = Math.round(this.class_summary_list[i]["avg_presision"] * 100) / 100;
                this.class_summary_list[i]["avg_recall"] = Math.round(this.class_summary_list[i]["avg_recall"] * 100) / 100;
            }

            // 画面項目（平均行）の四捨五入
            this.avg_row_iou = Math.round(this.avg_row_iou * 100) / 100;
            this.avg_row_true_count = Math.round(this.avg_row_true_count * 100) / 100;
            this.avg_row_true_area = Math.round(this.avg_row_true_area * 100) / 100;
            this.avg_row_tp = Math.round(this.avg_row_tp * 100) / 100;
            this.avg_row_fp = Math.round(this.avg_row_fp * 100) / 100;
            this.avg_row_fn = Math.round(this.avg_row_fn * 100) / 100;
            this.avg_row_presision = Math.round(this.avg_row_presision * 100) / 100;
            this.avg_row_recall = Math.round(this.avg_row_recall * 100) / 100;

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
        sort: [sort_number_without_tag, true, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag, sort_number_without_tag],
    });
}

// ページ読み込み時、データテーブルを作成
make_datatable();