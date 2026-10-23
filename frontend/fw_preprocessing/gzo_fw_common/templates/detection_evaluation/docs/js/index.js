var app = new Vue({
    el: "#information_table",
    data: {
        information: result_data.index_data,
        translation: translation,
        glossary: glossary.detection_evaluation
    }
});