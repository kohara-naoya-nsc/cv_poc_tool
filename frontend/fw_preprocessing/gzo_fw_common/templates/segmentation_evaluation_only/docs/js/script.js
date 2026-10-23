// 文字列にhtmlタグが含まれる場合、最初と最後のhtmlタグを削除する
function remove_outermost_html_tag(str) {
    str = str.replace(/^[^<]*(<([^<>]+)>)/i, function (match) {
        return match.replace(/(<([^<>]+)>)/i, "");
    }).replace(/(<([^<>]+)>)[^>]*$/i, function (match) {
        return match.replace(/(<([^<>]+)>)/i, "");
    });
    return str
}

// datatable.jsのソートに指定する関数
// 文字列から最初と最後のhtmlタグを除去したうえでNumber型に変換し、比較を行う。
function sort_number_without_tag(l, r) {
    l = Number(remove_outermost_html_tag(l));
    r = Number(remove_outermost_html_tag(r));
    return (l < r) ? -1 : ((l > r) ? 1 : 0);
}

// datatable.jsのソートに指定する関数
// 文字列から最初と最後のhtmlタグを除去したうえで、比較を行う。
function sort_strings_without_tag(l, r) {
    l = remove_outermost_html_tag(l);
    r = remove_outermost_html_tag(r);
    return (l < r) ? -1 : ((l > r) ? 1 : 0);
}