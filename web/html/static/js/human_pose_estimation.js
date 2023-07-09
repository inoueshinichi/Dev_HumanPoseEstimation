'use strict';

$(function() {

    $('#target').change(function () {
        // フォームで選択された全ファイルを取得
        var fileList = this.files;

        // 個数分の画像を表示する
        for (var i = 0, l = fileList.length; l > i; i++) {
            // Blob URLの作成
            var blobUrl = window.URL.createObjectURL(fileList[i]);

            // HTMLに書き出し (src属性にblob URLを指定)
            var view = '<a href="' + blobUrl + '" target="_blank"><img src="' + blobUrl + '"></a>';
            var figText = '入力画像'
            var caption = `<figcaption>${figText}</figcaption>`
            var frame = `<div>${view}${caption}</div>`
            var item = `<figure id="input_view">${frame}</figure>`;
            console.log('item', item);
            
            $('#inference').append(item);

        }
    });
});