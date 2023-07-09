'use strict';

/* 
クッキーを動かすにはサーバー環境が必要です。
ファイルスキーム(file:///)では動きません。
*/
const agree = Cookies.get('cookie-agree');
console.log('agree ', agree);
if (agree === 'yes') {
    alert('Cookieを受け入れました');
} else {
    alert('Cookieを確認できません');
}

$(function() {

    // drawer show
    $('#open_nav').click(function () {
        $('#nav').toggleClass('show');
    });

    // drawer hidden
    $('#nav').click(function () {
        $('#nav').toggleClass('show');
    });

    $('#input_image').change(function () {

        $('#inference').children().each(function () {
            $(this).remove()
        })

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
            var frame = `<div class="input_block">${view}${caption}</div>`
            var item = `<figure id="input_view">${frame}</figure>`;
            console.log('item', item);
            
            $('#inference').append(item).append(item);

        }
    });
});