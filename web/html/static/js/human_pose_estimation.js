'use strict';

const isDebug = true;

/* 
クッキーを動かすにはサーバー環境が必要です。
ファイルスキーム(file:///)では動きません。
*/
const cookie_expires = 1; // 1日
const agree = Cookies.get('cookie-agree');
console.log('agree ', agree);
if (agree === 'yes') {
    // alert('Cookieを受け入れました');
    console.log('Cookieを受けれました.');
} else {
    // alert('Cookieを確認できません');
    console.log('Cookieを確認できません.')
}

$(function() {

    /* Cookie */
    $('#btn_cookie_agree').click(function () {
        Cookies.set('cookie-agree', 'yes', { expires: cookie_expires });
        if (isDebug) {
            console.log("Cookieを作成しました.");
        }
        $('#cookie_panel').remove();
    });

    $('#btn_cookie_reject').click(function () {
        Cookies.remove('cookie-agree');
        if (isDebug) {
            console.log("Cookieを削除または拒否しました.");
        }
        $('#cookie_panel').remove();
    });

    /* Drawer Nav */
    // show
    $('#open_nav').click(function () {
        $('#nav').toggleClass('show');
    });
    // hidden
    $('#nav').click(function () {
        $('#nav').toggleClass('show');
    });

    /* Set images */
    $('#input_image').change(function () {

        $('#inference').children().each(function () {
            $(this).remove()
        });

        // フォームで選択された全ファイルを取得
        var fileList = this.files;

        // 個数分の画像を表示する
        for (var i = 0, len = fileList.length; len > i; i++) {

            // Blob URLの作成
            var blobUrl = window.URL.createObjectURL(fileList[i]);

            // HTMLに書き出し (src属性にblob URLを指定)
            var view = `<a href=${blobUrl} target="_blank"><img src="${blobUrl}" alt="入力画像_${i}"></a>`;
            var figure = `<figure class="input_view">${view}</figure>`;
            var caption = `<figcaption>入力画像_${i}</figcaptioin>`;
            var item1 = `<div class="col-sm-6"><div class="input">${figure}${caption}</div></div>`;
            // var item2 = `<span class="output">${figure}${caption}</span>`;
            var block = `<div class="row">${item1}</div>`
            var list = `<li id="item_list_${i}">${block}</li>`
            console.log('list', list);
            $('#inference').append(list);
        }

        // 姿勢推定結果の挿入(仮)
        // var index = 0
        // var blobUrl = window.URL.createObjectURL(fileList[index]);
        // var view = `<a href=${blobUrl} target="_blank"><img src="${blobUrl}" alt="入力画像_${index}"></a>`;
        // var figure = `<figure class="input_view">${view}</figure>`;
        // var caption = `<figcaption>姿勢推定画像_${index}</figcaptioin>`
        // var result = `<div class="col-sm-6"><div class="output">${figure}${caption}</div></div>`
        // $(`#item_list_${index}`).children('.row').append(result);
    });
});