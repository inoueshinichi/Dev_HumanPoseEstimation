'use strict';

const isDebug = true;
const API_VER = 'v1';

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

let cookieDict = document.cookie.split(';');//split(';')を使用しデータを1つずつに分ける

cookieDict.forEach(function (kv) {
    let kvSet = kv.split('=');
    let key = kvSet[0];
    let value = kvSet[1];
    console.log(`[Cookie] ${key}=${value}`);
})

/* Image & attribute array (shape(C,H,W), blobURL) */
let imageAttribList = [];

/* Cookie */
document.getElementById('btn_cookie_agree').addEventListener(
    'click',
    function () {
        Cookies.set('cookie-agree', 'yes', { expires: cookie_expires });
        if (isDebug) {
            console.log("Cookieを作成しました.");
        }
        document.getElementById('cookie_panel').remove()
    }
);

document.getElementById('btn_cookie_reject').addEventListener(
    'click',
    function () {
        Cookies.remove('cookie-agree');
        if (isDebug) {
            console.log("Cookieを削除または拒否しました.");
        }
        document.getElementById('cookie_panel').remove();
    }
);


/* Drawer Navigation */
document.getElementById('open_nav').addEventListener(
    'click',
    function () {
        // show
        const item = document.getElementById('nav');
        if (item.classList.contains("show")) {
            item.classList.remove("show");
        } else {
            item.classList.add("show");
        }
    }
);

document.getElementById('nav').addEventListener(
    'click',
    function () {
        // hide
        const item = document.getElementById('nav');
        if (item.classList.contains("show")) {
            item.classList.remove("show");
        }
    }
);

//===================================================
// <img>要素 -> Base64形式の文字列に変換
// img : HTMLImageElement
// mime_type : string "image/png", "image/jpeg", etc
//===================================================
function cvtImageToBase64(img, mime_type) {
    // New canvas
    let canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    // Draw image
    let ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    // To base64
    return canvas.toDataURL(mime_type);
}

//===================================================
// Base64形式の文字列 -> <img>要素に変換
// base64img : Base64形式の文字列
//===================================================
async function cvtBase64ToImage(base64img) {
    // 画像データの取得(同期処理)
    const loadImage = (src) => {
        const image = new Image();
        return new Promise((resolve, reject) => {
            image.onload = () => { // 成功
                resolve(image);
            };
            image.onerror = (e) => { // 失敗
                reject(e);
            };
            image.src = base64img; // トリガー
        });
    };

    const img = await loadImage(blobURL).catch(e => {
        console.log("[ERROR] image.onload");
        return;
    });

    return img;
}

/* Set images */
document.getElementById('input_image').addEventListener(
    'change',
    async function () {

        // 既存画像リストを削除
        let ui = document.getElementById('inference');
        while (ui.firstChild) {
            ui.removeChild(ui.firstChild);
        }
        imageAttribList.length = 0; // 配列全消し


        // フォームで選択された全ファイルを取得
        let fileList = this.files;

        console.log('fileList', fileList);
        
        // 個数分の画像を表示する
        for (var i = 0; i < fileList.length; i++) {

            // ファイル名を取得
            let fileName = fileList[i].name;

            // Blob URLの作成
            let blobURL = window.URL.createObjectURL(fileList[i]);
            // console.log(`fileList[${i}], `, fileList[i]);
            // console.log('input blobURL', blobURL);

            // 画像サイズの取得 (非同期)
            // const image = new Image(); // <img src="">
            // image.addEventListener('load', (event) => {
            //     // 非同期処理
            //     imageAttribList.push({
            //         'channel': 4, // RGBA (maybe)
            //         'height': image.naturalHeight,
            //         'width': image.naturalWidth,
            //         // 'filename': image.name, // filename
            //         'blob_url': image.src, // blobURL
            //     });

            //     console.log('imageAttribList ', imageAttribList);
            // });
            // image.src = blobURL; // イベントトリガー

            // 画像データの取得(同期処理)
            const loadImage = (src) => {
                const image = new Image();
                return new Promise((resolve, reject) => {
                    image.onload = () => { // 成功
                        resolve(image);
                    };
                    image.onerror = (e) => { // 失敗
                        reject(error);
                    };
                    image.src = src; // トリガー
                });
            };

            const image = await loadImage(blobURL).catch(e => {
                console.log("[ERROR] image.onload");
                return;
            });

            // console.log("image", image);

            imageAttribList.push({
                'channel': 3, // RGB (maybe)
                'height': image.height,
                'width': image.width,
                'filename': fileName,
                'blob_url': image.src // blobURL
            });

            
            // HTMLに書き出し (src属性にblob URLを指定)
            const linkImg = document.createElement('a');
            linkImg.setAttribute('href', `${blobURL}`);
            linkImg.setAttribute('target', '_blank');

            const img = document.createElement('img');
            img.setAttribute('src', `${blobURL}`);
            img.setAttribute('alt', `入力画像_${i}`)

            linkImg.appendChild(img);

            const figure = document.createElement('figure');
            figure.setAttribute('class', 'input_view');
            figure.appendChild(linkImg);

            const figcaption = document.createElement('figcaption');
            const caption = document.createTextNode(`入力画像_${i}`);
            figcaption.appendChild(caption);

            const inputOutlineDiv = document.createElement('div') ;
            inputOutlineDiv.setAttribute('class', 'col-sm-6');
            const inputDiv = document.createElement('div');
            inputDiv.setAttribute('class', 'input');

            const filename = document.createElement('figcaption');
            filename.setAttribute('class', 'filename');
            filename.innerHTML = `${fileName}`;
            inputDiv.appendChild(filename);
            inputDiv.appendChild(figure);
            inputDiv.appendChild(figcaption);
            inputOutlineDiv.appendChild(inputDiv);

            const blockDiv = document.createElement('div');
            blockDiv.setAttribute('class', 'row');
            blockDiv.appendChild(inputOutlineDiv);

            const listRecord = document.createElement('li');
            listRecord.setAttribute('id', `item_list_${i}`);
            listRecord.appendChild(blockDiv);

            // console.log('list', listRecord);
            document.getElementById('inference').appendChild(listRecord);
        }
    }
);

/* Inference request to web api server */
document.getElementById('inference_button').addEventListener(
    'click',
    async function () {
        const firstName = document.getElementById('input_first_name').value;
        const lastName = document.getElementById('input_last_name').value;
        const age = document.getElementById('input_age').value;

        const genderChoice = document.querySelector('input[name="gender"]:checked');
        let gender;
        if (genderChoice !== null) {
            gender = genderChoice.value;
        } else {
            gender = "";
        }

        // console.log(`first: ${firstName}, last: ${lastName}, age: ${age}, gender: ${gender}`);

        let postData = {};
        postData['id'] = 0;
        postData['host'] = window.location.hostname;
        postData['port'] = window.location.port;
        postData['first_name'] = firstName;
        postData['last_name'] = lastName;
        postData['age'] = age;
        postData['gender'] = gender; // 0: 男性, 1: 女性, 2: その他
        postData['images'] = []

        console.log('imageAttribList: ', imageAttribList);

        for (let i = 0; i < imageAttribList.length; i++) {
            const channel = imageAttribList[i].channel;
            const height = imageAttribList[i].height;
            const width = imageAttribList[i].width;
            const filename = imageAttribList[i].filename;

            const blob = await fetch(imageAttribList[i].blob_url).then(r => r.blob()); // 同期処理 (blob-url to blob)
            const filetype = blob.type;

            // console.log('out blob', blob);
            // console.log('out blob_url', imageAttribList[i].blob_url);
            
            // dataURLに変換(同期処理)
            const reader = new FileReader();
            reader.readAsDataURL(blob); // input: Blob or File
            await new Promise(resolve => reader.onload = () => resolve()); // 同期処理のため待機

            const base64DataURL = reader.result;
            const base64Str = base64DataURL.split(',')[1];

            const value = {
                'meta': {
                    'filename': filename,
                    'type': filetype, // mime_type : 'image/png', 'image/jpeg', etc
                    'shape': [channel, height, width]
                },
                'data': base64Str
            };

            postData['images'].push(value);
        }

        console.log('postData: ', postData);
        const jsonData = JSON.stringify(postData);

    
        /* バックエンドに送信 */
        const ajax = new XMLHttpRequest();

        ajax.onload = function () {        // レスポンスを受け取った時の処理（非同期）
            console.log('[Ajax Response Header]');
            console.log(this.getAllResponseHeaders());
            
            // JSON形式出ない(HTMLテキスト)場合
            // const resText = ajax.responseText;
            // if (res.length > 0) {
            //     alert(res);
            // }

            // JSON形式
            let resJson = ajax.response
            // console.log('resJSON', resJson);

            const resArray = resJson['images'];
            console.log('resArray', resArray);
            console.log('length ', resArray.length);

            for (let i = 0; i < resArray.length; i++) {
                const filename = resArray[i]['filename'];
                const type = resArray[i]['type'];
                const base64Str = resArray[i]['data'];
                const index = resArray[i]['index'];

                const img_src = `data:${type};base64,${base64Str}`;

                // HTMLに受信画像を表示させる
                const item_id = `item_list_${index}`;
                const selector = `#${item_id} .row`;
                let target = document.getElementById('inference').querySelector(selector);

                // リンク
                const linkImg = document.createElement('a');
                linkImg.setAttribute('href', ``);
                linkImg.setAttribute('target', '_blank');

                const img = document.createElement('img');
                img.setAttribute('src', `${img_src}`);
                img.setAttribute('alt', `出力画像_${i}`)

                linkImg.appendChild(img);

                const figure = document.createElement('figure');
                figure.setAttribute('class', 'output_view');
                figure.appendChild(linkImg);

                const figcapDown = document.createElement('figcaption');
                const caption = document.createTextNode(`出力画像_${i}`);
                figcapDown.appendChild(caption);

                const outputOutlineDiv = document.createElement('div');
                outputOutlineDiv.setAttribute('class', 'col-sm-6');
                const outputDiv = document.createElement('div');
                outputDiv.setAttribute('class', 'output');

                const figcapUp = document.createElement('figcaption');
                figcapUp.setAttribute('class', 'filename');
                figcapUp.innerHTML = `${filename}`;
                outputDiv.appendChild(figcapUp);
                outputDiv.appendChild(figure);
                outputDiv.appendChild(figcapDown);
                outputOutlineDiv.appendChild(outputDiv);

                target.appendChild(outputOutlineDiv);

                console.log(outputOutlineDiv);
            }
        };

        ajax.onerror = function () {       // エラーが起きた時の処理（非同期）
            alert("error!");
        };

        ajax.onreadystatechange = function () {
            if (this.readyState == 0) {
                console.log('UNSENT:初期状態');
            }
            if (this.readyState == 1) {
                console.log('OPENED:openメソッド実行');
            }
            if (this.readyState == 2) {
                console.log('HEADERS_RECEIVED:レスポンスヘッダー受信');
            }
            if (this.readyState == 3) {
                console.log('LOADING:データ受信中');
            }
            if (this.readyState == 4) {
                console.log('DONE:リクエスト完了');
            }
        };

        ajax.open('post',
                  `/api/${API_VER}/predict/`, 
                  true); // 非同期: true
        ajax.setRequestHeader('Content-Type', 'application/json');
        ajax.responseType = 'json';
        ajax.send(jsonData); // 送信
    }
);