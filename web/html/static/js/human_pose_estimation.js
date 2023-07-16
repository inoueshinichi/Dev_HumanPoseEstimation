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


/* Set images */
document.getElementById('input_image').addEventListener(
    'change',
    async function () {

        // 既存画像リストを削除
        let item = document.getElementById('inference').firstChild;
        while (item.firstChild) {
            item.removeChild(item.firstChild);
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
            console.log('input blobURL', blobURL);

            // 画像サイズの取得
            const image = new Image();
            image.addEventListener('load', (event) => {
                // 非同期処理
                imageAttribList.push({
                    'channel': 4, // RGBA (maybe)
                    'height': image.naturalHeight,
                    'width': image.naturalWidth,
                    'type': image.type,
                    'blob_url': image.src, // blobURL
                    'file': fileList[i]
                });

                console.log('imageAttribList ', imageAttribList);
            });
            image.src = blobURL; // イベントトリガー
            // await new Promise(resolve => image.load = () => resolve()); // 同期処理のため待機
            // console.log("promise result", image.result);

            // imageAttribList.push({
            //     'channel': 4, // RGBA (maybe)
            //     'height': image.naturalHeight,
            //     'width': image.naturalWidth,
            //     'blob_url': image.src, // blobURL
            //     'file': fileList[i]
            // });

            
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

            console.log('list', listRecord);
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

        console.log(`first: ${firstName}, last: ${lastName}, age: ${age}, gender: ${gender}`);

        let postData = {};
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
            const blobURL = imageAttribList[i].blob;
            const filename = blobURL.name;
            const filetype = blobURL.type;

            console.log('output blobURL', blobURL)
            
            // dataURLに変換(同期処理)
            const reader = new FileReader();
            reader.readAsDataURL(blobURL);
            await new Promise(resolve => reader.onload = () => resolve()); // 同期処理のため待機

            const base64DataURL = reader.result;
            const base64Str = base64DataURL.split(',')[1];

            const value = {
                'meta': {
                    'filename': filename,
                    'type': filetype,
                    'shape': [channel, height, width]
                },
                'data': base64Str
            };

            postData['images'].push(value);
        }

        console.log('postData: ', postData);
        const jsonData = JSON.stringify(postData);
    }
);