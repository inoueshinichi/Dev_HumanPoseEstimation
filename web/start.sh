#!/bin/sh

# NginxのWebサーバ設定からリバースプロキシーサーバ設定への変更
sed -e "s/{{PORT}}/$PORT/g" /etc/nginx/nginx.conf.tmp > /etc/nginx/nginx.conf
sed -i -e "s^{{APP_SERVER}}^$APP_SERVER^g" /etc/nginx/nginx.conf

# 設定ファイルをログで確認
echo '[Check Nginx configuration file]'
cat /etc/nginx/nginx.conf

# PID1としてNginxを起動する
exec nginx -g "daemon off;"