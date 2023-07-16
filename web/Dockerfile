# Dockerfile for Nginx

# Image
From nginx:1.17.5-alpine

Label author="inoue.shinichi.1800@gmail.com"

COPY ./start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

COPY ./nginx.conf.tmp /etc/nginx/nginx.conf.tmp
COPY ./html/ /usr/share/nginx/html/

ENV PORT=80
EXPOSE 80
ENV APP_SERVER "http://127.0.0.1:8080"

CMD ["sh", "/usr/local/bin/start.sh"]

