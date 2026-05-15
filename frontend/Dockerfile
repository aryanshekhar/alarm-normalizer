# ── Build stage ───────────────────────────────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# ── Serve stage ───────────────────────────────────────────────────────────────
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/nginx.conf.template
ENV BACKEND_URL=http://backend:8000
EXPOSE 80
# envsubst only replaces ${BACKEND_URL}; native nginx vars ($host etc.) are preserved
CMD ["/bin/sh", "-c", \
  "envsubst '${BACKEND_URL}' < /etc/nginx/conf.d/nginx.conf.template \
   > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
