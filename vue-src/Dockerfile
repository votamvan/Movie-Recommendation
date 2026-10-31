FROM node:26.13.0-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install -g npm@12.2.1 && npm ci && npm uninstall -g npm
COPY --chown=node:node . .
RUN mkdir -p /app/node_modules/.vite && chown node:node /app /app/node_modules/.vite
USER node
CMD ["node", "node_modules/vite/bin/vite.js", "--host", "0.0.0.0", "--port", "8080"]
