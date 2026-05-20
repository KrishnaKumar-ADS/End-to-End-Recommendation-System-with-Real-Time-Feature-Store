FROM node:20-alpine

RUN apk add --no-cache libc6-compat

WORKDIR /app
COPY . .

ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN set -eux; \
    if [ -d "/app/project/frontend" ]; then \
    FRONTEND_DIR="/app/project/frontend"; \
    elif [ -d "/app/frontend" ]; then \
    FRONTEND_DIR="/app/frontend"; \
    else \
    echo "Frontend directory not found under /app/project/frontend or /app/frontend"; \
    exit 1; \
    fi; \
    npm --prefix "$FRONTEND_DIR" ci --include=dev; \
    npm --prefix "$FRONTEND_DIR" run build

COPY start-space.sh /usr/local/bin/start-space.sh
RUN chmod +x /usr/local/bin/start-space.sh

EXPOSE 7860

CMD ["/usr/local/bin/start-space.sh"]