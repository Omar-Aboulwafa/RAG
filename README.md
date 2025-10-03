# Deploy to production
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Check services
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f phoenix
docker-compose -f docker-compose.prod.yml logs -f backend-api

# Scale backend for high traffic
docker-compose -f docker-compose.prod.yml up -d --scale backend-api=3
