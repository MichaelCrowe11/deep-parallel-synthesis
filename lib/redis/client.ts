import { Redis } from "@upstash/redis"

// Singleton pattern for Redis client
let redisClient: Redis | null = null

export function getRedisClient() {
  if (!redisClient) {
    redisClient = new Redis({
      url: process.env.KV_REST_API_URL!,
      token: process.env.KV_REST_API_TOKEN!,
    })
  }
  return redisClient
}

// Cache helpers
export async function getCachedReasoning(query: string) {
  const redis = getRedisClient()
  const cacheKey = `reasoning:${Buffer.from(query).toString("base64").slice(0, 50)}`
  return await redis.get(cacheKey)
}

export async function setCachedReasoning(query: string, result: any, ttl = 3600) {
  const redis = getRedisClient()
  const cacheKey = `reasoning:${Buffer.from(query).toString("base64").slice(0, 50)}`
  await redis.setex(cacheKey, ttl, JSON.stringify(result))
}
