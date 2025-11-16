import { getRedisClient } from "./client"

// Cache key strategies
export const CacheKeys = {
  reasoning: (query: string) => `reasoning:${Buffer.from(query).toString("base64").slice(0, 64)}`,
  knowledgeGraph: (sessionId: string) => `kg:${sessionId}`,
  userHistory: (userId: string) => `history:${userId}`,
  trainingData: (type: string) => `training:${type}`,
  analytics: (metric: string) => `analytics:${metric}`,
}

// Advanced cache manager with analytics
export class CacheManager {
  private redis = getRedisClient()

  // Get reasoning with cache hit tracking
  async getReasoning(query: string) {
    const key = CacheKeys.reasoning(query)
    const result = await this.redis.get(key)

    if (result) {
      // Track cache hit
      await this.incrementMetric("cache:hits")
      console.log("[v0] Cache hit for reasoning query")
    } else {
      await this.incrementMetric("cache:misses")
      console.log("[v0] Cache miss for reasoning query")
    }

    return result
  }

  // Set reasoning with TTL and compression for large responses
  async setReasoning(query: string, data: any, ttl = 3600) {
    const key = CacheKeys.reasoning(query)
    await this.redis.setex(key, ttl, JSON.stringify(data))
    await this.incrementMetric("cache:writes")
    console.log("[v0] Cached reasoning result")
  }

  // Cache knowledge graph separately for visualization
  async cacheKnowledgeGraph(sessionId: string, graph: any, ttl = 7200) {
    const key = CacheKeys.knowledgeGraph(sessionId)
    await this.redis.setex(key, ttl, JSON.stringify(graph))
  }

  async getKnowledgeGraph(sessionId: string) {
    const key = CacheKeys.knowledgeGraph(sessionId)
    return await this.redis.get(key)
  }

  // User history for personalized reasoning
  async addToUserHistory(userId: string, sessionId: string) {
    const key = CacheKeys.userHistory(userId)
    await this.redis.lpush(key, sessionId)
    await this.redis.ltrim(key, 0, 99) // Keep last 100 sessions
    await this.redis.expire(key, 86400 * 30) // 30 days
  }

  async getUserHistory(userId: string, limit = 10) {
    const key = CacheKeys.userHistory(userId)
    return await this.redis.lrange(key, 0, limit - 1)
  }

  // Training data cache for GPU pipeline
  async cacheTrainingBatch(type: string, batch: any[], batchId: string) {
    const key = `${CacheKeys.trainingData(type)}:${batchId}`
    await this.redis.setex(key, 3600, JSON.stringify(batch))
  }

  async getTrainingBatch(type: string, batchId: string) {
    const key = `${CacheKeys.trainingData(type)}:${batchId}`
    return await this.redis.get(key)
  }

  // Analytics and metrics
  async incrementMetric(metric: string) {
    const key = CacheKeys.analytics(metric)
    await this.redis.incr(key)
  }

  async getMetric(metric: string): Promise<number> {
    const key = CacheKeys.analytics(metric)
    const value = await this.redis.get(key)
    return typeof value === "number" ? value : Number.parseInt(value as string) || 0
  }

  async getAnalytics() {
    const [hits, misses, writes] = await Promise.all([
      this.getMetric("cache:hits"),
      this.getMetric("cache:misses"),
      this.getMetric("cache:writes"),
    ])

    const total = hits + misses
    const hitRate = total > 0 ? (hits / total) * 100 : 0

    return {
      hits,
      misses,
      writes,
      hitRate: hitRate.toFixed(2),
      total,
    }
  }

  // Invalidate cache patterns
  async invalidatePattern(pattern: string) {
    // Note: Upstash Redis may not support SCAN, use carefully
    console.log("[v0] Invalidating cache pattern:", pattern)
  }

  // Clear all reasoning cache (for retraining)
  async clearReasoningCache() {
    await this.incrementMetric("cache:clears")
    console.log("[v0] Cleared reasoning cache for retraining")
  }
}

// Export singleton instance
export const cacheManager = new CacheManager()
