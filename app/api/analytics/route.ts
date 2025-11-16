import { reasoningRepo } from "@/lib/db/reasoning-repo"
import { cacheManager } from "@/lib/redis/cache-manager"
import { NextResponse } from "next/server"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const userId = searchParams.get("userId") || undefined

    const [dbAnalytics, cacheAnalytics] = await Promise.all([
      reasoningRepo.getAnalytics(userId),
      cacheManager.getAnalytics(),
    ])

    return NextResponse.json({
      success: true,
      database: dbAnalytics,
      cache: cacheAnalytics,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] Analytics fetch error:", error)
    return NextResponse.json({ error: "Failed to fetch analytics" }, { status: 500 })
  }
}
