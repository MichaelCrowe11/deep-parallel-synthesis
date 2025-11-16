import { cacheManager } from "@/lib/redis/cache-manager"
import { NextResponse } from "next/server"

export async function GET() {
  try {
    const analytics = await cacheManager.getAnalytics()

    return NextResponse.json({
      success: true,
      analytics,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] Cache analytics error:", error)
    return NextResponse.json({ error: "Failed to fetch cache analytics" }, { status: 500 })
  }
}
