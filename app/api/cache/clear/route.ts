import { cacheManager } from "@/lib/redis/cache-manager"
import { NextResponse } from "next/server"

export async function POST() {
  try {
    await cacheManager.clearReasoningCache()

    return NextResponse.json({
      success: true,
      message: "Cache cleared successfully",
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] Cache clear error:", error)
    return NextResponse.json({ error: "Failed to clear cache" }, { status: 500 })
  }
}
