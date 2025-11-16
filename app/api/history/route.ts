import { reasoningRepo } from "@/lib/db/reasoning-repo"
import { NextResponse } from "next/server"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const userId = searchParams.get("userId")
    const limit = Number.parseInt(searchParams.get("limit") || "20")

    if (!userId) {
      return NextResponse.json({ error: "userId is required" }, { status: 400 })
    }

    const history = await reasoningRepo.getUserHistory(userId, limit)

    return NextResponse.json({
      success: true,
      history,
      count: history.length,
    })
  } catch (error) {
    console.error("[v0] History fetch error:", error)
    return NextResponse.json({ error: "Failed to fetch history" }, { status: 500 })
  }
}
