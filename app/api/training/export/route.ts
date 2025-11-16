import { reasoningRepo } from "@/lib/db/reasoning-repo"
import { NextResponse } from "next/server"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const type = searchParams.get("type") || "math"
    const minConfidence = Number.parseFloat(searchParams.get("minConfidence") || "0.7")
    const limit = Number.parseInt(searchParams.get("limit") || "10000")

    console.log("[v0] Exporting training data:", { type, minConfidence, limit })

    const data = await reasoningRepo.getTrainingData(type, limit, minConfidence)

    return NextResponse.json({
      success: true,
      data,
      count: data.length,
      filters: { type, minConfidence, limit },
      exported_at: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] Training export error:", error)
    return NextResponse.json({ error: "Failed to export training data" }, { status: 500 })
  }
}
