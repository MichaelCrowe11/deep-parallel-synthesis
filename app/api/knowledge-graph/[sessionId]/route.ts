import { reasoningRepo } from "@/lib/db/reasoning-repo"
import { cacheManager } from "@/lib/redis/cache-manager"
import { NextResponse } from "next/server"

export async function GET(req: Request, { params }: { params: { sessionId: string } }) {
  try {
    const { sessionId } = params

    // Check cache first
    const cached = await cacheManager.getKnowledgeGraph(sessionId)
    if (cached) {
      console.log("[v0] Returning cached knowledge graph")
      return NextResponse.json({
        success: true,
        cached: true,
        ...JSON.parse(cached as string),
      })
    }

    // Fetch from database
    const { session, nodes, edges } = await reasoningRepo.getSessionWithGraph(sessionId)

    const graphData = {
      session: {
        id: session.id,
        query: session.query,
        type: session.reasoning_type,
        confidence: session.confidence_score,
      },
      nodes: nodes.map((n) => ({
        id: n.node_id,
        label: n.label,
        type: n.type,
        properties: n.properties,
      })),
      edges: edges.map((e) => ({
        from: e.from_node,
        to: e.to_node,
        label: e.label,
        properties: e.properties,
      })),
    }

    // Cache for future requests
    await cacheManager.cacheKnowledgeGraph(sessionId, graphData)

    return NextResponse.json({
      success: true,
      cached: false,
      ...graphData,
    })
  } catch (error) {
    console.error("[v0] Knowledge graph fetch error:", error)
    return NextResponse.json({ error: "Failed to fetch knowledge graph" }, { status: 500 })
  }
}
