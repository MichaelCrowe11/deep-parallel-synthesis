import { createClient } from "@/lib/supabase/server"

export interface ReasoningSession {
  id: string
  query: string
  response: string
  model: string
  confidence_score?: number
  reasoning_type?: string
  knowledge_graph?: any
  cached?: boolean
  processing_time_ms?: number
  credits_used: number
  user_id?: string
  created_at: string
}

export interface KnowledgeNode {
  id: string
  session_id: string
  node_id: string
  label: string
  type: string
  properties?: any
  created_at: string
}

export interface KnowledgeEdge {
  id: string
  session_id: string
  from_node: string
  to_node: string
  label: string
  properties?: any
  created_at: string
}

export class ReasoningRepository {
  // Save complete reasoning session with knowledge graph
  async saveSession(data: {
    query: string
    response: any
    model: string
    confidence_score?: number
    reasoning_type: string
    knowledge_graph?: any
    processing_time_ms?: number
    user_id?: string
  }) {
    const supabase = await createClient()

    const { data: session, error } = await supabase
      .from("reasoning_sessions")
      .insert({
        query: data.query,
        response: JSON.stringify(data.response),
        model: data.model,
        confidence_score: data.confidence_score,
        reasoning_type: data.reasoning_type,
        knowledge_graph: data.knowledge_graph,
        processing_time_ms: data.processing_time_ms,
        credits_used: 1,
        user_id: data.user_id,
      })
      .select()
      .single()

    if (error) {
      console.error("[v0] Failed to save reasoning session:", error)
      throw error
    }

    // Save knowledge graph nodes and edges separately
    if (data.knowledge_graph && session) {
      await this.saveKnowledgeGraph(session.id, data.knowledge_graph)
    }

    return session
  }

  // Save knowledge graph structure
  async saveKnowledgeGraph(sessionId: string, graph: any) {
    const supabase = await createClient()

    if (graph.nodes && graph.nodes.length > 0) {
      const nodes = graph.nodes.map((node: any) => ({
        session_id: sessionId,
        node_id: node.id,
        label: node.label,
        type: node.type,
        properties: node.properties || {},
      }))

      const { error: nodesError } = await supabase.from("knowledge_nodes").insert(nodes)

      if (nodesError) {
        console.error("[v0] Failed to save knowledge nodes:", nodesError)
      }
    }

    if (graph.edges && graph.edges.length > 0) {
      const edges = graph.edges.map((edge: any) => ({
        session_id: sessionId,
        from_node: edge.from,
        to_node: edge.to,
        label: edge.label,
        properties: edge.properties || {},
      }))

      const { error: edgesError } = await supabase.from("knowledge_edges").insert(edges)

      if (edgesError) {
        console.error("[v0] Failed to save knowledge edges:", edgesError)
      }
    }
  }

  // Get user's reasoning history
  async getUserHistory(userId: string, limit = 20) {
    const supabase = await createClient()

    const { data, error } = await supabase
      .from("reasoning_sessions")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(limit)

    if (error) {
      console.error("[v0] Failed to fetch user history:", error)
      throw error
    }

    return data as ReasoningSession[]
  }

  // Get reasoning session with knowledge graph
  async getSessionWithGraph(sessionId: string) {
    const supabase = await createClient()

    const [sessionResult, nodesResult, edgesResult] = await Promise.all([
      supabase.from("reasoning_sessions").select("*").eq("id", sessionId).single(),
      supabase.from("knowledge_nodes").select("*").eq("session_id", sessionId),
      supabase.from("knowledge_edges").select("*").eq("session_id", sessionId),
    ])

    if (sessionResult.error) {
      throw sessionResult.error
    }

    return {
      session: sessionResult.data as ReasoningSession,
      nodes: nodesResult.data as KnowledgeNode[],
      edges: edgesResult.data as KnowledgeEdge[],
    }
  }

  // Get training data for specific reasoning type
  async getTrainingData(reasoningType: string, limit = 1000, minConfidence = 0.7) {
    const supabase = await createClient()

    const { data, error } = await supabase
      .from("reasoning_sessions")
      .select("query, response, confidence_score, reasoning_type")
      .eq("reasoning_type", reasoningType)
      .gte("confidence_score", minConfidence)
      .order("created_at", { ascending: false })
      .limit(limit)

    if (error) {
      console.error("[v0] Failed to fetch training data:", error)
      throw error
    }

    return data
  }

  // Get analytics for dashboard
  async getAnalytics(userId?: string) {
    const supabase = await createClient()

    let query = supabase
      .from("reasoning_sessions")
      .select("reasoning_type, confidence_score, processing_time_ms, created_at")

    if (userId) {
      query = query.eq("user_id", userId)
    }

    const { data, error } = await query.order("created_at", { ascending: false })

    if (error) {
      console.error("[v0] Failed to fetch analytics:", error)
      throw error
    }

    // Calculate statistics
    const typeDistribution: Record<string, number> = {}
    let totalConfidence = 0
    let totalProcessingTime = 0
    let count = 0

    data.forEach((session: any) => {
      const type = session.reasoning_type || "unknown"
      typeDistribution[type] = (typeDistribution[type] || 0) + 1

      if (session.confidence_score) {
        totalConfidence += session.confidence_score
      }

      if (session.processing_time_ms) {
        totalProcessingTime += session.processing_time_ms
      }

      count++
    })

    return {
      totalSessions: count,
      typeDistribution,
      averageConfidence: count > 0 ? (totalConfidence / count).toFixed(3) : 0,
      averageProcessingTime: count > 0 ? Math.round(totalProcessingTime / count) : 0,
      recentSessions: data.slice(0, 10),
    }
  }
}

export const reasoningRepo = new ReasoningRepository()
