// Knowledge graph builder for reasoning chains
export interface KnowledgeGraphNode {
  id: string
  label: string
  type: "concept" | "operation" | "result" | "premise" | "conclusion"
  properties?: Record<string, any>
}

export interface KnowledgeGraphEdge {
  from: string
  to: string
  label: string
  properties?: Record<string, any>
}

export class KnowledgeGraphBuilder {
  private nodes: KnowledgeGraphNode[] = []
  private edges: KnowledgeGraphEdge[] = []
  private nodeIdCounter = 0

  addNode(label: string, type: KnowledgeGraphNode["type"], properties?: Record<string, any>) {
    const id = `node_${this.nodeIdCounter++}`
    this.nodes.push({ id, label, type, properties })
    return id
  }

  addEdge(from: string, to: string, label: string, properties?: Record<string, any>) {
    this.edges.push({ from, to, label, properties })
  }

  // Extract graph from reasoning phases
  buildFromReasoning(phases: any[]) {
    const phaseNodes: string[] = []

    phases.forEach((phase, index) => {
      const phaseId = this.addNode(phase.name, "concept", {
        confidence: phase.confidence,
        content: phase.content,
      })
      phaseNodes.push(phaseId)

      if (index > 0) {
        this.addEdge(phaseNodes[index - 1], phaseId, "leads_to")
      }

      // Extract key concepts from content
      const concepts = this.extractConcepts(phase.content)
      concepts.forEach((concept) => {
        const conceptId = this.addNode(concept, "operation")
        this.addEdge(phaseId, conceptId, "uses")
      })
    })

    // Add final result node
    const resultId = this.addNode("Final Answer", "result")
    if (phaseNodes.length > 0) {
      this.addEdge(phaseNodes[phaseNodes.length - 1], resultId, "produces")
    }

    return this.build()
  }

  // Simple concept extraction (can be enhanced with NLP)
  private extractConcepts(content: string): string[] {
    const concepts: string[] = []

    // Math operations
    const mathOps = ["addition", "subtraction", "multiplication", "division", "equation", "formula"]
    mathOps.forEach((op) => {
      if (content.toLowerCase().includes(op)) {
        concepts.push(op)
      }
    })

    // Logic operations
    const logicOps = ["premise", "conclusion", "inference", "deduction", "induction"]
    logicOps.forEach((op) => {
      if (content.toLowerCase().includes(op)) {
        concepts.push(op)
      }
    })

    return [...new Set(concepts)] // Remove duplicates
  }

  build() {
    return {
      nodes: this.nodes,
      edges: this.edges,
    }
  }

  clear() {
    this.nodes = []
    this.edges = []
    this.nodeIdCounter = 0
  }
}
