"use client"

import type React from "react"

import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface GraphNode {
  id: string
  label: string
  type: string
  x?: number
  y?: number
}

interface GraphEdge {
  from: string
  to: string
  label: string
}

interface KnowledgeGraphProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  width?: number
  height?: number
}

export function KnowledgeGraph({ nodes, edges, width = 800, height = 600 }: KnowledgeGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [positions, setPositions] = useState<Map<string, { x: number; y: number }>>(new Map())

  // Physics-based layout algorithm
  useEffect(() => {
    if (!nodes.length) return

    // Initialize positions
    const nodePositions = new Map<string, { x: number; y: number; vx: number; vy: number }>()

    nodes.forEach((node, i) => {
      const angle = (i / nodes.length) * Math.PI * 2
      const radius = Math.min(width, height) * 0.3
      nodePositions.set(node.id, {
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      })
    })

    // Run force-directed layout simulation
    let iterations = 0
    const maxIterations = 100
    const interval = setInterval(() => {
      if (iterations++ >= maxIterations) {
        clearInterval(interval)
        return
      }

      // Repulsive forces between nodes
      nodes.forEach((node1) => {
        nodes.forEach((node2) => {
          if (node1.id === node2.id) return

          const pos1 = nodePositions.get(node1.id)!
          const pos2 = nodePositions.get(node2.id)!

          const dx = pos1.x - pos2.x
          const dy = pos1.y - pos2.y
          const distance = Math.sqrt(dx * dx + dy * dy) || 1

          const force = 1000 / (distance * distance)
          pos1.vx += (dx / distance) * force
          pos1.vy += (dy / distance) * force
        })
      })

      // Attractive forces along edges
      edges.forEach((edge) => {
        const pos1 = nodePositions.get(edge.from)
        const pos2 = nodePositions.get(edge.to)

        if (!pos1 || !pos2) return

        const dx = pos2.x - pos1.x
        const dy = pos2.y - pos1.y
        const distance = Math.sqrt(dx * dx + dy * dy) || 1

        const force = distance * 0.01
        pos1.vx += (dx / distance) * force
        pos1.vy += (dy / distance) * force
        pos2.vx -= (dx / distance) * force
        pos2.vy -= (dy / distance) * force
      })

      // Update positions with damping
      nodePositions.forEach((pos) => {
        pos.x += pos.vx
        pos.y += pos.vy
        pos.vx *= 0.8
        pos.vy *= 0.8

        // Keep within bounds
        pos.x = Math.max(50, Math.min(width - 50, pos.x))
        pos.y = Math.max(50, Math.min(height - 50, pos.y))
      })

      // Update state
      const newPositions = new Map()
      nodePositions.forEach((pos, id) => {
        newPositions.set(id, { x: pos.x, y: pos.y })
      })
      setPositions(newPositions)
    }, 16)

    return () => clearInterval(interval)
  }, [nodes, edges, width, height])

  // Render graph
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || positions.size === 0) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw edges
    ctx.strokeStyle = "#34d399"
    ctx.lineWidth = 2
    edges.forEach((edge) => {
      const from = positions.get(edge.from)
      const to = positions.get(edge.to)

      if (!from || !to) return

      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()

      // Draw edge label
      const midX = (from.x + to.x) / 2
      const midY = (from.y + to.y) / 2
      ctx.fillStyle = "#10b981"
      ctx.font = "12px monospace"
      ctx.textAlign = "center"
      ctx.fillText(edge.label, midX, midY)
    })

    // Draw nodes
    nodes.forEach((node) => {
      const pos = positions.get(node.id)
      if (!pos) return

      // Node circle
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2)
      ctx.fillStyle = node.type === "result" ? "#10b981" : "#6366f1"
      ctx.fill()
      ctx.strokeStyle = "#fff"
      ctx.lineWidth = 3
      ctx.stroke()

      // Node label
      ctx.fillStyle = "#fff"
      ctx.font = "14px monospace"
      ctx.textAlign = "center"
      ctx.fillText(node.label, pos.x, pos.y + 40)
    })
  }, [positions, nodes, edges, width, height])

  // Handle mouse hover
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    let found: GraphNode | null = null
    nodes.forEach((node) => {
      const pos = positions.get(node.id)
      if (!pos) return

      const distance = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2)
      if (distance < 20) {
        found = node
      }
    })

    setHoveredNode(found)
  }

  return (
    <Card className="border-emerald-500/20 bg-black/40">
      <CardHeader>
        <CardTitle className="text-emerald-400">Knowledge Graph</CardTitle>
        <CardDescription>Interactive visualization of reasoning relationships</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="border border-emerald-500/20 rounded-lg bg-black/60"
            onMouseMove={handleMouseMove}
            style={{ cursor: hoveredNode ? "pointer" : "default" }}
          />
          {hoveredNode && (
            <div className="absolute top-4 right-4 bg-black/90 border border-emerald-500/30 rounded-lg p-4">
              <div className="font-mono text-sm space-y-2">
                <div className="text-emerald-400 font-bold">{hoveredNode.label}</div>
                <Badge variant="outline" className="border-emerald-500/50 text-emerald-300">
                  {hoveredNode.type}
                </Badge>
              </div>
            </div>
          )}
        </div>
        <div className="mt-4 flex gap-4 text-sm text-gray-400">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-indigo-500" />
            <span>Concept Nodes</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-emerald-500" />
            <span>Result Nodes</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-emerald-500" />
            <span>Relationships</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
