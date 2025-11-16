"use client"

import type React from "react"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, BarChart3, Brain, Database, Zap } from "lucide-react"

export default function DashboardPage() {
  const [analytics, setAnalytics] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("/api/analytics")
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setAnalytics(data)
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="animate-spin">
          <Brain className="size-12 text-emerald-500" />
        </div>
      </div>
    )
  }

  return (
    <main className="min-h-screen bg-black p-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <div>
          <h1 className="text-4xl font-bold text-emerald-400">System Dashboard</h1>
          <p className="text-gray-400 mt-2">Real-time analytics for Crowe Logic Framework</p>
        </div>

        {/* Stats Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <StatsCard
            icon={Database}
            title="Total Sessions"
            value={analytics?.database?.totalSessions || 0}
            description="Reasoning sessions stored"
          />
          <StatsCard
            icon={Zap}
            title="Cache Hit Rate"
            value={`${analytics?.cache?.hitRate || 0}%`}
            description="Redis cache performance"
          />
          <StatsCard
            icon={Activity}
            title="Avg Confidence"
            value={analytics?.database?.averageConfidence || 0}
            description="Overall reasoning quality"
          />
          <StatsCard
            icon={BarChart3}
            title="Avg Processing"
            value={`${analytics?.database?.averageProcessingTime || 0}ms`}
            description="Response time"
          />
        </div>

        {/* Type Distribution */}
        <Card className="border-emerald-500/20 bg-black/40">
          <CardHeader>
            <CardTitle className="text-emerald-400">Reasoning Type Distribution</CardTitle>
            <CardDescription>Sessions by problem category</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {analytics?.database?.typeDistribution &&
                Object.entries(analytics.database.typeDistribution).map(([type, count]: [string, any]) => (
                  <div key={type} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="border-emerald-500/50 text-emerald-300">
                        {type}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="h-2 w-32 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-500"
                          style={{
                            width: `${(count / analytics.database.totalSessions) * 100}%`,
                          }}
                        />
                      </div>
                      <span className="font-mono text-sm text-gray-400 w-12 text-right">{count}</span>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>

        {/* Recent Sessions */}
        <Card className="border-emerald-500/20 bg-black/40">
          <CardHeader>
            <CardTitle className="text-emerald-400">Recent Sessions</CardTitle>
            <CardDescription>Latest reasoning queries</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analytics?.database?.recentSessions?.slice(0, 5).map((session: any, i: number) => (
                <div key={i} className="flex items-center justify-between border-b border-gray-800 pb-3">
                  <div className="flex-1">
                    <Badge variant="outline" className="mb-1">
                      {session.reasoning_type}
                    </Badge>
                    <p className="text-sm text-gray-400">{new Date(session.created_at).toLocaleString()}</p>
                  </div>
                  {session.confidence_score && (
                    <Badge className="bg-emerald-500/20 text-emerald-400">
                      {Math.round(session.confidence_score * 100)}%
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}

function StatsCard({
  icon: Icon,
  title,
  value,
  description,
}: {
  icon: React.ElementType
  title: string
  value: string | number
  description: string
}) {
  return (
    <Card className="border-emerald-500/20 bg-black/40">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-gray-400">{title}</CardTitle>
        <Icon className="size-4 text-emerald-500" />
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold text-emerald-400 font-mono">{value}</div>
        <p className="text-xs text-gray-500 mt-1">{description}</p>
      </CardContent>
    </Card>
  )
}
