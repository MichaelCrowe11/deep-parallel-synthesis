-- Enhance reasoning_sessions table with knowledge graph support
ALTER TABLE reasoning_sessions 
ADD COLUMN IF NOT EXISTS knowledge_graph JSONB,
ADD COLUMN IF NOT EXISTS confidence_score DECIMAL(3,2),
ADD COLUMN IF NOT EXISTS reasoning_type TEXT,
ADD COLUMN IF NOT EXISTS cached BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_reasoning_sessions_user_id ON reasoning_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_sessions_created_at ON reasoning_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reasoning_sessions_type ON reasoning_sessions(reasoning_type);

-- Create table for knowledge graph nodes
CREATE TABLE IF NOT EXISTS knowledge_nodes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
  node_id TEXT NOT NULL,
  label TEXT NOT NULL,
  type TEXT NOT NULL,
  properties JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create table for knowledge graph edges
CREATE TABLE IF NOT EXISTS knowledge_edges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
  from_node TEXT NOT NULL,
  to_node TEXT NOT NULL,
  label TEXT NOT NULL,
  properties JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE knowledge_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_edges ENABLE ROW LEVEL SECURITY;

-- RLS policies for knowledge nodes
CREATE POLICY "Users can view own knowledge nodes"
  ON knowledge_nodes FOR SELECT
  USING (session_id IN (
    SELECT id FROM reasoning_sessions WHERE user_id = auth.uid()
  ));

CREATE POLICY "Users can insert own knowledge nodes"
  ON knowledge_nodes FOR INSERT
  WITH CHECK (session_id IN (
    SELECT id FROM reasoning_sessions WHERE user_id = auth.uid()
  ));

-- RLS policies for knowledge edges
CREATE POLICY "Users can view own knowledge edges"
  ON knowledge_edges FOR SELECT
  USING (session_id IN (
    SELECT id FROM reasoning_sessions WHERE user_id = auth.uid()
  ));

CREATE POLICY "Users can insert own knowledge edges"
  ON knowledge_edges FOR INSERT
  WITH CHECK (session_id IN (
    SELECT id FROM reasoning_sessions WHERE user_id = auth.uid()
  ));
