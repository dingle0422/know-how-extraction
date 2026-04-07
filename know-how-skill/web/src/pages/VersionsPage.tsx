import { useState, useEffect } from 'react'
import * as api from '../api/client'
import type { VersionInfo } from '../types'

export default function VersionsPage() {
  const [versions, setVersions] = useState<VersionInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [restoring, setRestoring] = useState<number | null>(null)

  const loadVersions = async () => {
    setLoading(true)
    try {
      const data = await api.listVersions()
      setVersions(data)
    } catch {
      // ignore
    }
    setLoading(false)
  }

  useEffect(() => {
    loadVersions()
  }, [])

  const handleRestore = async (id: number) => {
    if (!confirm('确定要恢复到此版本？当前 knowledge.json 将被覆盖。')) return
    setRestoring(id)
    try {
      await api.restoreVersion(id)
      alert('版本已恢复')
    } catch (e: any) {
      alert(`恢复失败: ${e.message}`)
    }
    setRestoring(null)
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-gray-800">版本管理</h1>
        <button
          onClick={loadVersions}
          disabled={loading}
          className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm hover:bg-gray-200 transition-colors"
        >
          {loading ? '加载中...' : '刷新'}
        </button>
      </div>

      {versions.length === 0 && !loading && (
        <div className="text-center py-12 text-gray-400 text-sm">
          暂无版本记录。通过增量修正工作流保存版本后，将在此处显示。
        </div>
      )}

      <div className="space-y-3">
        {versions.map((v) => (
          <div
            key={v.id}
            className="bg-white border border-gray-200 rounded-lg p-4 flex items-center justify-between"
          >
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded text-gray-500">
                  v{v.id}
                </span>
                <span className="text-sm font-medium text-gray-800">
                  {v.description || '(无描述)'}
                </span>
              </div>
              <div className="text-xs text-gray-400 space-x-4">
                <span>目录: {v.knowledge_dir}</span>
                <span>时间: {v.created_at}</span>
                {v.session_id && <span>会话: {v.session_id.slice(0, 8)}...</span>}
              </div>
            </div>
            <button
              onClick={() => handleRestore(v.id)}
              disabled={restoring === v.id}
              className="px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-700
                hover:bg-gray-50 disabled:opacity-50 transition-colors"
            >
              {restoring === v.id ? '恢复中...' : '恢复'}
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
