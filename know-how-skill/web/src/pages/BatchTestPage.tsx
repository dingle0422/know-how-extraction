import { useState, useEffect, useRef } from 'react'
import * as api from '../api/client'
import type { BatchTestResult } from '../types'

export default function BatchTestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [batchId, setBatchId] = useState('')
  const [result, setResult] = useState<BatchTestResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const pollRef = useRef<ReturnType<typeof setInterval>>()

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const res = await api.createBatchTest(file)
      setBatchId(res.id)
      setResult(res)
    } catch (e: any) {
      setError(e.message)
    }
    setLoading(false)
  }

  useEffect(() => {
    if (!batchId) return
    if (result?.status === 'completed' || result?.status === 'failed') return

    pollRef.current = setInterval(async () => {
      try {
        const res = await api.getBatchTest(batchId)
        setResult(res)
        if (res.status === 'completed' || res.status === 'failed') {
          clearInterval(pollRef.current)
        }
      } catch {
        clearInterval(pollRef.current)
      }
    }, 3000)

    return () => clearInterval(pollRef.current)
  }, [batchId, result?.status])

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <h1 className="text-lg font-bold text-gray-800">批量回测</h1>

      <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-4">
        <p className="text-sm text-gray-600">
          上传包含 <code className="bg-gray-100 px-1 rounded">question</code> 和
          <code className="bg-gray-100 px-1 rounded">answer</code> 列的 CSV/XLSX 文件
        </p>
        <div className="flex items-center gap-4">
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-sm"
          />
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium
              hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? '上传中...' : '开始回测'}
          </button>
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {result && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 text-sm">
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              result.status === 'completed' ? 'bg-green-100 text-green-700'
              : result.status === 'failed' ? 'bg-red-100 text-red-700'
              : 'bg-yellow-100 text-yellow-700'
            }`}>
              {result.status === 'completed' ? '已完成'
               : result.status === 'failed' ? '失败'
               : `进行中 ${result.completed}/${result.total}`}
            </span>
            {result.status === 'running' && (
              <div className="w-48 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: `${result.total ? (result.completed / result.total) * 100 : 0}%` }}
                />
              </div>
            )}
          </div>

          {result.results.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">#</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">问题</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">标准答案</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">模型答案</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {result.results.map((row) => (
                    <tr key={row.index} className="hover:bg-gray-50">
                      <td className="px-4 py-2 text-gray-500">{row.index}</td>
                      <td className="px-4 py-2 max-w-xs truncate" title={row.question}>
                        {row.question}
                      </td>
                      <td className="px-4 py-2 max-w-xs truncate text-gray-600" title={row.expected_answer}>
                        {row.expected_answer}
                      </td>
                      <td className="px-4 py-2 max-w-xs truncate text-gray-600" title={row.model_answer}>
                        {row.model_answer}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
