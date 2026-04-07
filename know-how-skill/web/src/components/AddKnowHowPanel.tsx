import { useState } from 'react'

const EMPTY_KH = JSON.stringify({
  title: '',
  scope: '',
  steps: [
    { step: '1', action: '', condition: null, constraint: null, policy_basis: null, outcome: null },
  ],
  exceptions: [],
}, null, 2)

interface Props {
  loading: boolean
  onGenerate: () => Promise<{ knowhow_json: Record<string, unknown>; knowhow_text: string } | null>
  onAdd: (knowhowJson: Record<string, unknown>) => void
  onCancel: () => void
}

export default function AddKnowHowPanel({ loading, onGenerate, onAdd, onCancel }: Props) {
  const [editJson, setEditJson] = useState(EMPTY_KH)
  const [generated, setGenerated] = useState(false)

  const handleGenerate = async () => {
    const result = await onGenerate()
    if (result?.knowhow_json) {
      setEditJson(JSON.stringify(result.knowhow_json, null, 2))
      setGenerated(true)
    }
  }

  const handleAdd = () => {
    try {
      const parsed = JSON.parse(editJson)
      onAdd(parsed)
    } catch {
      alert('JSON 格式不合法，请检查后重试')
    }
  }

  return (
    <div className="border-2 border-dashed border-blue-300 rounded-lg bg-blue-50/30 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-blue-800">新增知识块</h3>
        <button
          onClick={onCancel}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          取消
        </button>
      </div>

      <div className="flex gap-2">
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="px-4 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-medium
            hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'AI 生成中...' : generated ? 'AI 重新生成' : 'AI 生成'}
        </button>
        <span className="text-xs text-gray-500 self-center">
          或直接在下方编辑 JSON
        </span>
      </div>

      <div>
        <label className="text-xs text-gray-600 font-medium mb-1 block">
          知识块内容（JSON 格式）：
        </label>
        <textarea
          value={editJson}
          onChange={(e) => setEditJson(e.target.value)}
          rows={14}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-xs font-mono resize-y
            focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
        />
      </div>

      <button
        onClick={handleAdd}
        disabled={loading}
        className="px-6 py-2 bg-green-600 text-white rounded-lg text-sm font-medium
          hover:bg-green-700 disabled:opacity-50 transition-colors"
      >
        确认添加
      </button>
    </div>
  )
}
