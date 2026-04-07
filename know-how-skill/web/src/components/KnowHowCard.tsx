import { useState } from 'react'
import type { ActivatedKnowHowItem, SinglePatchResult } from '../types'

interface Props {
  item: ActivatedKnowHowItem
  patchResult?: SinglePatchResult
  isLoading?: boolean
  onAiUpdate: () => void
  onManualSave: (patchedJson: Record<string, unknown>) => void
}

export default function KnowHowCard({
  item, patchResult, isLoading, onAiUpdate, onManualSave,
}: Props) {
  const [mode, setMode] = useState<'view' | 'manual' | 'ai_review'>('view')
  const [editJson, setEditJson] = useState('')

  const hasBeenPatched = !!patchResult

  const handleStartManual = () => {
    const base = patchResult?.patched_json ?? tryParseKhText(item.kh_text)
    setEditJson(JSON.stringify(base, null, 2))
    setMode('manual')
  }

  const handleManualSave = () => {
    try {
      const parsed = JSON.parse(editJson)
      onManualSave(parsed)
      setMode('view')
    } catch {
      alert('JSON 格式不合法，请检查后重试')
    }
  }

  const handleAiUpdate = () => {
    onAiUpdate()
    setMode('ai_review')
  }

  const handleEditAiResult = () => {
    if (patchResult) {
      setEditJson(JSON.stringify(patchResult.patched_json, null, 2))
      setMode('manual')
    }
  }

  return (
    <div className={`rounded-lg border-2 transition-all ${
      hasBeenPatched
        ? 'border-green-400 bg-green-50'
        : 'border-gray-200 bg-white'
    }`}>
      {/* Header */}
      <div className="p-4 pb-2">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono bg-gray-100 px-2 py-0.5 rounded text-gray-500">
            {item.entry_key}
          </span>
          <h3 className="text-sm font-semibold text-gray-900 truncate">
            {item.title || '(无标题)'}
          </h3>
          {hasBeenPatched && (
            <span className="text-xs bg-green-200 text-green-800 px-2 py-0.5 rounded-full">
              已修改
            </span>
          )}
        </div>
        {item.scope && (
          <p className="text-xs text-gray-500 mb-2">{item.scope}</p>
        )}
      </div>

      {/* Content */}
      {mode === 'view' && (
        <div className="px-4 pb-2">
          {hasBeenPatched ? (
            <div className="space-y-2">
              <div className="text-xs text-gray-500 font-medium">修正后预览：</div>
              <pre className="text-xs text-gray-700 whitespace-pre-wrap bg-green-100/50 rounded p-2 max-h-40 overflow-y-auto">
                {patchResult!.patched_text || JSON.stringify(patchResult!.patched_json, null, 2)}
              </pre>
              {patchResult!.diff_description && (
                <p className="text-xs text-gray-500 italic">{patchResult!.diff_description}</p>
              )}
            </div>
          ) : (
            <pre className="text-xs text-gray-600 whitespace-pre-wrap bg-gray-50 rounded p-2 max-h-32 overflow-y-auto">
              {item.kh_text.slice(0, 500)}
              {item.kh_text.length > 500 && '...'}
            </pre>
          )}
        </div>
      )}

      {/* Manual edit mode */}
      {mode === 'manual' && (
        <div className="px-4 pb-2 space-y-2">
          <div className="text-xs text-gray-500 font-medium">编辑知识块 JSON：</div>
          <textarea
            value={editJson}
            onChange={(e) => setEditJson(e.target.value)}
            rows={12}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-xs font-mono resize-y
              focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <div className="flex gap-2">
            <button
              onClick={handleManualSave}
              className="px-4 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-medium
                hover:bg-blue-700 transition-colors"
            >
              保存修改
            </button>
            <button
              onClick={() => setMode('view')}
              className="px-4 py-1.5 border border-gray-300 text-gray-600 rounded-lg text-xs
                hover:bg-gray-50 transition-colors"
            >
              取消
            </button>
          </div>
        </div>
      )}

      {/* AI review mode */}
      {mode === 'ai_review' && isLoading && (
        <div className="px-4 pb-2">
          <div className="flex items-center gap-2 text-xs text-blue-600 py-3">
            <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            AI 正在生成修正方案...
          </div>
        </div>
      )}
      {mode === 'ai_review' && !isLoading && hasBeenPatched && (
        <div className="px-4 pb-2 space-y-2">
          <div className="text-xs text-gray-500 font-medium">AI 修正结果：</div>
          <pre className="text-xs text-gray-700 whitespace-pre-wrap bg-blue-50 rounded p-2 max-h-40 overflow-y-auto">
            {patchResult!.patched_text || JSON.stringify(patchResult!.patched_json, null, 2)}
          </pre>
          {patchResult!.diff_description && (
            <p className="text-xs text-gray-500 italic">{patchResult!.diff_description}</p>
          )}
          <div className="flex gap-2">
            <button
              onClick={() => setMode('view')}
              className="px-4 py-1.5 bg-green-600 text-white rounded-lg text-xs font-medium
                hover:bg-green-700 transition-colors"
            >
              确认采纳
            </button>
            <button
              onClick={handleEditAiResult}
              className="px-4 py-1.5 border border-blue-300 text-blue-700 rounded-lg text-xs
                hover:bg-blue-50 transition-colors"
            >
              二次编辑
            </button>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="px-4 py-3 border-t border-gray-100 flex gap-2">
        <button
          onClick={handleAiUpdate}
          disabled={isLoading}
          className="px-4 py-1.5 bg-blue-600 text-white rounded-lg text-xs font-medium
            hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? 'AI 处理中...' : hasBeenPatched ? 'AI 重新生成' : 'AI 更新'}
        </button>
        <button
          onClick={handleStartManual}
          disabled={isLoading}
          className="px-4 py-1.5 border border-gray-300 text-gray-700 rounded-lg text-xs font-medium
            hover:bg-gray-50 disabled:opacity-50 transition-colors"
        >
          手动更新
        </button>
      </div>
    </div>
  )
}

function tryParseKhText(text: string): Record<string, unknown> {
  try {
    return JSON.parse(text)
  } catch {
    return { title: '', scope: '', steps: [], exceptions: [] }
  }
}
