import ReactDiffViewer, { DiffMethod } from 'react-diff-viewer-continued'
import { useState } from 'react'
import type { DiffItem } from '../types'

interface Props {
  items: DiffItem[]
}

export default function DiffViewer({ items }: Props) {
  const [viewMode, setViewMode] = useState<'text' | 'json'>('text')

  if (!items.length) {
    return <p className="text-sm text-gray-500 p-4">暂无 Diff 数据</p>
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <button
          onClick={() => setViewMode('text')}
          className={`px-3 py-1 rounded text-xs font-medium ${
            viewMode === 'text' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600'
          }`}
        >
          文本视图
        </button>
        <button
          onClick={() => setViewMode('json')}
          className={`px-3 py-1 rounded text-xs font-medium ${
            viewMode === 'json' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600'
          }`}
        >
          JSON 视图
        </button>
      </div>

      {items.map((item) => (
        <div key={item.entry_key} className="border border-gray-200 rounded-lg overflow-hidden">
          <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
            <span className="text-xs font-mono text-gray-500 mr-2">[{item.entry_key}]</span>
            <span className="text-sm font-medium text-gray-800">{item.title}</span>
            {item.diff_description && (
              <p className="text-xs text-gray-500 mt-1">{item.diff_description}</p>
            )}
          </div>
          <ReactDiffViewer
            oldValue={
              viewMode === 'text'
                ? item.original_text
                : JSON.stringify(item.original_json, null, 2)
            }
            newValue={
              viewMode === 'text'
                ? item.patched_text
                : JSON.stringify(item.patched_json, null, 2)
            }
            splitView={true}
            compareMethod={DiffMethod.WORDS}
            leftTitle="修正前"
            rightTitle="修正后"
            styles={{
              contentText: { fontSize: '12px', lineHeight: '1.6' },
            }}
          />
        </div>
      ))}
    </div>
  )
}
