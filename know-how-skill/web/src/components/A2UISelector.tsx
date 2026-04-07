import { useState } from 'react'
import type { ErrorType } from '../types'
import { ERROR_TYPE_LABELS } from '../types'

interface Props {
  onSelect: (errorType: ErrorType, notes: string) => void
  disabled?: boolean
}

const btnStyles: Record<ErrorType, string> = {
  correct: 'border-green-300 bg-green-50 text-green-700 hover:bg-green-100',
  conclusion_error: 'border-red-300 bg-red-50 text-red-700 hover:bg-red-100',
  logic_error: 'border-orange-300 bg-orange-50 text-orange-700 hover:bg-orange-100',
  detail_error: 'border-yellow-300 bg-yellow-50 text-yellow-700 hover:bg-yellow-100',
  other: 'border-gray-300 bg-gray-50 text-gray-700 hover:bg-gray-100',
}

const selectedRing: Record<ErrorType, string> = {
  correct: 'ring-2 ring-green-500',
  conclusion_error: 'ring-2 ring-red-500',
  logic_error: 'ring-2 ring-orange-500',
  detail_error: 'ring-2 ring-yellow-500',
  other: 'ring-2 ring-gray-500',
}

export default function A2UISelector({ onSelect, disabled }: Props) {
  const [selected, setSelected] = useState<ErrorType | null>(null)
  const [notes, setNotes] = useState('')

  const needsNotes = selected === 'other'

  const handleClick = (type: ErrorType) => {
    if (type === 'correct') {
      onSelect(type, '')
      return
    }
    if (type === 'other') {
      setSelected(type)
      return
    }
    onSelect(type, notes)
  }

  const handleConfirmOther = () => {
    if (!notes.trim()) return
    onSelect('other', notes.trim())
  }

  return (
    <div className="space-y-3">
      <p className="text-sm font-medium text-gray-700">请评估模型回答质量：</p>
      <div className="flex flex-wrap gap-2">
        {(Object.entries(ERROR_TYPE_LABELS) as [ErrorType, string][]).map(([type, label]) => (
          <button
            key={type}
            disabled={disabled}
            onClick={() => handleClick(type)}
            className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all
              disabled:opacity-50 disabled:cursor-not-allowed
              ${btnStyles[type]}
              ${selected === type ? selectedRing[type] : ''}`}
          >
            {label}
          </button>
        ))}
      </div>

      {needsNotes && (
        <div className="space-y-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <label className="text-sm font-medium text-gray-700">
            请描述具体的错误类型 <span className="text-red-500">*</span>
          </label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="例如：数值计算错误、引用了过期政策、遗漏了关键前置条件..."
            rows={3}
            autoFocus
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none
              focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleConfirmOther}
            disabled={disabled || !notes.trim()}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium
              hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            确认提交
          </button>
        </div>
      )}
    </div>
  )
}
