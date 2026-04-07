import { useState } from 'react'

interface Props {
  onSubmit: (correctedAnswer: string, correctedReasoning: string) => void
  disabled?: boolean
}

export default function CorrectionEditor({ onSubmit, disabled }: Props) {
  const [answer, setAnswer] = useState('')
  const [reasoning, setReasoning] = useState('')

  const canSubmit = answer.trim().length > 0 && !disabled

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          修正后的正确答案
        </label>
        <textarea
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          placeholder="请输入您认为正确的答案..."
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          修正后的思维链
        </label>
        <textarea
          value={reasoning}
          onChange={(e) => setReasoning(e.target.value)}
          placeholder="请描述正确的推理过程..."
          rows={4}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <button
        onClick={() => onSubmit(answer, reasoning)}
        disabled={!canSubmit}
        className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium
          hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        提交修正
      </button>
    </div>
  )
}
