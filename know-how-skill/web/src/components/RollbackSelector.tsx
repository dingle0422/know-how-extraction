import type { RollbackTarget } from '../types'

interface Props {
  onSelect: (target: RollbackTarget) => void
  disabled?: boolean
}

const options: { target: RollbackTarget; label: string; desc: string }[] = [
  { target: 'awaiting_eval', label: '重新评估', desc: '回到模型回答评估阶段' },
  { target: 'awaiting_correction', label: '重新修正', desc: '回到修正答案编辑阶段' },
  { target: 'awaiting_kh_selection', label: '重选知识块', desc: '回到知识块选择阶段' },
]

export default function RollbackSelector({ onSelect, disabled }: Props) {
  return (
    <div className="space-y-3">
      <p className="text-sm font-medium text-gray-700">回测结果不满意，请选择回退到哪一步：</p>
      <div className="space-y-2">
        {options.map((opt) => (
          <button
            key={opt.target}
            disabled={disabled}
            onClick={() => onSelect(opt.target)}
            className="w-full text-left px-4 py-3 rounded-lg border border-gray-200 bg-white
              hover:border-blue-300 hover:bg-blue-50 transition-all
              disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <div className="text-sm font-medium text-gray-800">{opt.label}</div>
            <div className="text-xs text-gray-500">{opt.desc}</div>
          </button>
        ))}
      </div>
    </div>
  )
}
