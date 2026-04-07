import type { SessionState } from '../types'
import { STEP_ORDER, STEP_LABELS } from '../types'

const VISIBLE_STEPS: SessionState[] = [
  'idle', 'awaiting_eval', 'awaiting_correction',
  'awaiting_kh_selection', 'showing_diff', 'awaiting_test_eval', 'saved',
]

interface Props {
  currentState: SessionState
}

export default function StepNavigator({ currentState }: Props) {
  const currentIdx = STEP_ORDER.indexOf(currentState)

  return (
    <div className="flex items-center gap-1 px-4 py-3 bg-white border-b border-gray-200 overflow-x-auto">
      {VISIBLE_STEPS.map((step, i) => {
        const stepIdx = STEP_ORDER.indexOf(step)
        const isActive = step === currentState
        const isPast = stepIdx < currentIdx
        const isBetween = !isActive && !isPast && STEP_ORDER.indexOf(step) > currentIdx

        return (
          <div key={step} className="flex items-center">
            {i > 0 && (
              <div className={`w-6 h-0.5 mx-1 ${isPast ? 'bg-blue-400' : 'bg-gray-200'}`} />
            )}
            <div
              className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : isPast
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-400'
              }`}
            >
              {STEP_LABELS[step]}
            </div>
          </div>
        )
      })}
    </div>
  )
}
