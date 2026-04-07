export type SessionState =
  | 'idle'
  | 'inferring'
  | 'awaiting_eval'
  | 'awaiting_correction'
  | 'showing_activated_kh'
  | 'awaiting_kh_selection'
  | 'showing_diff'
  | 'testing'
  | 'awaiting_test_eval'
  | 'saved'

export type ErrorType =
  | 'correct'
  | 'conclusion_error'
  | 'logic_error'
  | 'detail_error'
  | 'other'

export type RollbackTarget =
  | 'awaiting_eval'
  | 'awaiting_correction'
  | 'awaiting_kh_selection'

export interface Session {
  id: string
  state: SessionState
  question: string
  knowledge_dirs: string[]
  inference_result: Record<string, unknown>
  expert_evaluation: Record<string, unknown>
  expert_correction: Record<string, unknown>
  activated_knowhow: ActivatedKnowHowItem[]
  selected_entry_keys: string[]
  patches: Record<string, PatchInfo>
  test_result: Record<string, unknown>
}

export interface ActivatedKnowHowItem {
  entry_key: string
  source_dir: string
  knowledge_dir: string
  title: string
  scope: string
  kh_text: string
  reasoning_chain: string
  derived_answer: string
}

export interface PatchInfo {
  entry_key: string
  title: string
  original: Record<string, unknown>
  patched: Record<string, unknown>
  original_text: string
  patched_text: string
  operations: Record<string, unknown>[]
  patch_log: Record<string, unknown>[]
  diff_description: string
}

export type KHUpdateMode = 'idle' | 'ai_loading' | 'ai_done' | 'manual'

export interface SinglePatchResult {
  entry_key: string
  title: string
  original_text: string
  patched_text: string
  original_json: Record<string, unknown>
  patched_json: Record<string, unknown>
  operations: Record<string, unknown>[]
  diff_description: string
}

export interface GenerateNewKHResult {
  knowhow_json: Record<string, unknown>
  knowhow_text: string
}

export interface DiffItem {
  entry_key: string
  title: string
  original_text: string
  patched_text: string
  original_json: Record<string, unknown>
  patched_json: Record<string, unknown>
  operations: Record<string, unknown>[]
  diff_description: string
}

export interface BatchTestResult {
  id: string
  status: string
  total: number
  completed: number
  results: BatchTestRow[]
}

export interface BatchTestRow {
  index: number
  question: string
  expected_answer: string
  model_answer: string
  match: string
}

export interface VersionInfo {
  id: number
  knowledge_dir: string
  description: string
  session_id: string
  created_at: string
}

export const STEP_LABELS: Record<SessionState, string> = {
  idle: '提问',
  inferring: '推理中',
  awaiting_eval: '评估',
  awaiting_correction: '修正',
  showing_activated_kh: '加载知识块',
  awaiting_kh_selection: '选择知识块',
  showing_diff: 'Diff 对比',
  testing: '回测中',
  awaiting_test_eval: '验证结果',
  saved: '已保存',
}

export const STEP_ORDER: SessionState[] = [
  'idle',
  'inferring',
  'awaiting_eval',
  'awaiting_correction',
  'showing_activated_kh',
  'awaiting_kh_selection',
  'showing_diff',
  'testing',
  'awaiting_test_eval',
  'saved',
]

export const ERROR_TYPE_LABELS: Record<ErrorType, string> = {
  correct: '回答正确',
  conclusion_error: '结论错误',
  logic_error: '逻辑错误',
  detail_error: '细节错误',
  other: '其他问题',
}
