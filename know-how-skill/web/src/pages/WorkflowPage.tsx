import { useState, useEffect } from 'react'
import { useSessionStore } from '../stores/sessionStore'
import StepNavigator from '../components/StepNavigator'
import A2UISelector from '../components/A2UISelector'
import CorrectionEditor from '../components/CorrectionEditor'
import KnowHowCard from '../components/KnowHowCard'
import DiffViewer from '../components/DiffViewer'
import RollbackSelector from '../components/RollbackSelector'
import AddKnowHowPanel from '../components/AddKnowHowPanel'
import type { SessionState, ErrorType, RollbackTarget } from '../types'

export default function WorkflowPage() {
  const {
    session, loading, error, diffItems, activatedKH,
    patchResults, perBlockLoading,
    createSession, submitQuestion, submitEval, submitCorrection,
    loadActivatedKH, aiUpdateBlock, manualUpdateBlock,
    generateNewKH, addNewKH, confirmAllPatches,
    loadDiff, runTest, save, rollback,
    clearError,
  } = useSessionStore()

  const [question, setQuestion] = useState('')
  const [showAddPanel, setShowAddPanel] = useState(false)
  const [kdInput, setKdInput] = useState('')
  const [showKdConfig, setShowKdConfig] = useState(false)

  const hasKnowledgeDirs = (session?.knowledge_dirs?.length ?? 0) > 0

  useEffect(() => {
    if (!session) {
      createSession()
    }
  }, [])

  useEffect(() => {
    if (session?.state === 'awaiting_kh_selection') {
      loadActivatedKH()
    }
    if (session?.state === 'showing_diff') {
      loadDiff()
    }
  }, [session?.state])

  const state: SessionState = (session?.state as SessionState) || 'idle'
  const inferResult = session?.inference_result as Record<string, unknown> | undefined
  const testResult = session?.test_result as Record<string, unknown> | undefined

  const patchedCount = Object.keys(patchResults).length

  const handleSubmitQuestion = () => {
    if (!question.trim()) return
    submitQuestion(question.trim())
  }

  const handleEval = (errorType: ErrorType, notes: string) => {
    submitEval(errorType, notes)
  }

  const handleCorrection = (answer: string, reasoning: string) => {
    submitCorrection(answer, reasoning)
  }

  const handleRollback = (target: RollbackTarget) => {
    rollback(target)
  }

  const handleConfirmPatches = () => {
    confirmAllPatches()
  }

  const handleSetKnowledgeDirs = () => {
    const dirs = kdInput.split('\n').map((s) => s.trim()).filter(Boolean)
    if (dirs.length === 0) return
    createSession(dirs)
    setShowKdConfig(false)
  }

  const handleAddNewKH = async (knowhowJson: Record<string, unknown>) => {
    await addNewKH(knowhowJson)
    setShowAddPanel(false)
  }

  return (
    <div className="flex flex-col h-[calc(100vh-57px)]">
      <StepNavigator currentState={state} />

      {error && (
        <div className="mx-4 mt-2 px-4 py-2 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between">
          <span className="text-sm text-red-700">{error}</span>
          <button onClick={clearError} className="text-red-400 hover:text-red-600 text-sm">
            关闭
          </button>
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Q&A + controls */}
        <div className="w-1/2 border-r border-gray-200 flex flex-col overflow-y-auto p-4 space-y-4">
          {/* Knowledge dir warning */}
          {!hasKnowledgeDirs && (
            <div className="px-4 py-3 bg-amber-50 border border-amber-300 rounded-lg">
              <div className="flex items-start gap-2">
                <span className="text-amber-600 text-lg leading-none">⚠</span>
                <div className="flex-1 space-y-2">
                  <p className="text-sm text-amber-800 font-medium">
                    当前未配置知识目录，推理将无法检索到任何知识块。
                  </p>
                  {!showKdConfig ? (
                    <button
                      onClick={() => setShowKdConfig(true)}
                      className="text-xs text-amber-700 underline hover:text-amber-900"
                    >
                      手动指定知识目录路径
                    </button>
                  ) : (
                    <div className="space-y-2">
                      <textarea
                        value={kdInput}
                        onChange={(e) => setKdInput(e.target.value)}
                        placeholder={'每行一个知识目录的绝对路径，例如：\nC:\\project\\extraction\\qa_know_how_build\\knowledge\\建筑行业'}
                        rows={3}
                        className="w-full px-2 py-1.5 text-xs border border-amber-300 rounded bg-white
                          focus:outline-none focus:ring-1 focus:ring-amber-500 resize-none"
                      />
                      <div className="flex gap-2">
                        <button
                          onClick={handleSetKnowledgeDirs}
                          disabled={!kdInput.trim()}
                          className="px-3 py-1 bg-amber-600 text-white rounded text-xs font-medium
                            hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          确认并重建会话
                        </button>
                        <button
                          onClick={() => setShowKdConfig(false)}
                          className="px-3 py-1 bg-gray-200 text-gray-600 rounded text-xs hover:bg-gray-300"
                        >
                          取消
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Question input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">提问</label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="输入问题..."
              rows={3}
              disabled={state !== 'idle' && state !== 'saved'}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none
                focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50"
            />
            <button
              onClick={handleSubmitQuestion}
              disabled={loading || !question.trim() || (state !== 'idle' && state !== 'saved')}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium
                hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading && state === 'inferring' ? '推理中...' : '提交推理'}
            </button>
          </div>

          {/* Model answer display */}
          {inferResult && state !== 'idle' && state !== 'inferring' && (
            inferResult.final_answer ? (
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">模型回答</label>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <pre className="text-sm text-gray-800 whitespace-pre-wrap">
                    {String(inferResult.final_answer ?? '')}
                  </pre>
                </div>
                <div className="text-xs text-gray-400">
                  {'有效知识块: '}{String(inferResult.total_valid_count ?? 0)}
                  {' | 候选: '}{String(inferResult.retrieval_candidates_count ?? 0)}
                </div>
              </div>
            ) : (
              <div className="px-4 py-3 bg-amber-50 border border-amber-300 rounded-lg space-y-1">
                <p className="text-sm text-amber-800 font-medium">
                  未检索到任何知识块，模型无法生成回答。
                </p>
                <p className="text-xs text-amber-700">
                  {!hasKnowledgeDirs
                    ? '原因：当前会话没有配置知识目录。请在上方设置知识目录路径后重新创建会话。'
                    : '原因：已配置的知识目录中没有匹配该问题的知识块，请检查知识目录内容是否完整（需包含 knowledge.json 文件）。'}
                </p>
              </div>
            )
          )}

          {/* A2UI evaluation */}
          {state === 'awaiting_eval' && (
            <A2UISelector onSelect={handleEval} disabled={loading} />
          )}

          {/* Correction editor */}
          {state === 'awaiting_correction' && (
            <CorrectionEditor onSubmit={handleCorrection} disabled={loading} />
          )}

          {/* Test result */}
          {(state === 'awaiting_test_eval' || state === 'saved') && !!testResult?.final_answer && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700">回测结果</label>
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <pre className="text-sm text-gray-800 whitespace-pre-wrap">
                  {String(testResult.final_answer ?? '')}
                </pre>
              </div>
            </div>
          )}

          {/* Test evaluation actions */}
          {state === 'awaiting_test_eval' && (
            <div className="space-y-4">
              <button
                onClick={save}
                disabled={loading}
                className="w-full px-6 py-3 bg-green-600 text-white rounded-lg text-sm font-medium
                  hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                满意 - 保存版本
              </button>
              <RollbackSelector onSelect={handleRollback} disabled={loading} />
            </div>
          )}

          {state === 'saved' && (
            <div className="px-4 py-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-700 font-medium">
                版本已保存！知识块修正已持久化。
              </p>
            </div>
          )}

          {loading && (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              处理中...
            </div>
          )}
        </div>

        {/* Right panel: Knowledge blocks / Diff */}
        <div className="w-1/2 flex flex-col overflow-y-auto p-4">
          {(state === 'awaiting_kh_selection' || state === 'showing_activated_kh') && (
            <div className="space-y-4">
              {/* Header with confirm button */}
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-gray-800">
                  激活的知识块
                  {patchedCount > 0 && (
                    <span className="ml-2 text-xs text-green-600 font-normal">
                      ({patchedCount} 个已修改)
                    </span>
                  )}
                </h2>
                <button
                  onClick={handleConfirmPatches}
                  disabled={loading || patchedCount === 0}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg text-xs font-medium
                    hover:bg-blue-700 disabled:opacity-50 transition-colors"
                >
                  进入 Diff 对比 ({patchedCount})
                </button>
              </div>

              <p className="text-xs text-gray-500">
                对每个知识块点击「AI 更新」或「手动更新」进行修改，也可点击下方「新增知识块」
              </p>

              {/* Per-block cards */}
              <div className="space-y-3">
                {activatedKH.map((item) => (
                  <KnowHowCard
                    key={item.entry_key}
                    item={item}
                    patchResult={patchResults[item.entry_key]}
                    isLoading={perBlockLoading[item.entry_key]}
                    onAiUpdate={() => aiUpdateBlock(item.entry_key)}
                    onManualSave={(json) => manualUpdateBlock(item.entry_key, json)}
                  />
                ))}
                {activatedKH.length === 0 && (
                  <p className="text-sm text-gray-500">无激活的知识块</p>
                )}
              </div>

              {/* Add new KH */}
              {showAddPanel ? (
                <AddKnowHowPanel
                  loading={loading}
                  onGenerate={generateNewKH}
                  onAdd={handleAddNewKH}
                  onCancel={() => setShowAddPanel(false)}
                />
              ) : (
                <button
                  onClick={() => setShowAddPanel(true)}
                  className="w-full py-3 border-2 border-dashed border-gray-300 rounded-lg
                    text-sm text-gray-500 hover:border-blue-400 hover:text-blue-600 transition-colors"
                >
                  + 新增知识块
                </button>
              )}
            </div>
          )}

          {(state === 'showing_diff' || state === 'testing' || state === 'awaiting_test_eval' || state === 'saved') && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-gray-800">修正前后 Diff 对比</h2>
                {state === 'showing_diff' && (
                  <button
                    onClick={runTest}
                    disabled={loading}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg text-xs font-medium
                      hover:bg-blue-700 disabled:opacity-50 transition-colors"
                  >
                    {loading ? '回测中...' : '确认修正并回测'}
                  </button>
                )}
              </div>
              <DiffViewer items={diffItems} />
            </div>
          )}

          {state === 'awaiting_eval' && !!inferResult?.synthesis_analysis && (
            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-gray-800">推理分析</h2>
              <div className="bg-gray-50 rounded-lg p-4">
                <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                  {String(inferResult.synthesis_analysis ?? '')}
                </pre>
              </div>
            </div>
          )}

          {(state === 'idle' || state === 'inferring') && (
            <div className="flex items-center justify-center h-full text-gray-400 text-sm">
              <p>输入问题并提交推理后，此处将显示知识块信息</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
