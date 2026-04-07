import axios from 'axios'
import type {
  Session, ActivatedKnowHowItem, DiffItem,
  ErrorType, RollbackTarget, BatchTestResult, VersionInfo,
  SinglePatchResult, GenerateNewKHResult,
} from '../types'

const api = axios.create({ baseURL: '/api' })

export async function createSession(knowledgeDirs?: string[]): Promise<Session> {
  const { data } = await api.post('/sessions', { knowledge_dirs: knowledgeDirs ?? [] })
  return data
}

export async function getSession(id: string): Promise<Session> {
  const { data } = await api.get(`/sessions/${id}`)
  return data
}

export async function submitInference(id: string, question: string): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/infer`, { question })
  return data
}

export async function submitEvaluation(id: string, errorType: ErrorType, notes = ''): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/evaluate`, { error_type: errorType, notes })
  return data
}

export async function submitCorrection(id: string, correctedAnswer: string, correctedReasoning: string): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/correct`, {
    corrected_answer: correctedAnswer,
    corrected_reasoning: correctedReasoning,
  })
  return data
}

export async function getActivatedKnowHow(id: string): Promise<ActivatedKnowHowItem[]> {
  const { data } = await api.get(`/sessions/${id}/activated-knowhow`)
  return data.items
}

export async function selectKnowHow(id: string, entryKeys: string[]): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/select-knowhow`, { entry_keys: entryKeys })
  return data
}

export async function aiUpdateKnowHow(id: string, entryKey: string): Promise<SinglePatchResult> {
  const { data } = await api.post(`/sessions/${id}/ai-update-knowhow`, { entry_key: entryKey })
  return data
}

export async function manualUpdateKnowHow(id: string, entryKey: string, patchedJson: Record<string, unknown>): Promise<SinglePatchResult> {
  const { data } = await api.post(`/sessions/${id}/manual-update-knowhow`, {
    entry_key: entryKey,
    patched_json: patchedJson,
  })
  return data
}

export async function generateNewKnowHow(id: string): Promise<GenerateNewKHResult> {
  const { data } = await api.post(`/sessions/${id}/generate-new-knowhow`)
  return data
}

export async function addNewKnowHow(id: string, knowhowJson: Record<string, unknown>, knowledgeDir = ''): Promise<SinglePatchResult> {
  const { data } = await api.post(`/sessions/${id}/add-new-knowhow`, {
    knowhow_json: knowhowJson,
    knowledge_dir: knowledgeDir,
  })
  return data
}

export async function confirmPatches(id: string): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/confirm-patches`)
  return data
}

export async function getDiff(id: string): Promise<DiffItem[]> {
  const { data } = await api.get(`/sessions/${id}/diff`)
  return data.items
}

export async function runTest(id: string): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/test`)
  return data
}

export async function saveVersion(id: string): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/save`)
  return data
}

export async function rollback(id: string, targetStep: RollbackTarget): Promise<Session> {
  const { data } = await api.post(`/sessions/${id}/rollback`, { target_step: targetStep })
  return data
}

export async function createBatchTest(file: File, knowledgeDirs?: string[]): Promise<BatchTestResult> {
  const form = new FormData()
  form.append('file', file)
  if (knowledgeDirs?.length) {
    form.append('knowledge_dirs', JSON.stringify(knowledgeDirs))
  }
  const { data } = await api.post('/batch-test', form)
  return data
}

export async function getBatchTest(id: string): Promise<BatchTestResult> {
  const { data } = await api.get(`/batch-test/${id}`)
  return data
}

export async function listVersions(knowledgeDir?: string): Promise<VersionInfo[]> {
  const params = knowledgeDir ? { knowledge_dir: knowledgeDir } : {}
  const { data } = await api.get('/versions', { params })
  return data
}

export async function restoreVersion(id: number): Promise<void> {
  await api.post(`/versions/${id}/restore`)
}
