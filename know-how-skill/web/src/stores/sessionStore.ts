import { create } from 'zustand'
import type { Session, DiffItem, ActivatedKnowHowItem, ErrorType, RollbackTarget, SinglePatchResult, GenerateNewKHResult } from '../types'
import * as api from '../api/client'

interface SessionStore {
  session: Session | null
  loading: boolean
  error: string
  diffItems: DiffItem[]
  activatedKH: ActivatedKnowHowItem[]
  patchResults: Record<string, SinglePatchResult>
  perBlockLoading: Record<string, boolean>

  createSession: (kd?: string[]) => Promise<void>
  submitQuestion: (question: string) => Promise<void>
  submitEval: (errorType: ErrorType, notes?: string) => Promise<void>
  submitCorrection: (answer: string, reasoning: string) => Promise<void>
  loadActivatedKH: () => Promise<void>
  selectKH: (keys: string[]) => Promise<void>
  aiUpdateBlock: (entryKey: string) => Promise<SinglePatchResult | null>
  manualUpdateBlock: (entryKey: string, patchedJson: Record<string, unknown>) => Promise<SinglePatchResult | null>
  generateNewKH: () => Promise<GenerateNewKHResult | null>
  addNewKH: (knowhowJson: Record<string, unknown>, knowledgeDir?: string) => Promise<SinglePatchResult | null>
  confirmAllPatches: () => Promise<void>
  loadDiff: () => Promise<void>
  runTest: () => Promise<void>
  save: () => Promise<void>
  rollback: (target: RollbackTarget) => Promise<void>
  clearError: () => void
}

export const useSessionStore = create<SessionStore>((set, get) => ({
  session: null,
  loading: false,
  error: '',
  diffItems: [],
  activatedKH: [],
  patchResults: {},
  perBlockLoading: {},

  createSession: async (kd) => {
    set({ loading: true, error: '' })
    try {
      const session = await api.createSession(kd)
      set({ session, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  submitQuestion: async (question) => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.submitInference(s.id, question)
      set({ session, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  submitEval: async (errorType, notes) => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.submitEvaluation(s.id, errorType, notes)
      set({ session, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  submitCorrection: async (answer, reasoning) => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.submitCorrection(s.id, answer, reasoning)
      const activated = await api.getActivatedKnowHow(s.id)
      set({ session, activatedKH: activated, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  loadActivatedKH: async () => {
    const s = get().session
    if (!s) return
    try {
      const items = await api.getActivatedKnowHow(s.id)
      set({ activatedKH: items })
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  selectKH: async (keys) => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.selectKnowHow(s.id, keys)
      set({ session, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  aiUpdateBlock: async (entryKey) => {
    const s = get().session
    if (!s) return null
    set((state) => ({ perBlockLoading: { ...state.perBlockLoading, [entryKey]: true }, error: '' }))
    try {
      const result = await api.aiUpdateKnowHow(s.id, entryKey)
      const session = await api.getSession(s.id)
      set((state) => ({
        session,
        patchResults: { ...state.patchResults, [entryKey]: result },
        perBlockLoading: { ...state.perBlockLoading, [entryKey]: false },
      }))
      return result
    } catch (e: any) {
      set((state) => ({
        perBlockLoading: { ...state.perBlockLoading, [entryKey]: false },
        error: e.message,
      }))
      return null
    }
  },

  manualUpdateBlock: async (entryKey, patchedJson) => {
    const s = get().session
    if (!s) return null
    set((state) => ({ perBlockLoading: { ...state.perBlockLoading, [entryKey]: true }, error: '' }))
    try {
      const result = await api.manualUpdateKnowHow(s.id, entryKey, patchedJson)
      const session = await api.getSession(s.id)
      set((state) => ({
        session,
        patchResults: { ...state.patchResults, [entryKey]: result },
        perBlockLoading: { ...state.perBlockLoading, [entryKey]: false },
      }))
      return result
    } catch (e: any) {
      set((state) => ({
        perBlockLoading: { ...state.perBlockLoading, [entryKey]: false },
        error: e.message,
      }))
      return null
    }
  },

  generateNewKH: async () => {
    const s = get().session
    if (!s) return null
    set({ loading: true, error: '' })
    try {
      const result = await api.generateNewKnowHow(s.id)
      set({ loading: false })
      return result
    } catch (e: any) {
      set({ loading: false, error: e.message })
      return null
    }
  },

  addNewKH: async (knowhowJson, knowledgeDir) => {
    const s = get().session
    if (!s) return null
    set({ loading: true, error: '' })
    try {
      const result = await api.addNewKnowHow(s.id, knowhowJson, knowledgeDir)
      const session = await api.getSession(s.id)
      const activated = await api.getActivatedKnowHow(s.id)
      set((state) => ({
        session,
        activatedKH: activated,
        patchResults: { ...state.patchResults, [result.entry_key]: result },
        loading: false,
      }))
      return result
    } catch (e: any) {
      set({ loading: false, error: e.message })
      return null
    }
  },

  confirmAllPatches: async () => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.confirmPatches(s.id)
      set({ session, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  loadDiff: async () => {
    const s = get().session
    if (!s) return
    try {
      const items = await api.getDiff(s.id)
      set({ diffItems: items })
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  runTest: async () => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.runTest(s.id)
      set({ session, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  save: async () => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.saveVersion(s.id)
      set({ session, loading: false })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  rollback: async (target) => {
    const s = get().session
    if (!s) return
    set({ loading: true, error: '' })
    try {
      const session = await api.rollback(s.id, target)
      set({ session, loading: false, diffItems: [] })
    } catch (e: any) {
      set({ loading: false, error: e.message })
    }
  },

  clearError: () => set({ error: '' }),
}))
