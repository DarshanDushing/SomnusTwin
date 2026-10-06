// src/App.jsx
// SomnusTwin — main application shell.
// Wires together all components into the full dashboard.

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Moon, BarChart2, FlaskConical, History,
  ShieldAlert, Cpu, LayoutDashboard,
} from 'lucide-react'

import { api } from './lib/api'
import { getPatientName, getPatientDisplay } from './lib/patientNames'

// Components
import StarField from './components/StarField'
import SubjectSidebar from './components/SubjectSidebar'
import TwinAvatar from './components/TwinAvatar'
import NightPlayback from './components/NightPlayback'
import StatCards from './components/StatCards'
import ReasonCodes from './components/ReasonCodes'
import WhatIfPanel from './components/WhatIfPanel'
import ClinicianPanel from './components/ClinicianPanel'
import TwinStateHistory from './components/TwinStateHistory'
import ValidationPage from './pages/ValidationPage'

// ─── Disclaimer Banner ───────────────────────────────────────────────────────
function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner" role="banner" aria-label="Research disclaimer">
      <ShieldAlert size={12} style={{ color: 'var(--accent-amber)', flexShrink: 0 }} />
      <span>
        <strong style={{ color: 'var(--accent-amber)' }}>Research prototype</strong>
        {' '}using open clinical datasets (PhysioNet) —{' '}
        <strong>not a medical device</strong>, not for clinical use. For research and demonstration only.
      </span>
    </div>
  )
}

// ─── Top nav tabs ─────────────────────────────────────────────────────────────
const TABS = [
  { id: 'dashboard',   label: 'Dashboard',     icon: LayoutDashboard },
  { id: 'twin-state',  label: 'Twin History',  icon: History },
  { id: 'what-if',     label: 'What-If',       icon: FlaskConical },
  { id: 'validation',  label: 'Validation',    icon: BarChart2 },
]

// ─── Loading skeleton ─────────────────────────────────────────────────────────
function Skeleton({ height = 80 }) {
  return (
    <div style={{
      height,
      borderRadius: 12,
      background: 'var(--surface-alt)',
      animation: 'pulse 1.5s ease infinite',
    }} />
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [subjects, setSubjects] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [nightData, setNightData] = useState(null)
  const [selectedSubject, setSelectedSubject] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [apiStatus, setApiStatus] = useState('unknown')
  const [loadingNight, setLoadingNight] = useState(false)

  // Current minute for twin avatar / stat cards — driven up from NightPlayback
  const [currentMinute, setCurrentMinute] = useState(0)
  const [currentPrediction, setCurrentPrediction] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)

  // ── Bootstrap: check API, load subjects ────────────────────────────────────
  useEffect(() => {
    api.health()
      .then(({ data }) => {
        setApiStatus(data.status === 'ok' ? 'ok' : 'degraded')
      })
      .catch(() => setApiStatus('offline'))

    api.listSubjects()
      .then(({ data }) => {
        const list = data.subjects ?? []
        setSubjects(list)
        // Auto-select first subject
        if (list.length > 0 && !selectedId) {
          handleSelectSubject(list[0].subject_id, list)
        }
      })
      .catch(() => {
        setSubjects([])
        setApiStatus('offline')
      })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Subject selection ───────────────────────────────────────────────────────
  const handleSelectSubject = useCallback((id, subjectList = subjects) => {
    setSelectedId(id)
    setNightData(null)
    setCurrentMinute(0)
    setCurrentPrediction(null)

    const meta = subjectList.find(s => s.subject_id === id) ?? null
    setSelectedSubject(meta)

    setLoadingNight(true)
    api.getNight(id)
      .then(({ data }) => setNightData(data))
      .catch(() => setNightData(null))
      .finally(() => setLoadingNight(false))
  }, [subjects])

  // Current signal row at playhead
  const currentRow = nightData?.signal?.[currentMinute] ?? {}
  const riskScore = currentPrediction?.risk_score ?? 0
  const reasons = currentPrediction?.reasons ?? []

  // ── Layout ──────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', position: 'relative' }}>
      {/* Ambient star background */}
      <StarField />

      {/* ── Sidebar ── */}
      <SubjectSidebar
        subjects={subjects}
        selectedId={selectedId}
        onSelect={(id) => handleSelectSubject(id)}
        apiStatus={apiStatus}
      />

      {/* ── Main area ── */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        position: 'relative',
        zIndex: 1,
      }}>

        {/* ── Top bar ── */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 20px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--bg-elevated)',
          backdropFilter: 'blur(12px)',
          flexShrink: 0,
          flexWrap: 'wrap',
        }}>
          {/* Brand */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 16 }}>
            <Moon size={18} style={{ color: 'var(--primary)' }} />
            <span style={{
              fontFamily: 'Space Grotesk, sans-serif',
              fontWeight: 700,
              fontSize: '1rem',
              color: 'var(--text-primary)',
            }}>
              SomnusTwin
            </span>
            <span style={{
              fontSize: '0.6rem',
              padding: '1px 7px',
              borderRadius: 999,
              background: 'rgba(111,227,208,0.1)',
              color: 'var(--primary)',
              border: '1px solid rgba(111,227,208,0.2)',
              fontWeight: 500,
            }}>
              v1.0 · PhysioNet
            </span>
          </div>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {TABS.map(tab => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  id={`tab-${tab.id}`}
                  className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                  style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                >
                  <Icon size={13} />
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Model status chip (only shown on error) */}
          {(apiStatus === 'offline' || apiStatus === 'degraded') && (
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px', background: 'rgba(217,119,87,0.1)', borderRadius: 999, border: '1px solid rgba(217,119,87,0.3)' }}>
              <Cpu size={12} style={{ color: 'var(--accent-coral)' }} />
              <span style={{
                fontSize: '0.7rem',
                color: 'var(--accent-coral)',
                fontWeight: 600,
              }}>
                {apiStatus === 'offline' ? 'API Offline' : 'API Degraded'}
              </span>
            </div>
          )}
        </div>

        {/* ── Tab content ── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px 60px' }}>
          <AnimatePresence mode="wait">

            {/* ═══ DASHBOARD TAB ═══ */}
            {activeTab === 'dashboard' && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                style={{ display: 'flex', flexDirection: 'column', gap: 16 }}
              >
                {/* ── Row 1: Avatar + Stat cards ── */}
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-start' }}>
                  {/* Digital Twin Avatar */}
                  <div className="card" style={{
                    padding: 20,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 12,
                    minWidth: 280,
                    flex: '0 0 auto',
                  }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                      <span style={{ fontFamily: 'Space Grotesk', fontSize: '0.85rem', fontWeight: 600 }}>
                        Digital Twin
                      </span>
                      {selectedSubject && (
                        <>
                          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {getPatientName(selectedSubject.subject_id)}
                          </span>
                          <span style={{
                            fontSize: '0.6rem', padding: '1px 7px', borderRadius: 999,
                            background: 'rgba(95,212,196,0.08)',
                            color: 'var(--text-secondary)',
                            border: '1px solid var(--border)',
                          }}>
                            {selectedSubject.subject_id}
                          </span>
                        </>
                      )}
                    </div>
                    <TwinAvatar riskScore={riskScore} />
                    {selectedSubject && (
                      <p style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
                        Tonight's twin reflects real PhysioNet signal
                      </p>
                    )}
                  </div>

                  {/* Right column: stat cards + reason codes */}
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12, minWidth: 280 }}>
                    {/* Stat cards */}
                    {nightData ? (
                      <StatCards signal={nightData.signal} currentMinute={currentMinute} />
                    ) : (
                      <div style={{ display: 'flex', gap: 12 }}>
                        <Skeleton height={90} />
                        <Skeleton height={90} />
                        <Skeleton height={90} />
                      </div>
                    )}

                    {/* Reason codes */}
                    <div className="card" style={{ padding: 16 }}>
                      <p style={{
                        fontFamily: 'Space Grotesk',
                        fontSize: '0.82rem',
                        fontWeight: 600,
                        marginBottom: 10,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}>
                        <span style={{
                          width: 8, height: 8, borderRadius: '50%',
                          background: 'var(--primary)',
                          boxShadow: '0 0 8px var(--primary)',
                          display: 'inline-block',
                        }} />
                        SHAP Reason Codes
                      </p>
                      <ReasonCodes reasons={reasons} isAnimated={!isPlaying} />
                    </div>
                  </div>
                </div>

                {/* ── Row 2: Night Playback ── */}
                <div className="card" style={{ padding: 20 }}>
                  {!selectedId ? (
                    <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-secondary)' }}>
                      <Moon size={32} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
                      <p style={{ fontSize: '0.9rem' }}>Select a subject from the sidebar to begin</p>
                    </div>
                  ) : loadingNight ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <Skeleton height={100} />
                      <Skeleton height={100} />
                      <Skeleton height={60} />
                    </div>
                  ) : nightData ? (
                    <NightPlayback
                      subjectId={selectedId}
                      nightData={nightData}
                      onMinuteChange={setCurrentMinute}
                      onPredictionChange={setCurrentPrediction}
                      onPlayStateChange={setIsPlaying}
                    />
                  ) : (
                    <p style={{ color: 'var(--accent-amber)', fontSize: '0.85rem' }}>
                      ⚠ Could not load night data. Run <code>fetch_physionet.py</code> first.
                    </p>
                  )}
                </div>

                {/* ── Row 3: Clinician panel ── */}
                {selectedSubject && (
                  <div className="card" style={{ padding: 20 }}>
                    <ClinicianPanel subject={selectedSubject} riskScore={riskScore} />
                  </div>
                )}
              </motion.div>
            )}

            {/* ═══ TWIN STATE HISTORY TAB ═══ */}
            {activeTab === 'twin-state' && (
              <motion.div
                key="twin-state"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >
                {selectedSubject && (
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                    Patient: <strong style={{ color: 'var(--text-primary)' }}>
                      {getPatientDisplay(selectedSubject.subject_id)}
                    </strong>
                    <span style={{ marginLeft: 8, opacity: 0.6 }}>· Fictitious name, real PhysioNet data</span>
                  </p>
                )}
                <div className="card" style={{ padding: 20 }}>
                  <TwinStateHistory subjectId={selectedId} />
                </div>
              </motion.div>
            )}

            {/* ═══ WHAT-IF TAB ═══ */}
            {activeTab === 'what-if' && (
              <motion.div
                key="what-if"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                style={{ maxWidth: 600 }}
              >
                {selectedSubject && (
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                    Simulating for: <strong style={{ color: 'var(--text-primary)' }}>
                      {getPatientDisplay(selectedSubject.subject_id)}
                    </strong>
                    <span style={{ marginLeft: 8, opacity: 0.6 }}>· Using real EHR profile</span>
                  </p>
                )}
                <div className="card" style={{ padding: 24 }}>
                  <WhatIfPanel ehr={selectedSubject} />
                </div>
              </motion.div>
            )}

            {/* ═══ VALIDATION TAB ═══ */}
            {activeTab === 'validation' && (
              <motion.div
                key="validation"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >
                <ValidationPage />
              </motion.div>
            )}

          </AnimatePresence>
        </div>
      </div>

      {/* ── Always-visible disclaimer ── */}
      <DisclaimerBanner />
    </div>
  )
}
